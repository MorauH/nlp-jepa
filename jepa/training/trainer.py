"""Training loop for :class:`jepa.models.stack.JepaDecodeModel`.

It runs the encoder/predictor (JEPA loss) and the decoder (token loss)
together, computes both losses and updates the model in a single backward
pass.

The decoder consumes detached latents, so its loss only trains the decoder,
while the JEPA loss trains the encoder and predictor.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from contextlib import nullcontext
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import set_seed

from ..models.stack import JepaDecodeModel
from ..runtime import available_cpu_count, resolve_amp_dtype
from .data import LengthBucketBatchSampler, example_lengths, tokenize_batch
from .optim import parameter_groups


class JepaTrainer:
    """Minimal training loop for :class:`JepaDecodeModel`.

    Usage::

        trainer = JepaTrainer(model, args, train_dataset, eval_dataset, tokenizer)
        history = trainer.train()
    """

    def __init__(
        self,
        model: JepaDecodeModel,
        args,
        train_dataset,
        eval_dataset,
        processing_class,
        *,
        jepa_loss_weight: float = 1.0,
        decoder_loss_weight: float = 1.0,
        max_grad_norm: float | None = 1.0,
        save_components: bool = True,
        history: dict | None = None,
        use_amp: bool = True,
        amp_dtype: str = "auto",
        compile_model: bool = False,
        compile_mode: str = "default",
        decoder_loss_chunk_size: int | None = None,
        pad_to_multiple_of: int = 64,
        log_every: int = 10,
        right_padded: bool = False,
        save_every_steps: int | None = None,
    ):
        # Keep an uncompiled reference: ``torch.compile`` wraps the module and
        # prefixes state-dict keys with ``_orig_mod.``, so checkpoints must be
        # written from the original model.
        self.model = model
        self._raw_model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

        self.jepa_loss_weight = jepa_loss_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.max_grad_norm = max_grad_norm
        self.save_components = save_components

        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.decoder_loss_chunk_size = decoder_loss_chunk_size
        self.pad_to_multiple_of = pad_to_multiple_of
        self.log_every = max(1, int(log_every))
        # Also checkpoint every N optimizer steps (not only at epoch end), so
        # a crash/timeout loses at most N steps and `verify` can be run on a
        # partially trained model.  None or 0 disables mid-epoch saves.
        self.save_every_steps = (
            int(save_every_steps) if save_every_steps else None
        )

        # Set True when the tokenizer pads on the right (jepa/training/train.py does).
        # For right-aligned padding, a causal encoder never lets a real token
        # attend to a pad, so the padding mask can be dropped and SDPA uses its
        # fused causal kernel.  Keep False for a left-padding tokenizer.
        self.right_padded = right_padded

        self._amp_dtype: torch.dtype | None = None
        self._scaler: torch.amp.GradScaler | None = None

        if history is None:
            history = {
                "train_loss": [],
                "train_jepa_loss": [],
                "train_decoder_loss": [],
                "val_loss": [],
                "val_jepa_loss": [],
                "val_decoder_loss": [],
            }
        self.history = history

        assert args.optim == "adamw_torch"
        assert args.eval_strategy == "epoch"

    def select_device(self) -> torch.device:
        """Return the device to use for training."""
        if self.args.use_cpu:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_amp_dtype(self, device: torch.device) -> torch.dtype | None:
        """Pick the autocast dtype, or None when AMP is off."""
        if not self.use_amp:
            return None
        return resolve_amp_dtype(device, self.amp_dtype)

    def _autocast(self, device: torch.device):
        """Return the autocast context (a no-op when AMP is disabled)."""
        if self._amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=self._amp_dtype)

    @staticmethod
    def _configure_backends(device: torch.device) -> None:
        """Enable the fast math paths that are safe for this workload."""
        if device.type != "cuda":
            return
        # TF32 for any remaining fp32 matmuls (the VISReg loss, for example).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def _make_loader(
        self, dataset, batch_size: int, shuffle: bool, device, collate_fn
    ):
        num_workers = min(8, available_cpu_count())
        kwargs = dict(
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )
        if num_workers > 0:
            kwargs["prefetch_factor"] = 4

        if shuffle:
            # Bucket by length to avoid padding every batch to 1024.
            return DataLoader(
                dataset,
                batch_sampler=LengthBucketBatchSampler(
                    example_lengths(dataset),
                    batch_size,
                    seed=self.args.seed,
                ),
                **kwargs,
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            **kwargs,
        )

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
        """Return the average validation losses."""
        self.model.eval()
        horizon = getattr(self.model, "pred_horizon", 1)

        totals = torch.zeros(3, device=device)
        n_batches = 0

        for labels, attention_mask in val_loader:
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            # The trajectory decoder needs at least k + 1 tokens.
            if labels.shape[1] <= horizon:
                continue

            with self._autocast(device):
                jepa_out, decoder_out = self.model(
                    labels=labels,
                    attention_mask=attention_mask,
                    training=True,  # compute losses even in eval mode
                    right_padded=self.right_padded,
                    loss_chunk_size=self.decoder_loss_chunk_size,
                )

                jepa_loss = jepa_out["loss"]
                decoder_loss = decoder_out["loss"]
                total = (
                    self.jepa_loss_weight * jepa_loss
                    + self.decoder_loss_weight * decoder_loss
                )

            totals += torch.stack(
                [total.detach(), jepa_loss.detach(), decoder_loss.detach()]
            )
            n_batches += 1

        self.model.train()

        if n_batches == 0:
            return {
                "loss": float("nan"),
                "jepa_loss": float("nan"),
                "decoder_loss": float("nan"),
                "perplexity": float("nan"),
            }

        values = (totals / n_batches).tolist()
        averages = dict(zip(("loss", "jepa_loss", "decoder_loss"), values))
        averages["perplexity"] = math.exp(min(averages["decoder_loss"], 20.0))
        return averages

    def train(self) -> dict:
        """Train the model and return the loss history."""
        args = self.args
        device = self.select_device()
        print(f"Device: {device} | dataloader workers: {min(8, available_cpu_count())}")

        set_seed(getattr(args, "seed", 42))
        self._configure_backends(device)
        self.model.to(device)

        self._amp_dtype = self._resolve_amp_dtype(device)
        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self._amp_dtype == torch.float16),
        )
        amp_label = str(self._amp_dtype).replace("torch.", "") if self._amp_dtype else "off"
        print(
            f"AMP: {amp_label} | compile: {self.compile_model} "
            f"(mode={self.compile_mode}) | "
            f"decoder loss chunk: {self.decoder_loss_chunk_size} | "
            f"pad multiple: {self.pad_to_multiple_of}"
        )

        if self.compile_model:
            # ``dynamic=True`` keeps the varying sequence lengths and the
            # variable number of valid tokens in the VISReg loss from
            # triggering a recompile for every batch shape.  ``suppress_errors``
            # makes a backend failure fall back to eager execution instead of
            # killing a long training run.
            # Variable sequence lengths create a few shape variants.  The
            # default limit of 8 makes Dynamo fall back to eager halfway
            # through an epoch; 64 covers every 64-token bucket.
            torch._dynamo.config.recompile_limit = 64
            torch._dynamo.config.cache_size_limit = 64
            torch._dynamo.config.suppress_errors = True
            try:
                self.model = torch.compile(
                    self.model,
                    mode=self.compile_mode,
                    dynamic=True,
                )
                print("torch.compile: enabled")
            except Exception as exc:  # pragma: no cover - depends on the backend
                warnings.warn(
                    f"torch.compile failed ({exc!r}); continuing eagerly.",
                    RuntimeWarning,
                )
                self.model = self._raw_model

        optimizer_kwargs = {}
        if device.type == "cuda":
            # Fused AdamW keeps the whole update in one CUDA kernel.
            optimizer_kwargs["fused"] = True
        param_groups = parameter_groups(
            self._raw_model, getattr(args, "weight_decay", 0.0)
        )
        try:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=args.learning_rate,
                **optimizer_kwargs,
            )
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=args.learning_rate,
                **optimizer_kwargs,
            )
        if optimizer_kwargs:
            print("Optimizer: AdamW (fused)")

        tokenizer = self.processing_class
        collate = partial(
            tokenize_batch,
            tokenizer=tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        train_loader = self._make_loader(
            self.train_dataset,
            args.per_device_train_batch_size,
            shuffle=True,
            device=device,
            collate_fn=collate,
        )
        val_loader = self._make_loader(
            self.eval_dataset,
            args.per_device_eval_batch_size,
            shuffle=False,
            device=device,
            collate_fn=collate,
        )

        horizon = getattr(self.model, "pred_horizon", 1)
        num_epochs = max(1, int(round(args.num_train_epochs)))

        # Linear warmup for the first 1% of steps, then cosine decay to 10%.
        total_steps = max(1, len(train_loader) * num_epochs)
        warmup_steps = max(1, int(total_steps * 0.01))
        eta_min_ratio = 0.1

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        with warnings.catch_warnings():
            # _LRScheduler steps once during construction, before the first
            # optimizer step; the warning is harmless.
            warnings.filterwarnings(
                "ignore", message="Detected call of `lr_scheduler.step"
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        self.model.train()
        global_step = 0

        for epoch in range(num_epochs):
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if isinstance(batch_sampler, LengthBucketBatchSampler):
                batch_sampler.set_epoch(epoch)
            running = torch.zeros(3, device=device)
            n_batches = 0
            pbar = tqdm(
                train_loader,
                desc=f"epoch {epoch + 1}/{num_epochs}",
            )

            for labels, attention_mask in pbar:
                labels = labels.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)

                if labels.shape[1] <= horizon:
                    continue

                with self._autocast(device):
                    jepa_out, decoder_out = self.model(
                        labels=labels,
                        attention_mask=attention_mask,
                        training=True,
                        right_padded=self.right_padded,
                        loss_chunk_size=self.decoder_loss_chunk_size,
                    )

                    jepa_loss = jepa_out["loss"]
                    decoder_loss = decoder_out["loss"]
                    loss = (
                        self.jepa_loss_weight * jepa_loss
                        + self.decoder_loss_weight * decoder_loss
                    )

                optimizer.zero_grad(set_to_none=True)
                self._scaler.scale(loss).backward()

                if self.max_grad_norm is not None:
                    self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                self._scaler.step(optimizer)
                self._scaler.update()
                scheduler.step()

                # Accumulate on the GPU and only synchronize when logging;
                # calling .item() every step stalls the pipeline.
                running += torch.stack(
                    [loss.detach(), jepa_loss.detach(), decoder_loss.detach()]
                )
                n_batches += 1

                if n_batches % self.log_every == 0:
                    mean_loss, mean_jepa, mean_decoder = (
                        running / n_batches
                    ).tolist()
                    pbar.set_postfix(
                        loss=f"{mean_loss:.4f}",
                        jepa=f"{mean_jepa:.4f}",
                        dec=f"{mean_decoder:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )

                global_step += 1
                if (
                    self.save_every_steps
                    and global_step % self.save_every_steps == 0
                ):
                    mean_loss, mean_jepa, mean_decoder = (
                        running / n_batches
                    ).tolist()
                    # Epoch-level history stays untouched; this records where
                    # the mid-epoch checkpoint was taken and the running
                    # train losses at that point.
                    self.history["last_checkpoint"] = {
                        "epoch": epoch + 1,
                        "epoch_step": n_batches,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        "train_loss": mean_loss,
                        "train_jepa_loss": mean_jepa,
                        "train_decoder_loss": mean_decoder,
                    }
                    pbar.write(
                        f"step {global_step}/{total_steps}: saving checkpoint "
                        f"to {args.output_dir}"
                    )
                    self._save(args.output_dir)

            mean_loss, mean_jepa, mean_decoder = (
                running / max(1, n_batches)
            ).tolist()
            train_stats = {
                "loss": mean_loss,
                "jepa_loss": mean_jepa,
                "decoder_loss": mean_decoder,
            }
            val_stats = self.evaluate(val_loader, device)

            self.history.setdefault("train_loss", []).append(train_stats["loss"])
            self.history.setdefault("train_jepa_loss", []).append(
                train_stats["jepa_loss"]
            )
            self.history.setdefault("train_decoder_loss", []).append(
                train_stats["decoder_loss"]
            )
            self.history.setdefault("val_loss", []).append(val_stats["loss"])
            self.history.setdefault("val_jepa_loss", []).append(
                val_stats["jepa_loss"]
            )
            self.history.setdefault("val_decoder_loss", []).append(
                val_stats["decoder_loss"]
            )

            print(
                f"epoch {epoch + 1}: "
                f"train {train_stats['loss']:.4f} "
                f"(jepa {train_stats['jepa_loss']:.4f}, "
                f"decoder {train_stats['decoder_loss']:.4f}) | "
                f"val {val_stats['loss']:.4f} "
                f"(jepa {val_stats['jepa_loss']:.4f}, "
                f"decoder {val_stats['decoder_loss']:.4f}) | "
                f"val_ppl {val_stats['perplexity']:.2f}"
            )

            if device.type == "cuda":
                print(
                    f"epoch {epoch + 1} peak VRAM: "
                    f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB allocated, "
                    f"{torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB reserved"
                )

            print(f"Saving to {args.output_dir}.")
            self.history["last_checkpoint"] = {
                "epoch": epoch + 1,
                "epoch_step": n_batches,
                "global_step": global_step,
                "total_steps": total_steps,
                "end_of_epoch": True,
            }
            self._save(args.output_dir)
        return self.history

    def _save(self, output_dir: str) -> None:
        """Save the full stack and, optionally, each component separately."""
        os.makedirs(output_dir, exist_ok=True)

        # Save the original module, not the OptimizedModule wrapper (whose
        # state dict would be prefixed with ``_orig_mod.``).
        self._raw_model.save_pretrained(output_dir)

        if self.save_components:
            for name in ("encoder", "predictor", "decoder"):
                component = getattr(self._raw_model, name, None)
                if component is not None:
                    component.save_pretrained(os.path.join(output_dir, name))

        with open(os.path.join(output_dir, "history.json"), "w") as handle:
            json.dump(self.history, handle, indent=4)
