"""Training loop for :class:`jepa.models.stack.JepaDecodeModel`.

One loop serves both entry points: joint training (encoder + predictor +
decoder, ``python -m jepa train``) and decoder-only training against a frozen
stack (``python -m jepa train-decoder``).  A frozen model returns no JEPA loss,
so the loop then optimizes the decoder loss alone -- same AMP, compile,
schedule, logging and checkpointing either way.

The decoder consumes detached hints, so its loss only trains the decoder, while
the JEPA loss trains the encoder and predictor.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import set_seed

from ..models.stack import JepaDecodeModel
from ..runtime import available_cpu_count, resolve_amp_dtype, select_device
from .data import LengthBucketBatchSampler, example_lengths, tokenize_batch
from .optim import parameter_groups


@dataclass
class TrainConfig:
    """The few training knobs the loop reads.

    Everything else (optimizer, AMP, compile, LR schedule) is fixed by the
    module-level constants in ``jepa/training/train.py``.
    """

    output_dir: str | Path
    learning_rate: float
    batch_size: int
    epochs: int = 1
    weight_decay: float = 0.01
    seed: int = 42
    device: str | None = None
    # Loss history written next to the checkpoint.  Both entry points save
    # into runs/jepa, so each uses its own file: decoder-only training must
    # not overwrite the joint run's history (which `train --resume` reads).
    history_file: str = "history.json"


class JepaTrainer:
    """Training loop for :class:`JepaDecodeModel`.

    Usage::

        trainer = JepaTrainer(model, config, train_dataset, eval_dataset, tokenizer)
        history = trainer.train()
    """

    def __init__(
        self,
        model: JepaDecodeModel,
        config: TrainConfig,
        train_dataset,
        eval_dataset,
        tokenizer,
        *,
        jepa_loss_weight: float = 1.0,
        decoder_loss_weight: float = 1.0,
        max_grad_norm: float | None = 1.0,
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
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.jepa_loss_weight = jepa_loss_weight
        self.decoder_loss_weight = decoder_loss_weight
        self.max_grad_norm = max_grad_norm

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

        # Set True when the tokenizer pads on the right (jepa/training/train.py
        # does).  A causal encoder then never lets a real token attend to a pad,
        # so the padding mask can be dropped and SDPA uses its fused causal
        # kernel.  Keep False for a left-padding tokenizer.
        self.right_padded = right_padded

        self._amp_dtype: torch.dtype | None = None
        self._scaler: torch.amp.GradScaler | None = None

        # Fresh training starts empty; every key is created on first use, so a
        # frozen run has no fake JEPA entries.
        self.history = history if history is not None else {}

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
        self, dataset, shuffle: bool, device, collate_fn
    ) -> DataLoader:
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
                    self.config.batch_size,
                    seed=self.config.seed,
                ),
                **kwargs,
            )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            **kwargs,
        )

    def _losses(self, labels, attention_mask) -> dict[str, torch.Tensor]:
        """Forward one batch; return the weighted total and its components.

        A frozen JEPA part reports ``loss=None`` and a skipped decoder returns
        ``decoder_out=None``: the missing term is left out of the total, and
        its key never reaches the history.
        """
        jepa_out, decoder_out = self.model(
            labels=labels,
            attention_mask=attention_mask,
            training=True,
            right_padded=self.right_padded,
            loss_chunk_size=self.decoder_loss_chunk_size,
        )
        values = {}
        terms = []
        if jepa_out["loss"] is not None:
            values["jepa_loss"] = jepa_out["loss"]
            terms.append(self.jepa_loss_weight * jepa_out["loss"])
        if decoder_out is not None:
            values["decoder_loss"] = decoder_out["loss"]
            terms.append(self.decoder_loss_weight * decoder_out["loss"])
        values["loss"] = sum(terms[1:], terms[0])
        return values

    @staticmethod
    def _means(totals: dict[str, torch.Tensor], n: int) -> dict[str, float]:
        """Average the per-key totals with a single device synchronization."""
        names = list(totals)
        if not names:  # every batch was shorter than pred_horizon
            return {"loss": float("nan")}
        values = torch.stack(
            [totals[name] / max(1, n) for name in names]
        ).tolist()
        return dict(zip(names, values))

    def _accumulate(
        self, totals: dict[str, torch.Tensor], values: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        for name, value in values.items():
            if name not in totals:
                totals[name] = torch.zeros((), device=value.device)
            # Accumulate on the GPU and only synchronize when logging; calling
            # .item() every step stalls the pipeline.
            totals[name] += value.detach()
        return totals

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
        """Return the average validation losses."""
        self.model.eval()
        horizon = getattr(self.model, "pred_horizon", 1)
        totals: dict[str, torch.Tensor] = {}
        n_batches = 0

        for labels, attention_mask in val_loader:
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            # The trajectory decoder needs at least k + 1 tokens.
            if labels.shape[1] <= horizon:
                continue

            with self._autocast(device):
                self._accumulate(
                    totals, self._losses(labels, attention_mask)
                )
            n_batches += 1

        self.model.train()

        if n_batches == 0:
            return {"loss": float("nan")}

        averages = self._means(totals, n_batches)
        if "decoder_loss" in averages:  # absent in JEPA-only training
            averages["perplexity"] = math.exp(min(averages["decoder_loss"], 20.0))
        return averages

    def train(self) -> dict:
        """Train the model and return the loss history."""
        config = self.config
        device = select_device(config.device)
        print(f"Device: {device} | dataloader workers: {min(8, available_cpu_count())}")

        set_seed(config.seed)
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
        # ``parameter_groups`` skips frozen parameters, so a frozen run only
        # optimizes the decoder.
        param_groups = parameter_groups(self._raw_model, config.weight_decay)
        try:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                **optimizer_kwargs,
            )
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                **optimizer_kwargs,
            )
        if optimizer_kwargs:
            print("Optimizer: AdamW (fused)")

        collate = partial(
            tokenize_batch,
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        train_loader = self._make_loader(
            self.train_dataset, shuffle=True, device=device, collate_fn=collate
        )
        val_loader = self._make_loader(
            self.eval_dataset, shuffle=False, device=device, collate_fn=collate
        )

        horizon = getattr(self.model, "pred_horizon", 1)
        num_epochs = max(1, int(config.epochs))

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
        criterion = getattr(self._raw_model, "criterion", None)
        if not hasattr(criterion, "set_anneal_progress"):
            criterion = None

        for epoch in range(num_epochs):
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if isinstance(batch_sampler, LengthBucketBatchSampler):
                batch_sampler.set_epoch(epoch)
            totals: dict[str, torch.Tensor] = {}
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

                # Annealed winner-takes-all temperature (no-op unless
                # VISRegConfig.mh_temperature_start is set).  Updates a buffer
                # in place, so it does not trigger torch.compile recompiles.
                if criterion is not None:
                    criterion.set_anneal_progress(global_step / total_steps)

                with self._autocast(device):
                    values = self._losses(labels, attention_mask)
                    loss = values["loss"]

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

                self._accumulate(totals, values)
                n_batches += 1
                global_step += 1

                if n_batches % self.log_every == 0:
                    means = self._means(totals, n_batches)
                    postfix = {"loss": f"{means['loss']:.4f}"}
                    if "jepa_loss" in means:
                        postfix["jepa"] = f"{means['jepa_loss']:.4f}"
                    if "decoder_loss" in means:
                        postfix["dec"] = f"{means['decoder_loss']:.4f}"
                    postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                    pbar.set_postfix(postfix)

                if (
                    self.save_every_steps
                    and global_step % self.save_every_steps == 0
                ):
                    # Epoch-level history stays untouched; this records where
                    # the mid-epoch checkpoint was taken and the running train
                    # losses at that point.
                    self.history["last_checkpoint"] = {
                        "epoch": epoch + 1,
                        "epoch_step": n_batches,
                        "global_step": global_step,
                        "total_steps": total_steps,
                        **{
                            f"train_{name}": value
                            for name, value in self._means(totals, n_batches).items()
                        },
                    }
                    pbar.write(
                        f"step {global_step}/{total_steps}: saving checkpoint "
                        f"to {config.output_dir}"
                    )
                    self._save(config.output_dir)

            train_means = self._means(totals, n_batches)
            val_means = self.evaluate(val_loader, device)
            for name, value in train_means.items():
                self.history.setdefault(f"train_{name}", []).append(value)
            for name, value in val_means.items():
                if name != "perplexity":
                    self.history.setdefault(f"val_{name}", []).append(value)

            def describe(means: dict[str, float]) -> str:
                parts = ", ".join(
                    f"{name.replace('_loss', '')} {value:.4f}"
                    for name, value in means.items()
                    if name not in ("loss", "perplexity")
                )
                return f"{means['loss']:.4f} ({parts})"

            summary = (
                f"epoch {epoch + 1}: train {describe(train_means)} | "
                f"val {describe(val_means)}"
            )
            if "perplexity" in val_means:
                summary += f" | val_ppl {val_means['perplexity']:.2f}"
            print(summary)

            if device.type == "cuda":
                print(
                    f"epoch {epoch + 1} peak VRAM: "
                    f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB allocated, "
                    f"{torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB reserved"
                )

            print(f"Saving to {config.output_dir}.")
            self.history["last_checkpoint"] = {
                "epoch": epoch + 1,
                "epoch_step": n_batches,
                "global_step": global_step,
                "total_steps": total_steps,
                "end_of_epoch": True,
            }
            self._save(config.output_dir)
        return self.history

    def _save(self, output_dir: str | Path) -> None:
        """Save the full stack and each component separately."""
        os.makedirs(output_dir, exist_ok=True)

        # Save the original module, not the OptimizedModule wrapper (whose
        # state dict would be prefixed with ``_orig_mod.``).
        self._raw_model.save_pretrained(output_dir)
        for name in ("encoder", "predictor", "decoder"):
            getattr(self._raw_model, name).save_pretrained(
                os.path.join(output_dir, name)
            )

        with open(os.path.join(output_dir, self.config.history_file), "w") as handle:
            json.dump(self.history, handle, indent=4)
