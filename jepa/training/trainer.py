"""Training loop for :class:`jepa.models.jepa.JepaModel`.

One loop serves both stages (see ``docs/adr/0001-two-stage-training.md``):

* ``stage="jepa"`` optimizes the encoder and predictor with
  :meth:`JepaModel.jepa_loss`; the decoder is frozen and never run.
* ``stage="decoder"`` optimizes the decoder with
  :meth:`JepaModel.decoder_loss`; the encoder and predictor are frozen.

The trainer owns the freezing and the eval/train modes; the model itself has
no stage flags.  A stage's loss function is compiled as a whole, while
checkpoints are always written from the uncompiled model.
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

from ..models.jepa import JepaModel
from ..runtime import available_cpu_count, resolve_amp_dtype, select_device
from .data import LengthBucketBatchSampler, example_lengths, tokenize_batch
from .optim import parameter_groups

STAGES = ("jepa", "decoder")


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
    stage: str = "jepa"
    weight_decay: float = 0.01
    seed: int = 42
    device: str | None = None
    # DataLoader workers.  Tests use 0; the default mirrors the historical
    # hard-coded value.
    num_workers: int | None = None


class JepaTrainer:
    """Training loop for one stage of :class:`JepaModel`.

    Usage::

        trainer = JepaTrainer(model, config, train_dataset, eval_dataset, tokenizer)
        history = trainer.train()
    """

    def __init__(
        self,
        model: JepaModel,
        config: TrainConfig,
        train_dataset,
        eval_dataset,
        tokenizer,
        *,
        max_grad_norm: float | None = 1.0,
        history: dict | None = None,
        use_amp: bool = True,
        amp_dtype: str = "auto",
        compile_model: bool = False,
        compile_mode: str = "default",
        pad_to_multiple_of: int = 64,
        log_every: int = 10,
        save_every_steps: int | None = None,
    ):
        if config.stage not in STAGES:
            raise ValueError(
                f"stage must be one of {STAGES}; got {config.stage!r}"
            )

        # Keep an uncompiled reference: checkpoints are always written from the
        # original model.
        self.model = model
        self._raw_model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.max_grad_norm = max_grad_norm

        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.pad_to_multiple_of = pad_to_multiple_of
        self.log_every = max(1, int(log_every))
        self.num_workers = (
            min(8, available_cpu_count())
            if config.num_workers is None
            else max(0, int(config.num_workers))
        )
        # Also checkpoint every N optimizer steps (not only at epoch end), so
        # a crash/timeout loses at most N steps and `measure` can be run on a
        # partially trained model.  None or 0 disables mid-epoch saves.
        self.save_every_steps = (
            int(save_every_steps) if save_every_steps else None
        )

        self.history_file = f"history_{config.stage}.json"

        self._amp_dtype: torch.dtype | None = None
        self._scaler: torch.amp.GradScaler | None = None
        # Replaced by the compiled stage loss when compilation succeeds.
        self._loss_fn = self._raw_loss

        # Fresh training starts empty; every key is created on first use, so a
        # stage never gets the other stage's fake entries.
        self.history = history if history is not None else {}

    # ------------------------------------------------------------------
    # Stage ownership
    # ------------------------------------------------------------------
    def _stage_modules(self):
        """Return ``(trainable, frozen)`` module lists for this stage."""
        if self.config.stage == "jepa":
            return (
                [self._raw_model.encoder, self._raw_model.predictor],
                [self._raw_model.decoder],
            )
        return (
            [self._raw_model.decoder],
            [self._raw_model.encoder, self._raw_model.predictor],
        )

    def _apply_stage(self, training: bool) -> None:
        """Set requires_grad and train/eval modes owned by the stage."""
        # Set the top-level flag too: JepaModel.decoder_loss reads
        # ``self.training`` to pick the commit policy.
        self._raw_model.train(training)
        trainable, frozen = self._stage_modules()
        for module in trainable:
            module.requires_grad_(True)
            module.train(training)
        for module in frozen:
            module.requires_grad_(False)
            module.eval()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _raw_loss(self, labels, attention_mask=None, progress=None):
        if self.config.stage == "jepa":
            return self._raw_model.jepa_loss(
                labels, attention_mask=attention_mask, progress=progress
            )
        return self._raw_model.decoder_loss(
            labels, attention_mask=attention_mask
        )

    def _losses(
        self, labels, attention_mask, progress=None
    ) -> dict[str, torch.Tensor]:
        """Forward one batch; return the total and its components.

        The JEPA stage prefixes every term and diagnostic with ``jepa_``.  The
        decoder stage logs the unweighted token loss and nothing JEPA-related.
        """
        out = self._loss_fn(labels, attention_mask, progress)
        if self.config.stage == "jepa":
            values = {"jepa_loss": out.loss}
            for name, value in out.terms.items():
                values[f"jepa_{name}"] = value
            for name, value in out.diagnostics.items():
                values[f"jepa_{name}"] = value
        else:
            values = {"decoder_token": out.terms["token"]}
        values["loss"] = out.loss
        return values

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
        num_workers = self.num_workers
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
        horizon = self.model.pred_horizon
        totals: dict[str, torch.Tensor] = {}
        n_batches = 0

        for labels, attention_mask in val_loader:
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)

            # Both stages need at least k + 1 tokens.
            if labels.shape[1] <= horizon:
                continue

            with self._autocast(device):
                self._accumulate(
                    totals, self._losses(labels, attention_mask)
                )
            n_batches += 1

        self._apply_stage(training=True)

        if n_batches == 0:
            return {"loss": float("nan")}

        averages = self._means(totals, n_batches)
        if self.config.stage == "decoder":
            averages["perplexity"] = math.exp(
                min(averages["decoder_token"], 20.0)
            )
        return averages

    def train(self) -> dict:
        """Train the model and return the loss history."""
        config = self.config
        stage = config.stage
        device = select_device(config.device)
        print(f"Device: {device} | dataloader workers: {self.num_workers}")

        set_seed(config.seed)
        self._configure_backends(device)
        self.model.to(device)
        # Freeze the other stage before building the optimizer so its
        # parameters never enter a group.
        self._apply_stage(training=True)

        self._amp_dtype = self._resolve_amp_dtype(device)
        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self._amp_dtype == torch.float16),
        )
        amp_label = str(self._amp_dtype).replace("torch.", "") if self._amp_dtype else "off"
        print(
            f"stage: {stage} | AMP: {amp_label} | compile: {self.compile_model} "
            f"(mode={self.compile_mode}) | "
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
                # Compile only the stage's loss: the other stage's modules are
                # frozen and must stay out of the graph.
                self._loss_fn = torch.compile(
                    self._raw_loss,
                    mode=self.compile_mode,
                    dynamic=True,
                )
                print(f"torch.compile: enabled for {stage} loss")
            except Exception as exc:  # pragma: no cover - depends on the backend
                warnings.warn(
                    f"torch.compile failed ({exc!r}); continuing eagerly.",
                    RuntimeWarning,
                )
                self._loss_fn = self._raw_loss

        optimizer_kwargs = {}
        if device.type == "cuda":
            # Fused AdamW keeps the whole update in one CUDA kernel.
            optimizer_kwargs["fused"] = True
        # ``parameter_groups`` skips frozen parameters, so only the stage's
        # modules are optimized.
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

        horizon = self.model.pred_horizon
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
        global_step = 0

        for epoch in range(num_epochs):
            self._apply_stage(training=True)
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

                # Annealed winner-takes-all temperature.  A 0-dim tensor, so
                # torch.compile treats it as a graph input instead of a Python
                # value to guard on.  Only the JEPA stage reads it.
                progress = None
                if stage == "jepa":
                    progress = torch.tensor(
                        global_step / total_steps, device=device
                    )

                with self._autocast(device):
                    values = self._losses(labels, attention_mask, progress)
                    loss = values["loss"]

                optimizer.zero_grad(set_to_none=True)
                self._scaler.scale(loss).backward()

                if self.max_grad_norm is not None:
                    self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._raw_model.parameters(), self.max_grad_norm
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
                    if "decoder_token" in means:
                        postfix["dec"] = f"{means['decoder_token']:.4f}"
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
                self.history.setdefault(f"val_{name}", []).append(value)

            def describe(means: dict[str, float]) -> str:
                parts = ", ".join(
                    f"{name.replace('_loss', '')} {value:.4f}"
                    for name, value in means.items()
                    if name in ("jepa_loss", "decoder_token")
                )
                return f"{means['loss']:.4f} ({parts})"

            def describe_terms(means: dict[str, float]) -> str:
                return ", ".join(
                    f"{name.removeprefix('jepa_')} {value:.4f}"
                    for name, value in means.items()
                    if name.startswith("jepa_") and name != "jepa_loss"
                )

            summary = (
                f"epoch {epoch + 1} [{stage}]: train {describe(train_means)} | "
                f"val {describe(val_means)}"
            )
            if "perplexity" in val_means:
                summary += f" | val_ppl {val_means['perplexity']:.2f}"
            train_terms = describe_terms(train_means)
            if train_terms:
                summary += f"\n  train terms: {train_terms}"
            val_terms = describe_terms(val_means)
            if val_terms:
                summary += f"\n  val terms:   {val_terms}"
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
        """Save the full model checkpoint and this stage's history."""
        os.makedirs(output_dir, exist_ok=True)

        # Save the original module, not a compiled wrapper (whose state dict
        # would be prefixed with ``_orig_mod.``).
        self._raw_model.save_pretrained(output_dir)

        with open(os.path.join(output_dir, self.history_file), "w") as handle:
            json.dump(self.history, handle, indent=4)
