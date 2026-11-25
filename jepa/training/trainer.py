"""Training loop for :class:`jepa.models.jepa.JepaModel`.

One loop serves both stages (``docs/adr/0002-joint-decoder-gradient.md``):
``stage="jepa"`` trains encoder+predictor, plus the decoder when
``config.decoder_w > 0`` (its gradient reaches the encoder through the hint
``z_t``); with ``decoder_w == 0`` it never runs the decoder.
``stage="decoder"`` trains the decoder against the frozen JEPA.

Engineering settings are the module constants here; :class:`TrainConfig` holds
only per-run values.  AMP and ``torch.compile`` are CUDA-only.  The stage loss
is compiled; checkpoints come from the uncompiled model.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import set_seed

from ..horizon import cycle_horizon, draw_horizon
from ..models.commitment import PiArgmax, Policy
from ..models.jepa import JepaModel
from ..models.losses import LossOutput
from ..runtime import select_device
from .data import LengthBucketBatchSampler, example_lengths, tokenize_batch

STAGES = ("jepa", "decoder")

USE_AMP = True
COMPILE_MODEL = True
COMPILE_MODE = "default"
# Pad to a multiple of this many tokens (aligned tiles, fewer compiled shapes).
PAD_TO_MULTIPLE_OF = 64
LOG_EVERY = 10  # steps between tqdm updates (each one synchronizes the GPU)
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
SEED = 42
# Linear warmup for WARMUP_FRACTION of the steps, then cosine decay to the floor.
WARMUP_FRACTION = 0.01
LR_FLOOR_FRACTION = 0.1
SAVE_EVERY_STEPS = 2000  # None or 0 disables mid-epoch saves


def _cpu_count() -> int:
    """Number of CPUs this process may use."""
    if hasattr(os, "process_cpu_count"):  # Python >= 3.13
        return max(1, os.process_cpu_count() or 1)
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


DEFAULT_NUM_WORKERS = min(8, _cpu_count())


@dataclass
class TrainConfig:
    """Per-run values.  Engineering settings are the module constants above."""

    output_dir: str | Path
    learning_rate: float
    batch_size: int
    epochs: int = 1
    stage: str = "jepa"
    device: str | None = None
    num_workers: int | None = None
    resume: bool = False
    save_every_steps: int | None = SAVE_EVERY_STEPS
    # A resumed run uses the seed saved in its trainer state instead.
    seed: int = SEED


def _parameter_groups(model, weight_decay: float):
    """Decayed and non-decayed parameters (no decay on biases and norms)."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def _lr_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup, then cosine decay to ``LR_FLOOR_FRACTION``."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return LR_FLOOR_FRACTION + (1 - LR_FLOOR_FRACTION) * 0.5 * (
        1 + math.cos(math.pi * progress)
    )


def _jepa_loss(model, labels, attention_mask, horizon, progress, policy):
    return model.jepa_loss(labels, attention_mask=attention_mask, horizon=horizon,
                           progress=progress, policy=policy)


def _decoder_loss(model, labels, attention_mask, horizon, progress, policy):
    return model.decoder_loss(labels, attention_mask=attention_mask,
                              horizon=horizon, policy=policy)


_EVAL_POLICY = PiArgmax()  # one instance; torch.compile guards on the object


@dataclass(frozen=True)
class _Stage:
    """Trained modules, loss callable, log prefix and validation policy."""

    modules: tuple[str, ...]
    loss: Callable[..., LossOutput]
    prefix: str
    validation_policy: Policy | None


_STAGES = {
    "jepa": _Stage(("encoder", "predictor"), _jepa_loss, "jepa_", None),
    "decoder": _Stage(("decoder",), _decoder_loss, "decoder_", _EVAL_POLICY),
}


class JepaTrainer:
    """Training loop for one stage of :class:`JepaModel`."""

    def __init__(self, model: JepaModel, config: TrainConfig, train_dataset,
                 eval_dataset, tokenizer, *, history: dict | None = None):
        if config.stage not in _STAGES:
            raise ValueError(f"stage must be one of {STAGES}; got {config.stage!r}")
        self.model = model
        self.config = config
        self.stage = _STAGES[config.stage]
        if config.stage == "jepa" and model.joint_jepa_stage:
            # Joint JEPA stage: the decoder trains too, and validation uses
            # the inference hint, like the decoder stage.
            self.stage = replace(self.stage, modules=(*self.stage.modules, "decoder"),
                                 validation_policy=_EVAL_POLICY)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.num_workers = (DEFAULT_NUM_WORKERS if config.num_workers is None
                            else max(0, int(config.num_workers)))
        self.save_every_steps = (int(config.save_every_steps)
                                 if config.save_every_steps else None)
        self.history_file = f"history_{config.stage}.json"
        self.history = history if history is not None else {}
        self.seed = config.seed
        self._amp_dtype: torch.dtype | None = None
        self._loss_fn = partial(self.stage.loss, self.model)  # compiled in _prepare
        # Set by train(); _checkpoint writes what --resume needs.
        self._optimizer = None
        self._scheduler = None
        self._device: torch.device | None = None
        self._position: dict = {}
        self._num_epochs = 0
        self._total_steps = 0

    def _stage_modules(self):
        """Return ``(trainable, frozen)`` module lists for this stage."""
        trainable = [getattr(self.model, name) for name in self.stage.modules]
        frozen = [getattr(self.model, name) for name in ("encoder", "predictor", "decoder")
                  if name not in self.stage.modules]
        return trainable, frozen

    def _apply_stage(self, training: bool) -> None:
        """Set the stage modules' modes; the commit policy is a loss argument."""
        trainable, frozen = self._stage_modules()
        for module in trainable:
            module.requires_grad_(True)
            module.train(training)
        for module in frozen:
            module.requires_grad_(False)
            module.eval()

    def _losses(self, labels, attention_mask, horizon, progress=None, policy=None):
        """Forward one batch; return the logged values.

        ``loss`` is the weighted total.  Every term and diagnostic is logged
        under the stage prefix; a loss with several terms also logs its total
        as ``<prefix>loss`` (a single term already is the total).
        """
        out = self._loss_fn(labels, attention_mask, horizon, progress, policy)
        prefix = self.stage.prefix
        values = {f"{prefix}{name}": value
                  for name, value in (*out.terms.items(), *out.diagnostics.items())}
        if len(out.terms) > 1:
            values[f"{prefix}loss"] = out.loss
        values["loss"] = out.loss
        return values

    def _autocast(self, device):
        if self._amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=self._amp_dtype)

    @staticmethod
    def _configure_backends(device: torch.device) -> None:
        """Enable the fast math paths that are safe for this workload."""
        if device.type != "cuda":
            return
        # TF32 for the remaining fp32 matmuls (the VISReg loss, for example).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def _make_loader(self, dataset, shuffle, device, collate_fn) -> DataLoader:
        """Build a train (length-bucketed) or validation loader."""
        kwargs = dict(
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
            # A dedicated generator keeps the loader seed off the global RNG.
            generator=torch.Generator().manual_seed(self.seed),
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        if shuffle:  # bucket by length to avoid padding every batch to 1024
            sampler = LengthBucketBatchSampler(
                example_lengths(dataset), self.config.batch_size, seed=self.seed)
            return DataLoader(dataset, batch_sampler=sampler, **kwargs)
        return DataLoader(dataset, batch_size=self.config.batch_size,
                          shuffle=False, **kwargs)

    @staticmethod
    def _means(totals: dict[str, torch.Tensor], n: int) -> dict[str, float]:
        """Average the per-key totals with a single device synchronization."""
        names = list(totals)
        if not names:  # every batch was too short for the longest horizon
            return {"loss": float("nan")}
        values = torch.stack([totals[name] / max(1, n) for name in names]).tolist()
        return dict(zip(names, values))

    def _accumulate(self, totals, values):
        """Add this batch's values to the running totals, on the GPU."""
        for name, value in values.items():
            if name not in totals:
                totals[name] = torch.zeros((), device=value.device)
            totals[name] += value.detach()
        return totals

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
        """Return the average validation losses (and perplexity if weighted)."""
        self._apply_stage(training=False)
        token_count_key = f"{self.stage.prefix}tokens"
        token_term_key = f"{self.stage.prefix}token"
        totals: dict[str, torch.Tensor] = {}
        n_batches = 0
        # 0-dim tensor: the same graph input as training; 1.0 = final temperature.
        progress = torch.ones((), device=device)
        policy = self.stage.validation_policy
        token_sum = token_count = None
        horizon_range = self.model.config.horizon_range
        for index, (labels, attention_mask) in enumerate(val_loader):
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            if labels.shape[1] <= horizon_range[1]:
                continue
            # Cycled, not drawn: every epoch validates on the same k values.
            horizon = torch.tensor(cycle_horizon(horizon_range, index), device=device)
            with self._autocast(device):
                values = self._losses(labels, attention_mask, horizon, progress, policy)
                self._accumulate(totals, values)
            # A ``tokens`` diagnostic means the ``token`` term is a per-token
            # mean: weight it by its token count for an exact perplexity.  In
            # the joint JEPA stage the total also holds the JEPA terms.
            if token_count_key in values:
                count = values[token_count_key]
                contribution = values[token_term_key] * count
                token_sum = contribution if token_sum is None else token_sum + contribution
                token_count = count if token_count is None else token_count + count
            n_batches += 1
        self._apply_stage(training=True)
        if n_batches == 0:
            return {"loss": float("nan")}
        averages = self._means(totals, n_batches)
        if token_sum is not None:
            mean_token_loss = (token_sum / token_count.clamp_min(1)).item()
            averages["perplexity"] = math.exp(min(mean_token_loss, 20.0))
        return averages

    def _state_file(self, output_dir) -> str:
        return os.path.join(output_dir, f"trainer_state_{self.config.stage}.pt")

    @staticmethod
    def _restore_rng(rng: dict, device: torch.device) -> None:
        """Restore the RNG streams saved next to the last step."""
        if "cpu" in rng:
            torch.set_rng_state(rng["cpu"])
        if device.type == "cuda" and "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])

    def _trainer_state(self) -> dict:
        """Everything a resumed run needs besides the model weights."""
        state = {**self._position, "seed": self.seed,
                 "optimizer": self._optimizer.state_dict(),
                 "scheduler": self._scheduler.state_dict(),
                 "rng": {"cpu": torch.get_rng_state()}}
        if self._device is not None and self._device.type == "cuda":
            state["rng"]["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _load_trainer_state(self, output_dir) -> dict:
        path = self._state_file(output_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"--resume needs the trainer state at {path}, but it is "
                "missing. The checkpoint predates trainer-state saving or the "
                "output dir is wrong; re-run without --resume to start over."
            )
        return torch.load(path, map_location="cpu", weights_only=True)

    def _save(self, output_dir) -> None:
        """Save the model, this stage's history and the resume state."""
        os.makedirs(output_dir, exist_ok=True)
        # Save the original module, not a compiled wrapper (whose state dict
        # would be prefixed with ``_orig_mod.``).
        self.model.save_pretrained(output_dir)
        with open(os.path.join(output_dir, self.history_file), "w") as handle:
            json.dump(self.history, handle, indent=4)
        if self._optimizer is not None:
            torch.save(self._trainer_state(), self._state_file(output_dir))

    def _checkpoint(self, *, epoch, epoch_step, global_step, totals,
                    end_of_epoch=False) -> None:
        """Record where a resume continues, then save.

        ``epoch`` is 0-based.  The resume position points at the next batch to
        train: the start of the next epoch after an end-of-epoch save.  The
        history's ``last_checkpoint`` reports the epoch 1-based, for people.
        """
        self._position = {
            "epoch": epoch + 1 if end_of_epoch else epoch,
            "epoch_step": 0 if end_of_epoch else epoch_step,
            "global_step": global_step,
            "total_steps": self._total_steps,
            "epoch_totals": {} if end_of_epoch else
                {name: value.detach().cpu() for name, value in totals.items()},
        }
        checkpoint = {"epoch": epoch + 1, "epoch_step": epoch_step,
                      "global_step": global_step, "total_steps": self._total_steps}
        if end_of_epoch:
            checkpoint["end_of_epoch"] = True
        else:
            # Epoch history stays untouched; record the running train means.
            checkpoint.update({f"train_{name}": value
                               for name, value in self._means(totals, epoch_step).items()})
        self.history["last_checkpoint"] = checkpoint
        self._save(self.config.output_dir)

    def _prepare(self, device: torch.device) -> None:
        """Seed, backends, stage modes, AMP dtype and the compiled stage loss."""
        set_seed(self.seed)
        self._device = device
        self._configure_backends(device)
        self.model.to(device)
        # Freeze the other stage before the optimizer groups are built.
        self._apply_stage(training=True)
        self._amp_dtype = torch.bfloat16 if USE_AMP and device.type == "cuda" else None
        compile_on = COMPILE_MODEL and device.type == "cuda"
        print(f"stage: {self.config.stage} | AMP: {'bf16' if self._amp_dtype else 'off'}"
              f" | compile: {compile_on} (mode={COMPILE_MODE})"
              f" | pad multiple: {PAD_TO_MULTIPLE_OF}")
        if compile_on:
            self._compile_loss()

    def _compile_loss(self) -> None:
        """Compile the stage loss, falling back to eager on failure."""
        # dynamic=True avoids a recompile per sequence length or token count.
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = True
        try:
            # Only the stage's loss: the frozen stage stays out of the graph.
            self._loss_fn = torch.compile(self._loss_fn, mode=COMPILE_MODE, dynamic=True)
            print(f"torch.compile: enabled for {self.config.stage} loss")
        except Exception as exc:  # pragma: no cover - depends on the backend
            warnings.warn(f"torch.compile failed ({exc!r}); continuing eagerly.",
                          RuntimeWarning)
            self._loss_fn = partial(self.stage.loss, self.model)

    def _build_optimizer(self):
        """AdamW on the trainable parameters, fused on CUDA."""
        kwargs = {"fused": True} if self._device.type == "cuda" else {}
        groups = _parameter_groups(self.model, WEIGHT_DECAY)
        try:
            optimizer = torch.optim.AdamW(groups, lr=self.config.learning_rate, **kwargs)
        except (TypeError, RuntimeError):
            kwargs = {}
            optimizer = torch.optim.AdamW(groups, lr=self.config.learning_rate)
        if kwargs:
            print("Optimizer: AdamW (fused)")
        return optimizer

    def _build_scheduler(self, optimizer):
        """Linear warmup, then cosine decay, planned over the run's steps."""
        warmup_steps = max(1, int(self._total_steps * WARMUP_FRACTION))
        lr_lambda = partial(_lr_factor, warmup_steps=warmup_steps,
                            total_steps=self._total_steps)
        with warnings.catch_warnings():
            # LambdaLR steps once during construction; the warning is harmless.
            warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step")
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _restore(self, state, optimizer, scheduler):
        """Apply the resume state; return ``(global_step, epoch, skipped, totals)``."""
        if int(state.get("total_steps", self._total_steps)) != self._total_steps:
            warnings.warn(
                f"resuming with total_steps={self._total_steps}, but the run "
                f"started with total_steps={state.get('total_steps')}; the "
                "LR schedule will not be the one the run started with.",
                RuntimeWarning,
            )
        # Optimizer before scheduler: its groups hold the LR the scheduler set.
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        skip_batches = int(state["epoch_step"])
        resumed_totals = {name: value.to(self._device)
                          for name, value in state.get("epoch_totals", {}).items()}
        self._restore_rng(state.get("rng", {}), self._device)
        print(f"Resuming: epoch {start_epoch + 1}, step {global_step}/"
              f"{self._total_steps}, {skip_batches} batches into the epoch")
        return global_step, start_epoch, skip_batches, resumed_totals

    def _train_epoch(self, loader, optimizer, scheduler, *, epoch, global_step,
                     skip_batches=0, totals=None):
        """Run one epoch; return ``(totals, batches, global_step)``.

        A resumed epoch skips the ``skip_batches`` batches it already trained
        (the sampler order is deterministic per epoch) and continues the
        running ``totals`` saved with them.
        """
        device = self._device
        self._apply_stage(training=True)
        batch_sampler = getattr(loader, "batch_sampler", None)
        if isinstance(batch_sampler, LengthBucketBatchSampler):
            batch_sampler.set_epoch(epoch)
        horizon_range = self.model.config.horizon_range
        totals = dict(totals or {})
        n_batches = to_skip = skip_batches
        skipped = 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{self._num_epochs}")
        for labels, attention_mask in pbar:
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            if labels.shape[1] <= horizon_range[1]:
                continue
            if skipped < to_skip:
                skipped += 1
                continue
            # 0-dim tensors, so torch.compile treats the annealing progress and
            # the batch's k as graph inputs instead of Python values to guard on.
            horizon = torch.tensor(draw_horizon(horizon_range, self.seed, global_step),
                                   device=device)
            progress = torch.tensor(global_step / self._total_steps, device=device)
            with self._autocast(device):
                values = self._losses(labels, attention_mask, horizon, progress)
                loss = values["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if MAX_GRAD_NORM is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            self._accumulate(totals, values)
            n_batches += 1
            global_step += 1
            if n_batches % LOG_EVERY == 0:
                self._update_postfix(pbar, totals, n_batches, scheduler)
            if self.save_every_steps and global_step % self.save_every_steps == 0:
                pbar.write(f"step {global_step}/{self._total_steps}: saving "
                           f"checkpoint to {self.config.output_dir}")
                self._checkpoint(epoch=epoch, epoch_step=n_batches,
                                 global_step=global_step, totals=totals)
        return totals, n_batches, global_step

    @staticmethod
    def _update_postfix(pbar, totals, n_batches, scheduler) -> None:
        """Show the running mean loss and the LR in the progress bar."""
        loss = totals["loss"] / max(1, n_batches)
        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")

    def _describe_terms(self, means: dict[str, float]) -> str:
        """Per-term/diagnostic breakdown, for a loss with several terms."""
        prefix = self.stage.prefix
        total_key = f"{prefix}loss"
        if total_key not in means:
            return ""
        return ", ".join(f"{name.removeprefix(prefix)} {value:.4f}"
                         for name, value in means.items()
                         if name.startswith(prefix) and name != total_key)

    def _record_epoch(self, epoch, totals, n_batches, val_loader, device) -> None:
        """Append the epoch's means to the history and print the summary."""
        train_means = self._means(totals, n_batches)
        val_means = self.evaluate(val_loader, device)
        for split, means in (("train", train_means), ("val", val_means)):
            for name, value in means.items():
                self.history.setdefault(f"{split}_{name}", []).append(value)
        summary = (f"epoch {epoch + 1} [{self.config.stage}]: "
                   f"train {train_means['loss']:.4f} | val {val_means['loss']:.4f}")
        if "perplexity" in val_means:
            summary += f" | val_ppl {val_means['perplexity']:.2f}"
        if terms := self._describe_terms(train_means):
            summary += f"\n  train terms: {terms}"
        if terms := self._describe_terms(val_means):
            summary += f"\n  val terms:   {terms}"
        print(summary)
        if device.type == "cuda":
            print(f"epoch {epoch + 1} peak VRAM: "
                  f"{torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB allocated, "
                  f"{torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB reserved")

    def train(self) -> dict:
        """Train the model and return the loss history."""
        device = select_device(self.config.device)
        print(f"Device: {device} | dataloader workers: {self.num_workers}")
        state = None
        if self.config.resume:
            state = self._load_trainer_state(self.config.output_dir)
            # States saved before seeds were configurable used SEED.
            self.seed = int(state.get("seed", SEED))
        self.history["seed"] = self.seed
        print(f"Seed: {self.seed}")
        self._prepare(device)
        optimizer = self._build_optimizer()
        collate = partial(tokenize_batch, tokenizer=self.tokenizer,
                          pad_to_multiple_of=PAD_TO_MULTIPLE_OF)
        train_loader = self._make_loader(self.train_dataset, True, device, collate)
        val_loader = self._make_loader(self.eval_dataset, False, device, collate)
        self._num_epochs = max(1, int(self.config.epochs))
        self._total_steps = max(1, len(train_loader) * self._num_epochs)
        scheduler = self._build_scheduler(optimizer)
        self._optimizer, self._scheduler = optimizer, scheduler
        global_step = start_epoch = skip_batches = 0
        resumed_totals = {}
        if state is not None:
            global_step, start_epoch, skip_batches, resumed_totals = self._restore(
                state, optimizer, scheduler)
        for epoch in range(start_epoch, self._num_epochs):
            resuming = epoch == start_epoch
            totals, n_batches, global_step = self._train_epoch(
                train_loader, optimizer, scheduler, epoch=epoch,
                global_step=global_step,
                skip_batches=skip_batches if resuming else 0,
                totals=resumed_totals if resuming else None)
            self._record_epoch(epoch, totals, n_batches, val_loader, device)
            print(f"Saving to {self.config.output_dir}.")
            self._checkpoint(epoch=epoch, epoch_step=n_batches,
                             global_step=global_step, totals=totals,
                             end_of_epoch=True)
        return self.history
