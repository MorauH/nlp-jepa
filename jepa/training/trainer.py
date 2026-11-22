"""Training loop for :class:`jepa.models.jepa.JepaModel`.

One loop serves both stages (see ``docs/adr/0001-two-stage-training.md``):

* ``stage="jepa"`` optimizes the encoder and predictor with
  :meth:`JepaModel.jepa_loss`; the decoder is frozen and never run.
* ``stage="decoder"`` optimizes the decoder with
  :meth:`JepaModel.decoder_loss`; the encoder and predictor are frozen.

The trainer owns the freezing and the eval/train modes; the model itself has
no stage flags.  A stage's loss function is compiled as a whole, while
checkpoints are always written from the uncompiled model.

Engineering settings live here as module-level constants, the single source.
:class:`TrainConfig` carries only the per-run values.  AMP and
``torch.compile`` are enabled only on CUDA, so a CPU run is always eager fp32.

Every save also writes ``trainer_state_<stage>.pt`` (optimizer, scheduler,
position, RNG).  ``TrainConfig.resume`` continues from it: the schedule is not
restarted and the interrupted epoch is replayed only to skip consumed batches.
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
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import set_seed

from ..models.commitment import PiArgmax, Policy
from ..models.jepa import JepaModel
from ..models.losses import LossOutput
from ..runtime import select_device
from .data import LengthBucketBatchSampler, example_lengths, tokenize_batch

STAGES = ("jepa", "decoder")

# ---------------------------------------------------------------------------
# Engineering settings: the single source for constants the loop reads.
# ---------------------------------------------------------------------------
# bf16 autocast and torch.compile apply only on CUDA; CPU runs stay eager fp32
# so the test suite never needs a switch.
USE_AMP = True
COMPILE_MODEL = True
COMPILE_MODE = "default"
# Pad every batch to a multiple of this many tokens: aligned tensor-core tiles
# and fewer distinct compiled shapes.
PAD_TO_MULTIPLE_OF = 64
# Steps between tqdm updates.  Each update synchronizes the GPU.
LOG_EVERY = 10
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
SEED = 42
# Linear warmup for the first WARMUP_FRACTION of steps, then cosine decay to
# LR_FLOOR_FRACTION of the peak.
WARMUP_FRACTION = 0.01
LR_FLOOR_FRACTION = 0.1
# Checkpoint every N optimizer steps as well as at epoch end.  None or 0
# disables mid-epoch saves.
SAVE_EVERY_STEPS = 2000


def _cpu_count() -> int:
    """Number of CPUs this process may use."""
    if hasattr(os, "process_cpu_count"):  # Python >= 3.13
        count = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return max(1, count or 1)


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


def _parameter_groups(model, weight_decay: float):
    """Split trainable parameters into decayed and non-decayed groups.

    Weight decay must not touch biases or normalization scales: decaying an
    RMSNorm weight collapses the residual stream.
    """
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _jepa_loss(model, labels, attention_mask, progress, policy):
    return model.jepa_loss(labels, attention_mask=attention_mask, progress=progress)


def _decoder_loss(model, labels, attention_mask, progress, policy):
    return model.decoder_loss(labels, attention_mask=attention_mask, policy=policy)


# Stateless; one instance keeps torch.compile from recompiling the validation
# graph (it guards on the object).
_EVAL_POLICY = PiArgmax()


@dataclass(frozen=True)
class _Stage:
    """The per-stage differences the loop reads, looked up once.

    modules:          model attributes this stage trains; the rest freeze.
    loss:             ``(model, labels, attention_mask, progress, policy)``.
    prefix:           log-key prefix.
    logs_total:       log the weighted total under ``<prefix>loss``.
    token_weighted:   validation perplexity = exp(token-weighted mean loss).
    validation_policy: commit rule for validation (None = stage default).
    """

    modules: tuple[str, ...]
    loss: Callable[..., LossOutput]
    prefix: str
    logs_total: bool
    token_weighted: bool
    validation_policy: Policy | None


_STAGES = {
    "jepa": _Stage(
        modules=("encoder", "predictor"),
        loss=_jepa_loss,
        prefix="jepa_",
        logs_total=True,
        token_weighted=False,
        validation_policy=None,
    ),
    "decoder": _Stage(
        modules=("decoder",),
        loss=_decoder_loss,
        prefix="decoder_",
        logs_total=False,
        token_weighted=True,
        validation_policy=_EVAL_POLICY,
    ),
}


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
        history: dict | None = None,
    ):
        if config.stage not in _STAGES:
            raise ValueError(
                f"stage must be one of {STAGES}; got {config.stage!r}"
            )

        self.model = model
        self.config = config
        self.stage = _STAGES[config.stage]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.num_workers = (
            DEFAULT_NUM_WORKERS
            if config.num_workers is None
            else max(0, int(config.num_workers))
        )
        self.save_every_steps = (
            int(config.save_every_steps) if config.save_every_steps else None
        )

        self.history_file = f"history_{config.stage}.json"
        self._amp_dtype: torch.dtype | None = None
        # Replaced by the compiled stage loss when compilation is enabled.
        self._loss_fn = partial(self.stage.loss, self.model)
        self.history = history if history is not None else {}
        # Set by train(); _save writes them so --resume can continue the run.
        self._optimizer = None
        self._scheduler = None
        self._device: torch.device | None = None
        self._position: dict = {}

    # ------------------------------------------------------------------
    # Stage ownership
    # ------------------------------------------------------------------
    def _stage_modules(self):
        """Return ``(trainable, frozen)`` module lists for this stage."""
        trainable = [getattr(self.model, name) for name in self.stage.modules]
        frozen = [
            getattr(self.model, name)
            for name in ("encoder", "predictor", "decoder")
            if name not in self.stage.modules
        ]
        return trainable, frozen

    def _apply_stage(self, training: bool) -> None:
        """Set requires_grad and train/eval modes owned by the stage.

        There is deliberately no top-level ``model.train(...)``: the commit
        policy is an argument to :meth:`JepaModel.decoder_loss`, not a function
        of the model's mode.  Only the stage modules' modes matter.
        """
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
    def _losses(
        self, labels, attention_mask, progress=None, policy=None
    ) -> dict[str, torch.Tensor]:
        """Forward one batch; return the total and its components.

        Terms and diagnostics are logged under the stage's prefix.  The JEPA
        stage also logs its weighted total as ``jepa_loss``; the decoder's
        total is its token term.
        """
        out = self._loss_fn(labels, attention_mask, progress, policy)
        prefix = self.stage.prefix
        values = {
            f"{prefix}{name}": value
            for name, value in (*out.terms.items(), *out.diagnostics.items())
        }
        if self.stage.logs_total:
            values[f"{prefix}loss"] = out.loss
        values["loss"] = out.loss
        return values

    def _autocast(self, device: torch.device):
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

    def _make_loader(self, dataset, shuffle: bool, device, collate_fn) -> DataLoader:
        kwargs = dict(
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
            # A dedicated generator: creating a loader iterator otherwise draws
            # a base seed from the global RNG, which would desynchronize a
            # resumed run from the uninterrupted schedule (the shapes of the
            # VISReg slices depend on the global stream).
            generator=torch.Generator().manual_seed(SEED),
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 4

        if shuffle:
            # Bucket by length to avoid padding every batch to 1024.
            return DataLoader(
                dataset,
                batch_sampler=LengthBucketBatchSampler(
                    example_lengths(dataset),
                    self.config.batch_size,
                    seed=SEED,
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
            # Accumulate on the GPU and synchronize only when logging.
            totals[name] += value.detach()
        return totals

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
        """Return the average validation losses."""
        self._apply_stage(training=False)
        horizon = self.model.pred_horizon
        totals: dict[str, torch.Tensor] = {}
        n_batches = 0
        # 0-dim tensor, not None: the same graph inputs as training under
        # torch.compile.  1.0 means the final commitment temperature.
        progress = torch.ones((), device=device)
        policy = self.stage.validation_policy
        token_sum = token_count = None

        for labels, attention_mask in val_loader:
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            if labels.shape[1] <= horizon:
                continue
            with self._autocast(device):
                values = self._losses(labels, attention_mask, progress, policy)
                self._accumulate(totals, values)
            if self.stage.token_weighted:
                prefix = self.stage.prefix
                count = values[f"{prefix}tokens"]
                contribution = values[f"{prefix}token"] * count
                token_sum = (
                    contribution if token_sum is None else token_sum + contribution
                )
                token_count = count if token_count is None else token_count + count
            n_batches += 1

        self._apply_stage(training=True)
        if n_batches == 0:
            return {"loss": float("nan")}

        averages = self._means(totals, n_batches)
        if self.stage.token_weighted:
            # One path: perplexity of the token-weighted mean token loss.
            mean_token_loss = (token_sum / token_count.clamp_min(1)).item()
            averages["perplexity"] = math.exp(min(mean_token_loss, 20.0))
        return averages

    def _state_file(self, output_dir: str | Path) -> str:
        return os.path.join(output_dir, f"trainer_state_{self.config.stage}.pt")

    def _set_position(self, *, epoch, epoch_step, global_step, total_steps, totals):
        """Record where the next resume should continue from."""
        self._position = {
            "epoch": int(epoch),
            "epoch_step": int(epoch_step),
            "global_step": int(global_step),
            "total_steps": int(total_steps),
            "epoch_totals": {
                name: value.detach().cpu() for name, value in totals.items()
            },
        }

    @staticmethod
    def _restore_rng(rng: dict, device: torch.device) -> None:
        """Restore the RNG streams saved next to the last step."""
        if "cpu" in rng:
            torch.set_rng_state(rng["cpu"])
        if device.type == "cuda" and "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])

    def _trainer_state(self) -> dict:
        """Everything a resumed run needs besides the model weights."""
        state = {
            **self._position,
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "rng": {"cpu": torch.get_rng_state()},
        }
        if self._device is not None and self._device.type == "cuda":
            state["rng"]["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _load_trainer_state(self, output_dir: str | Path) -> dict:
        path = self._state_file(output_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"--resume needs the trainer state at {path}, but it is "
                "missing. The checkpoint predates trainer-state saving or the "
                "output dir is wrong; re-run without --resume to start over."
            )
        return torch.load(path, map_location="cpu", weights_only=True)

    def _save(self, output_dir: str | Path) -> None:
        """Save the model, this stage's history and the resume state."""
        os.makedirs(output_dir, exist_ok=True)
        # Save the original module, not a compiled wrapper (whose state dict
        # would be prefixed with ``_orig_mod.``).
        self.model.save_pretrained(output_dir)
        with open(os.path.join(output_dir, self.history_file), "w") as handle:
            json.dump(self.history, handle, indent=4)
        if self._optimizer is not None:
            torch.save(self._trainer_state(), self._state_file(output_dir))

    def train(self) -> dict:
        """Train the model and return the loss history."""
        config = self.config
        stage = config.stage
        device = select_device(config.device)
        print(f"Device: {device} | dataloader workers: {self.num_workers}")

        set_seed(SEED)
        self._configure_backends(device)
        self.model.to(device)
        # Freeze the other stage before building the optimizer so its
        # parameters never enter a group.
        self._apply_stage(training=True)

        # bf16 autocast on CUDA; the CPU path runs in fp32.
        self._amp_dtype = (
            torch.bfloat16 if USE_AMP and device.type == "cuda" else None
        )
        amp_label = "bf16" if self._amp_dtype else "off"
        compile_enabled = COMPILE_MODEL and device.type == "cuda"
        print(
            f"stage: {stage} | AMP: {amp_label} | compile: {compile_enabled} "
            f"(mode={COMPILE_MODE}) | pad multiple: {PAD_TO_MULTIPLE_OF}"
        )

        if compile_enabled:
            # ``dynamic=True`` keeps the varying sequence lengths and the
            # variable number of valid tokens from triggering a recompile for
            # every batch shape; ``suppress_errors`` falls back to eager
            # instead of killing a long run.  64 covers every 64-token bucket.
            torch._dynamo.config.recompile_limit = 64
            torch._dynamo.config.cache_size_limit = 64
            torch._dynamo.config.suppress_errors = True
            try:
                # Compile only the stage's loss: the other stage's modules are
                # frozen and must stay out of the graph.
                self._loss_fn = torch.compile(
                    self._loss_fn, mode=COMPILE_MODE, dynamic=True
                )
                print(f"torch.compile: enabled for {stage} loss")
            except Exception as exc:  # pragma: no cover - depends on the backend
                warnings.warn(
                    f"torch.compile failed ({exc!r}); continuing eagerly.",
                    RuntimeWarning,
                )
                self._loss_fn = partial(self.stage.loss, self.model)

        optimizer_kwargs = {}
        if device.type == "cuda":
            # Fused AdamW keeps the whole update in one CUDA kernel.
            optimizer_kwargs["fused"] = True
        groups = _parameter_groups(self.model, WEIGHT_DECAY)
        try:
            optimizer = torch.optim.AdamW(
                groups, lr=config.learning_rate, **optimizer_kwargs
            )
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop("fused", None)
            optimizer = torch.optim.AdamW(
                groups, lr=config.learning_rate, **optimizer_kwargs
            )
        if optimizer_kwargs:
            print("Optimizer: AdamW (fused)")

        collate = partial(
            tokenize_batch,
            tokenizer=self.tokenizer,
            pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
        )
        train_loader = self._make_loader(
            self.train_dataset, shuffle=True, device=device, collate_fn=collate
        )
        val_loader = self._make_loader(
            self.eval_dataset, shuffle=False, device=device, collate_fn=collate
        )

        horizon = self.model.pred_horizon
        num_epochs = max(1, int(config.epochs))

        total_steps = max(1, len(train_loader) * num_epochs)
        warmup_steps = max(1, int(total_steps * WARMUP_FRACTION))

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return LR_FLOOR_FRACTION + (1 - LR_FLOOR_FRACTION) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )

        with warnings.catch_warnings():
            # LambdaLR steps once during construction; the warning is harmless.
            warnings.filterwarnings(
                "ignore", message="Detected call of `lr_scheduler.step"
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        self._optimizer = optimizer
        self._scheduler = scheduler
        global_step = 0
        start_epoch = 0
        skip_batches = 0
        resumed_totals: dict[str, torch.Tensor] = {}
        if config.resume:
            state = self._load_trainer_state(config.output_dir)
            if int(state.get("total_steps", total_steps)) != total_steps:
                warnings.warn(
                    f"resuming with total_steps={total_steps}, but the run "
                    f"started with total_steps={state.get('total_steps')}; the "
                    "LR schedule will not be the one the run started with.",
                    RuntimeWarning,
                )
            # Optimizer before scheduler: its param groups carry the LR the
            # scheduler last set, and the scheduler restores its position.
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = int(state["global_step"])
            start_epoch = int(state["epoch"])
            skip_batches = int(state["epoch_step"])
            resumed_totals = {
                name: value.to(device)
                for name, value in state.get("epoch_totals", {}).items()
            }
            self._restore_rng(state.get("rng", {}), device)
            print(
                f"Resuming: epoch {start_epoch + 1}, step "
                f"{global_step}/{total_steps}, {skip_batches} batches into the "
                "epoch"
            )

        self._device = device
        prefix = self.stage.prefix
        score_key = prefix + ("loss" if self.stage.logs_total else "token")

        for epoch in range(start_epoch, num_epochs):
            self._apply_stage(training=True)
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if isinstance(batch_sampler, LengthBucketBatchSampler):
                batch_sampler.set_epoch(epoch)

            # The sampler is deterministic per epoch, so replaying the epoch
            # and skipping the already-consumed batches lands exactly after the
            # step where the interrupted checkpoint was written.
            resuming_epoch = epoch == start_epoch
            totals: dict[str, torch.Tensor] = (
                dict(resumed_totals) if resuming_epoch else {}
            )
            n_batches = skip_batches if resuming_epoch else 0
            to_skip = skip_batches if resuming_epoch else 0
            skipped = 0
            pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{num_epochs}")

            for labels, attention_mask in pbar:
                labels = labels.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                if labels.shape[1] <= horizon:
                    continue
                if skipped < to_skip:
                    skipped += 1
                    continue

                # 0-dim tensor, so torch.compile treats the annealing progress
                # as a graph input instead of a Python value to guard on.
                progress = torch.tensor(
                    global_step / total_steps, device=device
                )

                with self._autocast(device):
                    values = self._losses(labels, attention_mask, progress)
                    loss = values["loss"]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if MAX_GRAD_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), MAX_GRAD_NORM
                    )
                optimizer.step()
                scheduler.step()

                self._accumulate(totals, values)
                n_batches += 1
                global_step += 1

                if n_batches % LOG_EVERY == 0:
                    means = self._means(totals, n_batches)
                    postfix = {"loss": f"{means['loss']:.4f}"}
                    if score_key in means:
                        postfix[score_key.removeprefix(prefix)] = (
                            f"{means[score_key]:.4f}"
                        )
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
                    self._set_position(
                        epoch=epoch,
                        epoch_step=n_batches,
                        global_step=global_step,
                        total_steps=total_steps,
                        totals=totals,
                    )
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
                parts = (
                    f"{score_key.removeprefix(prefix)} {means[score_key]:.4f}"
                    if score_key in means
                    else ""
                )
                return f"{means['loss']:.4f} ({parts})"

            def describe_terms(means: dict[str, float]) -> str:
                if not self.stage.logs_total:
                    return ""
                return ", ".join(
                    f"{name.removeprefix(prefix)} {value:.4f}"
                    for name, value in means.items()
                    if name.startswith(prefix) and name != score_key
                )

            summary = (
                f"epoch {epoch + 1} [{stage}]: train {describe(train_means)} | "
                f"val {describe(val_means)}"
            )
            if "perplexity" in val_means:
                summary += f" | val_ppl {val_means['perplexity']:.2f}"
            if terms := describe_terms(train_means):
                summary += f"\n  train terms: {terms}"
            if terms := describe_terms(val_means):
                summary += f"\n  val terms:   {terms}"
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
            # The next epoch starts at step 0 with fresh running totals.
            self._set_position(
                epoch=epoch + 1,
                epoch_step=0,
                global_step=global_step,
                total_steps=total_steps,
                totals={},
            )
            self._save(config.output_dir)
        return self.history
