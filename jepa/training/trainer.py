"""Training loop for :class:`jepa.models.jepa.JepaModel`.

One loop serves both stages (see ``docs/adr/0001-two-stage-training.md``):

* ``stage="jepa"`` optimizes the encoder and predictor with
  :meth:`JepaModel.jepa_loss`; the decoder is frozen and never run.
* ``stage="decoder"`` optimizes the decoder with
  :meth:`JepaModel.decoder_loss`; the encoder and predictor are frozen.

The trainer owns the freezing and the eval/train modes; the model itself has
no stage flags.  A stage's loss function is compiled as a whole, while
checkpoints are always written from the uncompiled model.

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

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import set_seed

from ..models.commitment import PiArgmax
from ..models.jepa import JepaModel
from ..runtime import select_device
from .data import LengthBucketBatchSampler, example_lengths, tokenize_batch

STAGES = ("jepa", "decoder")


@dataclass
class TrainConfig:
    """The few training knobs the loop reads.

    Everything else (optimizer, AMP, compile, LR schedule, data padding) is
    fixed by the module-level constants in ``jepa/training/train.py``.
    """

    output_dir: str | Path
    learning_rate: float
    batch_size: int
    epochs: int = 1
    stage: str = "jepa"
    weight_decay: float = 0.01
    seed: int = 42
    device: str | None = None
    num_workers: int | None = None
    resume: bool = False


def _cpu_count() -> int:
    """Number of CPUs this process may use."""
    if hasattr(os, "process_cpu_count"):  # Python >= 3.13
        count = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return max(1, count or 1)


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

        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.compile_model = compile_model
        self.compile_mode = compile_mode
        self.pad_to_multiple_of = pad_to_multiple_of
        self.log_every = max(1, int(log_every))
        self.num_workers = (
            min(8, _cpu_count())
            if config.num_workers is None
            else max(0, int(config.num_workers))
        )
        # Checkpoint every N optimizer steps as well as at epoch end, so a
        # crash loses at most N steps.  None or 0 disables mid-epoch saves.
        self.save_every_steps = (
            int(save_every_steps) if save_every_steps else None
        )

        self.history_file = f"history_{config.stage}.json"
        self._amp_dtype: torch.dtype | None = None
        # Replaced by the compiled stage loss when compilation succeeds.
        self._loss_fn = self._raw_loss
        # One instance, reused: torch.compile guards on the object, so a fresh
        # PiArgmax() per epoch would recompile the validation graph.
        self._eval_policy = PiArgmax()
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
        if self.config.stage == "jepa":
            return (
                [self.model.encoder, self.model.predictor],
                [self.model.decoder],
            )
        return (
            [self.model.decoder],
            [self.model.encoder, self.model.predictor],
        )

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
    def _raw_loss(self, labels, attention_mask=None, progress=None, policy=None):
        if self.config.stage == "jepa":
            return self.model.jepa_loss(
                labels, attention_mask=attention_mask, progress=progress
            )
        return self.model.decoder_loss(
            labels, attention_mask=attention_mask, policy=policy
        )

    def _losses(
        self, labels, attention_mask, progress=None, policy=None
    ) -> dict[str, torch.Tensor]:
        """Forward one batch; return the total and its components.

        The JEPA stage prefixes every term and diagnostic with ``jepa_``.  The
        decoder stage logs only the unweighted token loss.
        """
        out = self._loss_fn(labels, attention_mask, progress, policy)
        if self.config.stage == "jepa":
            values = {"jepa_loss": out.loss}
            for name, value in (*out.terms.items(), *out.diagnostics.items()):
                values[f"jepa_{name}"] = value
        else:
            values = {"decoder_token": out.terms["token"]}
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
            generator=torch.Generator().manual_seed(self.config.seed),
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
        # Validation is the inference condition: π-argmax, no real future.
        policy = self._eval_policy if self.config.stage == "decoder" else None

        for labels, attention_mask in val_loader:
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            if labels.shape[1] <= horizon:
                continue
            with self._autocast(device):
                self._accumulate(
                    totals, self._losses(labels, attention_mask, progress, policy)
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

        set_seed(config.seed)
        self._configure_backends(device)
        self.model.to(device)
        # Freeze the other stage before building the optimizer so its
        # parameters never enter a group.
        self._apply_stage(training=True)

        # bf16 autocast on CUDA; the CPU path runs in fp32.
        self._amp_dtype = torch.bfloat16 if self.use_amp and device.type == "cuda" else None
        amp_label = "bf16" if self._amp_dtype else "off"
        print(
            f"stage: {stage} | AMP: {amp_label} | compile: {self.compile_model} "
            f"(mode={self.compile_mode}) | pad multiple: {self.pad_to_multiple_of}"
        )

        if self.compile_model:
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
                    self._raw_loss, mode=self.compile_mode, dynamic=True
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
        groups = _parameter_groups(self.model, config.weight_decay)
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
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (
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
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                optimizer.step()
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
