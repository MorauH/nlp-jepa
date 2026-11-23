"""Train one stage of the JEPA stack.

``--stage jepa`` (the default) trains the encoder and predictor with the JEPA
objective.  ``--stage decoder`` trains a decoder against a frozen JEPA loaded
from ``--from``, writing to ``<from>-decoder`` by default.  The two stages are
described in ``docs/adr/0001-two-stage-training.md``.

Run from the repository root::

    python -m jepa train                         # JEPA stage
    python -m jepa train --resume                # continue it
    python -m jepa train --stage decoder         # decoder stage on runs/jepa
    python -m jepa train --stage decoder --resume --epochs 2
    python -m jepa train --config docs/experiments/p1-baseline.json --seed 1

This module is CLI parsing, per-stage defaults, model building/loading and
paths.  Engineering settings live in :mod:`jepa.training.trainer`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import set_seed

from ..config import EncoderConfig, JepaConfig, apply_overrides
from ..data.tokenizer import load_tokenizer
from ..models.jepa import JepaModel
from ..paths import DEFAULT_RUN_DIR, FINEWEB_DIR, TOKENIZER_DIR
from .data import load_datasets
from .trainer import SEED, JepaTrainer, TrainConfig

# ---------------------------------------------------------------------------
# Per-stage defaults
# ---------------------------------------------------------------------------
# JEPA: 24 x <=1024 tokens is ~24k tokens per step; 3 epochs over the ~500M
# token set is ~90k steps.  Peak AdamW LR (1% linear warmup, cosine decay to
# 10%).  Earlier runs found 3e-4 stable at batch 3-6; square-root scaling to
# batch 24 gives 6e-4; we stay at the conservative 4e-4.
STAGE_DEFAULTS = {
    "jepa": {"batch_size": 24, "epochs": 3, "learning_rate": 4e-4},
    # The frozen backbone removes most of the memory cost, so the same batch
    # size is safe; one epoch is usually enough to adapt a fresh decoder.
    "decoder": {"batch_size": 24, "epochs": 1, "learning_rate": 6e-4},
}


def add_dataset_args(
    parser: argparse.ArgumentParser,
    *,
    batch_size: int | None,
    epochs: int | None,
) -> None:
    """Add the data/batch/epoch flags shared by both stages."""
    parser.add_argument(
        "--fineweb-dir",
        type=Path,
        default=FINEWEB_DIR,
        help="Prepared token chunks to train on (the default). Produced by "
        "`python -m jepa prepare-data`.",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Train on a few examples only (smoke test).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=TOKENIZER_DIR,
        help="Tokenizer directory trained by `python -m jepa train-tokenizer`. "
        "Must match the checkpoint's vocabulary.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help="The stage's default learning rate is tuned for its default; "
        "rescale it if you change this.",
    )
    parser.add_argument("--epochs", type=int, default=epochs)


def load_training_tokenizer(tokenizer_dir: Path):
    """Load the tokenizer and cap its sequence length at the model's max."""
    tokenizer = load_tokenizer(tokenizer_dir)
    max_length = EncoderConfig().max_position_embeddings
    tokenizer.model_max_length = min(
        tokenizer.model_max_length or max_length, max_length
    )
    tokenizer("test")  # warm any encoding cache
    return tokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jepa train", description=__doc__
    )
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_DEFAULTS),
        default="jepa",
        help="Which stage to train (default: jepa).",
    )
    parser.add_argument(
        "--from",
        dest="from_dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Decoder stage: JEPA checkpoint to load. A fresh decoder is "
        "attached unless --resume (default: runs/jepa).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the checkpoint, history_<stage>.json and "
        "trainer_state_<stage>.pt (default: runs/jepa for the JEPA stage; "
        "the decoder stage writes to a sibling of --from, e.g. "
        "runs/jepa-decoder).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the model, history and trainer state in the output "
        "dir, resuming the LR schedule and the epoch at the saved step. "
        "Missing trainer_state_<stage>.pt is an error.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON overrides nested like config.json, e.g. "
        "docs/experiments/p1-baseline.json. The decoder stage accepts only "
        "decoder.* keys. Not allowed with --resume.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Seed for model init and the data order (default: {SEED}). "
        "Not allowed with --resume, which keeps the run's seed.",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, ...")
    add_dataset_args(parser, batch_size=None, epochs=None)
    args = parser.parse_args(argv)

    if args.resume and args.config is not None:
        parser.error("--config cannot be combined with --resume: the "
                     "checkpoint's config.json is the run's config.")
    if args.resume and args.seed is not None:
        parser.error("--seed cannot be combined with --resume: the run keeps "
                     "the seed in its trainer state.")
    args.overrides = {}
    if args.config is not None:
        args.overrides = json.loads(args.config.read_text())
        if not isinstance(args.overrides, dict):
            parser.error(f"{args.config} must hold a JSON object")
        if args.stage == "decoder" and set(args.overrides) - {"decoder"}:
            parser.error(
                f"{args.config}: the decoder stage only accepts decoder.* "
                "overrides; the frozen JEPA is defined by --from."
            )
    if args.seed is None:
        args.seed = SEED

    defaults = STAGE_DEFAULTS[args.stage]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.lr is None:
        args.lr = defaults["learning_rate"]
    if args.output_dir is None:
        if args.stage == "jepa":
            args.output_dir = DEFAULT_RUN_DIR
        else:
            # A sibling, never the JEPA checkpoint itself: the decoder stage
            # overwrites its output dir, so the source must stay intact.
            args.output_dir = args.from_dir.parent / f"{args.from_dir.name}-decoder"
    if (
        args.stage == "decoder"
        and not args.resume
        and args.output_dir.resolve() == args.from_dir.resolve()
    ):
        parser.error(
            "--output-dir must differ from --from when starting a fresh "
            "decoder stage: it writes the full model, history and trainer "
            "state over the JEPA checkpoint. Omit --output-dir to use "
            f"{args.from_dir.name}-decoder next to --from."
        )
    return args


def build_model(tokenizer, overrides: dict | None = None) -> JepaModel:
    """Build a fresh JEPA + decoder model from the config defaults."""
    encoder_config = EncoderConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    config = JepaConfig.from_encoder(encoder_config)
    return JepaModel(apply_overrides(config, overrides or {}))


def load_model(args, tokenizer) -> JepaModel:
    """The stage's model: resumed, fresh, or a fresh decoder on ``--from``."""
    if args.resume:
        # The decoder stage resumes the model in its own output dir, which
        # holds the frozen JEPA plus the partially trained decoder.
        return JepaModel.from_pretrained(str(args.output_dir))
    if args.stage == "jepa":
        return build_model(tokenizer, args.overrides)
    base = JepaModel.from_pretrained(str(args.from_dir))
    # Fresh decoder weights on the frozen JEPA.
    return JepaModel(
        apply_overrides(base.config, args.overrides),
        encoder=base.encoder,
        predictor=base.predictor,
    )


def main(argv: list[str] | None = None) -> JepaTrainer:
    """Run one stage; return the trainer (its model is what was loaded)."""
    args = parse_args(argv)
    stage = args.stage
    output_dir = args.output_dir
    if args.resume and not (output_dir / "config.json").exists():
        raise FileNotFoundError(
            f"--resume needs a checkpoint in {output_dir}, but it has no "
            "config.json. Pass the run's --output-dir or start without "
            "--resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_training_tokenizer(args.tokenizer)

    # Seed before building the model so fresh weights are reproducible.  The
    # trainer re-seeds its own stream when training starts.  The model is
    # built before the data loads so a bad override fails fast.
    set_seed(args.seed)
    model = load_model(args, tokenizer)

    train_dataset, val_dataset = load_datasets(args.fineweb_dir, args.subset)
    print(
        f"train examples: {len(train_dataset):,} | "
        f"val examples: {len(val_dataset):,}"
    )

    history = None
    history_path = output_dir / f"history_{stage}.json"
    if args.resume and history_path.exists():
        with open(history_path) as handle:
            history = json.load(handle)

    expected_vocab = model.config.encoder.vocab_size
    if expected_vocab != len(tokenizer):
        source = output_dir if (stage == "jepa" or args.resume) else args.from_dir
        raise ValueError(
            f"{source} was trained with vocab_size={expected_vocab}, but "
            f"{args.tokenizer} has {len(tokenizer)} tokens. Pass the matching "
            "--tokenizer, or train from scratch without --resume."
        )

    print("Model loaded")
    print(f"Stage: {stage}")
    print(f"Output: {output_dir}")
    if args.overrides:
        print(f"Overrides ({args.config}): {json.dumps(args.overrides)}")
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("  encoder:  ", f"{sum(p.numel() for p in model.encoder.parameters()):,}")
    print("  predictor:", f"{sum(p.numel() for p in model.predictor.parameters()):,}")
    print("  decoder:  ", f"{sum(p.numel() for p in model.decoder.parameters()):,}")
    if stage == "decoder":
        print(
            f"Decoder window: {model.decoder.sliding_window} | "
            f"train policy: {model.config.decoder.train_policy} | "
            + ("resumed from checkpoint" if args.resume else "fresh decoder")
        )

    trainer = JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(output_dir),
            stage=stage,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            device=args.device,
            resume=args.resume,
            seed=args.seed,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        history=history,
    )

    trainer.train()
    return trainer


if __name__ == "__main__":
    main()
