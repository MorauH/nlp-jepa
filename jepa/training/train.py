"""Train one stage of the JEPA stack.

``--stage jepa`` (the default) trains the encoder and predictor with the JEPA
objective.  ``--stage decoder`` trains a decoder against a frozen JEPA loaded
from ``--from``.  The two stages are described in
``docs/adr/0001-two-stage-training.md``.

Run from the repository root::

    python -m jepa train                         # JEPA stage
    python -m jepa train --resume                # continue it
    python -m jepa train --stage decoder         # decoder stage on runs/jepa
    python -m jepa train --stage decoder --resume --epochs 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import set_seed

from ..config import EncoderConfig, JepaConfig
from ..data.tokenizer import load_tokenizer
from ..models.jepa import JepaModel
from ..paths import DEFAULT_RUN_DIR, FINEWEB_DIR, TOKENIZER_DIR
from .trainer import JepaTrainer, TrainConfig

# --subset trains on this many examples (smoke test).
SUBSET_TRAIN = 4
SUBSET_VAL = 2

SEED = 42
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

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

# Save a checkpoint (full model + history_<stage>.json, overwriting the output
# dir) every N optimizer steps, in addition to the end of each epoch.  A crash
# or SLURM timeout then loses at most N steps, and `verify` can be run on the
# partially trained model.  None disables mid-epoch saves.
SAVE_EVERY_STEPS = 2000

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
# bf16 autocast on CUDA. "auto" picks bf16 when supported (L4/Ada, Ampere,
# Blackwell) and falls back to fp16 otherwise.
USE_AMP = True
AMP_DTYPE = "auto"

# torch.compile with dynamic shapes. Compilation adds startup time but is
# amortized quickly on a full run; it is caught and disabled if the backend
# cannot compile the loss.
COMPILE_MODEL = True
COMPILE_MODE = "default"

# Compute the decoder token loss in position chunks so the full
# (batch, sequence, vocab) logits tensor is never materialized. None disables
# chunking.
DECODER_LOSS_CHUNK_SIZE = 256

# Pad every batch to a multiple of this many tokens. Keeps tensor-core tiles
# aligned and limits the number of distinct compiled shapes.
PAD_TO_MULTIPLE_OF = 64

# Steps between tqdm updates. Each update synchronizes the GPU, so batching
# them avoids a per-step stall.
LOG_EVERY = 10


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


def load_datasets(fineweb_dir: Path, subset: bool = False):
    """Load the prepared train/validation chunks (a tiny slice for --subset)."""
    if not fineweb_dir.is_dir():
        raise FileNotFoundError(
            f"{fineweb_dir} not found. "
            "Run `python -m jepa prepare-data` first."
        )
    dataset = load_from_disk(str(fineweb_dir))
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    if subset:
        train_dataset = train_dataset.select(
            range(min(SUBSET_TRAIN, len(train_dataset)))
        )
        val_dataset = val_dataset.select(range(min(SUBSET_VAL, len(val_dataset))))
    return train_dataset, val_dataset


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
        help="Where to write the checkpoint and history_<stage>.json "
        "(default: runs/jepa for both stages; decoder defaults to --from).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the checkpoint in the output dir (JEPA stage) or "
        "keep the decoder already in --from (decoder stage).",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, ...")
    add_dataset_args(parser, batch_size=None, epochs=None)
    args = parser.parse_args(argv)

    defaults = STAGE_DEFAULTS[args.stage]
    if args.batch_size is None:
        args.batch_size = defaults["batch_size"]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.lr is None:
        args.lr = defaults["learning_rate"]
    if args.output_dir is None:
        args.output_dir = DEFAULT_RUN_DIR if args.stage == "jepa" else args.from_dir
    return args


def build_model(tokenizer) -> JepaModel:
    """Build a fresh JEPA + decoder model from the config defaults."""
    encoder_config = EncoderConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    tokenizer.model_max_length = min(
        tokenizer.model_max_length or encoder_config.max_position_embeddings,
        encoder_config.max_position_embeddings,
    )
    return JepaModel(JepaConfig.from_encoder(encoder_config))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(SEED)
    stage = args.stage
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_training_tokenizer(args.tokenizer)
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

    if stage == "jepa":
        if args.resume:
            model = JepaModel.from_pretrained(str(output_dir))
        else:
            model = build_model(tokenizer)
    else:
        base = JepaModel.from_pretrained(str(args.from_dir))
        if args.resume:
            model = base
        else:
            # Fresh decoder weights, same architecture and frozen JEPA.
            model = JepaModel(
                base.config,
                encoder=base.encoder,
                predictor=base.predictor,
            )
        model.decoder_loss_chunk_size = DECODER_LOSS_CHUNK_SIZE

    expected_vocab = model.config.encoder.vocab_size
    if expected_vocab != len(tokenizer):
        source = output_dir if stage == "jepa" else args.from_dir
        raise ValueError(
            f"{source} was trained with vocab_size={expected_vocab}, but "
            f"{args.tokenizer} has {len(tokenizer)} tokens. Pass the matching "
            "--tokenizer, or train from scratch without --resume."
        )

    print("Model loaded")
    print(f"Stage: {stage}")
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
            weight_decay=WEIGHT_DECAY,
            seed=SEED,
            device=args.device,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_grad_norm=MAX_GRAD_NORM,
        history=history,
        use_amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        compile_model=COMPILE_MODEL,
        compile_mode=COMPILE_MODE,
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
        log_every=LOG_EVERY,
        save_every_steps=SAVE_EVERY_STEPS,
    )

    trainer.train()


if __name__ == "__main__":
    main()
