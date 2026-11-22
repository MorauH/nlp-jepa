"""Train the full JEPA encoder + predictor + decoder stack.

Builds a :class:`jepa.JepaDecodeModel` from component configs and hands it to
:class:`jepa.JepaTrainer`, which optimizes the VISReg (JEPA) loss and the
decoder token loss together.

This module owns the shared training constants and CLI helpers; the decoder-only
entry point (`jepa/training/train_decoder.py`) imports them and runs the same
trainer against a frozen stack.

Run from the repository root::

    python -m jepa train
    python -m jepa train --resume
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import set_seed

from ..config import EncoderConfig, JepaConfig
from ..data.tokenizer import load_tokenizer
from ..models.stack import JepaDecodeModel
from ..paths import DEFAULT_RUN_DIR, FINEWEB_DIR, TOKENIZER_DIR
from .trainer import JepaTrainer, TrainConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = DEFAULT_RUN_DIR

# --subset trains on this many examples (smoke test).
SUBSET_TRAIN = 4
SUBSET_VAL = 2

# Relative weights of the two objectives in the backward pass.
JEPA_LOSS_WEIGHT = 2.0
DECODER_LOSS_WEIGHT = 1.0

SEED = 42
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
# Defaults for `--batch-size` / `--epochs`.  24 x <=1024 tokens is ~24k tokens
# per step; 3 epochs over the ~500M-token set is ~90k steps.
BATCH_SIZE = 24
EPOCHS = 3
# Peak AdamW LR (1% linear warmup, cosine decay to 10%).  Earlier runs found
# 3e-4 stable at batch 3-6; square-root scaling to batch 24 gives 6e-4.  We
# stay at the conservative end (not 8e-4+).  Rescale (~sqrt) if you change
# BATCH_SIZE.
# 10 batch => 4e-4
# 24 batch => 6e-4
LEARNING_RATE = 4e-4

# Save a checkpoint (full stack + components + history.json, overwriting
# OUTPUT_DIR) every N optimizer steps, in addition to the end of each epoch.
# A crash or SLURM timeout then loses at most N steps, and `verify` can be run
# on the partially trained model.  None disables mid-epoch saves.
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
# cannot compile the model.
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

# Both tokenizers pad on the right. This lets causal attention skip the padding
# mask (Flash Attention); set False if you replace the tokenizer.
RIGHT_PADDED = True


def add_dataset_args(
    parser: argparse.ArgumentParser,
    *,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
) -> None:
    """Add the data/batch/epoch flags shared with `train-decoder`."""
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
        help="LEARNING_RATE is tuned for the default; rescale it if you change this.",
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
        "--resume",
        action="store_true",
        help=f"Continue from the checkpoint in {OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--jepa-only",
        action="store_true",
        help="Train only the encoder + predictor (JEPA loss); the decoder is "
        "never run and is saved unchanged. Train it afterwards with "
        "`python -m jepa train-decoder`.",
    )
    add_dataset_args(parser)
    return parser.parse_args(argv)


def build_model(tokenizer) -> JepaDecodeModel:
    """Build a fresh JEPA + decoder model from the config defaults."""
    encoder_config = EncoderConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    tokenizer.model_max_length = min(
        tokenizer.model_max_length or encoder_config.max_position_embeddings,
        encoder_config.max_position_embeddings,
    )
    return JepaDecodeModel(JepaConfig.from_encoder(encoder_config))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_training_tokenizer(args.tokenizer)
    train_dataset, val_dataset = load_datasets(args.fineweb_dir, args.subset)
    print(
        f"train examples: {len(train_dataset):,} | "
        f"val examples: {len(val_dataset):,}"
    )

    if args.resume:
        model = JepaDecodeModel.from_pretrained(str(OUTPUT_DIR))
        if model.config.encoder.vocab_size != len(tokenizer):
            raise ValueError(
                f"{OUTPUT_DIR} was trained with vocab_size="
                f"{model.config.encoder.vocab_size}, but {args.tokenizer} has "
                f"{len(tokenizer)} tokens. Pass the matching --tokenizer, or "
                "train from scratch without --resume."
            )
        history_path = OUTPUT_DIR / "history.json"
        history = None
        if history_path.exists():
            with open(history_path) as handle:
                history = json.load(handle)
    else:
        model = build_model(tokenizer)
        history = None

    if args.jepa_only:
        model.skip_decoder()

    print("Model loaded")
    if args.jepa_only:
        print("Mode: JEPA only (encoder + predictor); the decoder is not run")
    else:
        print(
            f"Decoder window: {model.decoder.sliding_window} | "
            f"train policy: {model.config.decoder.train_policy}"
        )
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("  encoder:  ", f"{sum(p.numel() for p in model.encoder.parameters()):,}")
    print("  predictor:", f"{sum(p.numel() for p in model.predictor.parameters()):,}")
    print("  decoder:  ", f"{sum(p.numel() for p in model.decoder.parameters()):,}")

    # FineWeb chunks are long, so keep an eye on memory. For batch 20, raise
    # the LR to 6e-4-8e-4.
    trainer = JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(OUTPUT_DIR),
            learning_rate=LEARNING_RATE,
            batch_size=args.batch_size,
            epochs=args.epochs,
            weight_decay=WEIGHT_DECAY,
            seed=SEED,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        jepa_loss_weight=JEPA_LOSS_WEIGHT,
        decoder_loss_weight=DECODER_LOSS_WEIGHT,
        max_grad_norm=MAX_GRAD_NORM,
        history=history,
        use_amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        compile_model=COMPILE_MODEL,
        compile_mode=COMPILE_MODE,
        decoder_loss_chunk_size=DECODER_LOSS_CHUNK_SIZE,
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
        log_every=LOG_EVERY,
        right_padded=RIGHT_PADDED,
        save_every_steps=SAVE_EVERY_STEPS,
    )

    trainer.train()


if __name__ == "__main__":
    main()
