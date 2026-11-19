"""Train the full JEPA encoder + predictor + decoder stack.

Builds a :class:`jepa.JepaDecodeModel` from component configs and hands it to
:class:`jepa.JepaTrainer`, which optimizes the VISReg (JEPA) loss and the
decoder token loss together.

Run from the repository root::

    python -m jepa train
    python -m jepa train --resume
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import TrainingArguments, set_seed

from ..config import (
    DecoderConfig,
    EncoderConfig,
    JepaConfig,
    PredictorConfig,
    VISRegConfig,
)
from ..data.tokenizer import load_tokenizer
from ..models.stack import JepaDecodeModel
from ..paths import DEFAULT_RUN_DIR, FINEWEB_DIR, TOKENIZER_DIR
from .trainer import JepaTrainer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = DEFAULT_RUN_DIR

# ---------------------------------------------------------------------------
# Model shape
# ---------------------------------------------------------------------------
MODEL_MAX_LENGTH = 1024
HIDDEN_SIZE = 448
INTERMEDIATE_SIZE = 896
NUM_ATTENTION_HEADS = 8
NUM_HIDDEN_LAYERS = 8

PREDICTOR_HIDDEN_LAYERS = 2
PREDICTOR_PREDICTION_COUNT = 24
PRED_HORIZON = 5

DECODER_HIDDEN_LAYERS = 2
DECODER_SLIDING_WINDOW = 96
VISREG_NUM_SLICES = 256
# Push the predictor candidates apart and make the score head discriminative.
# These are applied on resume too, so old checkpoints pick up the new weights.
VISREG_DIVERSITY_W = 0.5
VISREG_SCORE_W = 1.0

# Relative weights of the two objectives in the backward pass.
JEPA_LOSS_WEIGHT = 2.0
DECODER_LOSS_WEIGHT = 1.0

SEED = 42

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jepa train", description=__doc__
    )
    parser.add_argument(
        "--fineweb-dir",
        type=Path,
        default=FINEWEB_DIR,
        help="Prepared token chunks to train on (the default). Produced by "
        "`python -m jepa prepare-data`.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Continue from the checkpoint in {OUTPUT_DIR}.",
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
        "Must match the model when --resume is used.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args(argv)


def build_model(tokenizer) -> JepaDecodeModel:
    """Build a fresh JEPA + decoder model from the component configs."""
    max_length = min(tokenizer.model_max_length or MODEL_MAX_LENGTH, MODEL_MAX_LENGTH)
    tokenizer.model_max_length = max_length

    encoder_config = EncoderConfig(
        vocab_size=len(tokenizer),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        rope_theta=500_000.0,
        hidden_act="silu",
        max_position_embeddings=max_length,
        rms_norm_eps=1e-6,
        pad_token_id=tokenizer.pad_token_id,
        is_causal=True,
    )

    predictor_config = PredictorConfig.from_encoder(
        encoder_config,
        num_hidden_layers=PREDICTOR_HIDDEN_LAYERS,
        prediction_count=PREDICTOR_PREDICTION_COUNT,
    )
    decoder_config = DecoderConfig.from_encoder(
        encoder_config,
        num_hidden_layers=DECODER_HIDDEN_LAYERS,
        sliding_window=DECODER_SLIDING_WINDOW,
    )
    visreg_config = VISRegConfig(
        num_slices=VISREG_NUM_SLICES,
        diversity_w=VISREG_DIVERSITY_W,
        score_w=VISREG_SCORE_W,
    )

    config = JepaConfig.from_encoder(
        encoder_config,
        predictor=predictor_config,
        decoder=decoder_config,
        visreg=visreg_config,
        pred_horizon=PRED_HORIZON,
    )
    return JepaDecodeModel(config)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.tokenizer)
    tokenizer.model_max_length = min(
        tokenizer.model_max_length or MODEL_MAX_LENGTH, MODEL_MAX_LENGTH
    )
    tokenizer("test")  # warm any encoding cache

    if not args.fineweb_dir.is_dir():
        raise FileNotFoundError(
            f"{args.fineweb_dir} not found. "
            "Run `python -m jepa prepare-data` first."
        )
    dataset = load_from_disk(str(args.fineweb_dir))
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
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
        # Apply the current objective weights even when the checkpoint was
        # trained with older values.
        model.criterion.diversity_w = VISREG_DIVERSITY_W
        model.criterion.score_w = VISREG_SCORE_W
        model.config.visreg.diversity_w = VISREG_DIVERSITY_W
        model.config.visreg.score_w = VISREG_SCORE_W
        history_path = OUTPUT_DIR / "history.json"
        history = None
        if history_path.exists():
            with open(history_path) as handle:
                history = json.load(handle)
    else:
        model = build_model(tokenizer)
        history = None

    print("Model loaded")
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
    print("  encoder:  ", f"{sum(p.numel() for p in model.encoder.parameters()):,}")
    print("  predictor:", f"{sum(p.numel() for p in model.predictor.parameters()):,}")
    print("  decoder:  ", f"{sum(p.numel() for p in model.decoder.parameters()):,}")

    # FineWeb chunks are long, so keep an eye on memory. For batch 20, raise
    # the LR to 6e-4-8e-4.
    train_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        optim="adamw_torch",
        eval_strategy="epoch",
        # 3e-4 at batch 3-6; 6e-4-8e-4 at batch 20 (sqrt-to-linear scaling).
        learning_rate=1e-3,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to=[],
        seed=SEED,
    )

    N_TRAIN = 4
    N_VAL = 2
    train_subset = train_dataset.select(range(min(N_TRAIN, len(train_dataset))))
    val_subset = val_dataset.select(range(min(N_VAL, len(val_dataset))))

    trainer = JepaTrainer(
        model=model,
        args=train_args,
        train_dataset=train_subset if args.subset else train_dataset,
        eval_dataset=val_subset if args.subset else val_dataset,
        processing_class=tokenizer,
        jepa_loss_weight=JEPA_LOSS_WEIGHT,
        decoder_loss_weight=DECODER_LOSS_WEIGHT,
        max_grad_norm=1.0,
        save_components=True,
        history=history,
        use_amp=USE_AMP,
        amp_dtype=AMP_DTYPE,
        compile_model=COMPILE_MODEL,
        compile_mode=COMPILE_MODE,
        decoder_loss_chunk_size=DECODER_LOSS_CHUNK_SIZE,
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
        log_every=LOG_EVERY,
        right_padded=RIGHT_PADDED,
    )

    trainer.train()


if __name__ == "__main__":
    main()
