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
from .trainer import JepaTrainer, TrainConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = DEFAULT_RUN_DIR

# --subset trains on this many examples (smoke test).
SUBSET_TRAIN = 4
SUBSET_VAL = 2

# ---------------------------------------------------------------------------
# Model shape
# ---------------------------------------------------------------------------
MODEL_MAX_LENGTH = 1024
HIDDEN_SIZE = 448
INTERMEDIATE_SIZE = 896
NUM_ATTENTION_HEADS = 8
NUM_HIDDEN_LAYERS = 8

# Shared causal trunk over the latent sequence, run before the per-position
# candidate head.  Six layers let every position read the whole prefix through
# attention, so the predictor is no longer limited to z_t (or a fixed window).
PREDICTOR_TRUNK_LAYERS = 6
# Layers of the candidate head itself (expansion slots only, not time steps).
PREDICTOR_HIDDEN_LAYERS = 2
# Candidate futures per position.  Analysis of the old 24-candidate head found
# only ~5 distinct candidate groups, so 8 keeps the useful diversity at a third
# of the head's slot/score cost.
PREDICTOR_PREDICTION_COUNT = 8
# Past latents z_{t-W+1..t} the predictor slots (and score head) attend to.
# 0 = only z_t.  The trunk already carries the temporal context, so a separate
# window only adds tokens and activations.
PREDICTOR_CONTEXT_WINDOW = 0
# Tokens between z_t and the predicted latent, and the decoder's block length
# (one waypoint per block).  At k=5 the latents became too smooth
# (cos(z_t, z_t+5) ~0.90): the future was nearly copyable, only 2 of 8
# candidates were used, and the waypoint added little beyond z_t.  A longer
# horizon makes the future genuinely uncertain, which is what the candidates
# and the waypoint hint are for.  Stored in the checkpoint config, so it only
# applies to fresh models (`--resume` and `train-decoder` keep the saved k).
PRED_HORIZON = 20

# Deeper token decoder.  Its tables are tied (below) and the hint is injected
# at every layer, so the extra depth buys grammar, not capacity in the tables.
DECODER_HIDDEN_LAYERS = 4
# Past tokens visible to each decoder position.  Short on purpose: with 96
# tokens of context the decoder learned to ignore the latent hint (verify:
# predicted ~ shuffled hint), so a narrower window makes it lean on the hint
# for anything beyond local grammar.
DECODER_SLIDING_WINDOW = 32
# Probability that a decoder block is trained on the hindsight-best predictor
# candidate (closest to the real future) instead of the score-selected one.
# The hindsight choice carries real information about the continuation, which
# teaches the decoder to use the hint.  Validation always uses the
# score-selected waypoint (the inference condition), so a train/val decoder
# loss gap shows how much the decoder relies on the hint.  0.0 = old behaviour.
DECODER_HINDSIGHT_PROB = 1.0
# Share the decoder input embedding with the lm_head: one table instead of two
# (the encoder keeps its own, so the decoder loss cannot touch the encoder).
DECODER_TIE_EMBEDDINGS = True
# Re-inject the latent hint before every decoder layer instead of only at the
# input.  The per-layer projections start at zero, so this is a strict
# extension of input-only injection.
DECODER_HINT_EVERY_LAYER = True
VISREG_NUM_SLICES = 256
# Push the predictor candidates apart and make the score head discriminative.
# These are applied on resume too, so old checkpoints pick up the new weights.
VISREG_DIVERSITY_W = 0.5
VISREG_SCORE_W = 1.0

# Candidate-collapse fixes.  In the previous run only 2 of 8 slots were ever
# used: the other 6 drifted to norms ~50x z_t (the loss compares directions,
# so norm was free), where the gradient through the loss's normalization
# vanishes; the diversity penalty let them hide orthogonal to everything; and
# the sharp assignment gave them ~zero weight.
# 1. Fix every candidate to the encoder-latent scale (architecture; saved in
#    the predictor config).
PREDICTOR_NORMALIZE_CANDIDATES = True
# 2. Annealed winner-takes-all: the assignment-temperature multiplier starts
#    at VISREG_MH_TEMPERATURE_START (every slot shares every target, so each
#    finds a region) and decays geometrically to the final value (0.1, the
#    VISRegConfig default) over the first VISREG_MH_ANNEAL_FRACTION of steps.
VISREG_MH_TEMPERATURE_START = 1.0
VISREG_MH_ANNEAL_FRACTION = 0.3
# 3. Only penalize near-duplicate candidates (cosine above the margin), so an
#    unused slot gains nothing by pointing orthogonal to everything.
VISREG_DIVERSITY_MARGIN = 0.5
# Items 2-3 are applied on resume too (like the weights above).

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
# stay at the conservative end (not 8e-4+) because the 6-layer predictor
# trunk is new and untested.  Rescale (~sqrt) if you change BATCH_SIZE.
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
    """Load the tokenizer and cap its sequence length at MODEL_MAX_LENGTH."""
    tokenizer = load_tokenizer(tokenizer_dir)
    tokenizer.model_max_length = min(
        tokenizer.model_max_length or MODEL_MAX_LENGTH, MODEL_MAX_LENGTH
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
        trunk_layers=PREDICTOR_TRUNK_LAYERS,
        prediction_count=PREDICTOR_PREDICTION_COUNT,
        context_window=PREDICTOR_CONTEXT_WINDOW,
        normalize_candidates=PREDICTOR_NORMALIZE_CANDIDATES,
    )
    decoder_config = DecoderConfig.from_encoder(
        encoder_config,
        num_hidden_layers=DECODER_HIDDEN_LAYERS,
        sliding_window=DECODER_SLIDING_WINDOW,
        tie_embeddings=DECODER_TIE_EMBEDDINGS,
        hint_every_layer=DECODER_HINT_EVERY_LAYER,
    )
    visreg_config = VISRegConfig(
        num_slices=VISREG_NUM_SLICES,
        diversity_w=VISREG_DIVERSITY_W,
        score_w=VISREG_SCORE_W,
        diversity_margin=VISREG_DIVERSITY_MARGIN,
        mh_temperature_start=VISREG_MH_TEMPERATURE_START,
        mh_anneal_fraction=VISREG_MH_ANNEAL_FRACTION,
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
        # Apply the current objective weights even when the checkpoint was
        # trained with older values.
        model.criterion.diversity_w = VISREG_DIVERSITY_W
        model.criterion.score_w = VISREG_SCORE_W
        model.config.visreg.diversity_w = VISREG_DIVERSITY_W
        model.config.visreg.score_w = VISREG_SCORE_W
        for name, value in (
            ("diversity_margin", VISREG_DIVERSITY_MARGIN),
            ("mh_temperature_start", VISREG_MH_TEMPERATURE_START),
            ("mh_anneal_fraction", VISREG_MH_ANNEAL_FRACTION),
        ):
            setattr(model.criterion, name, value)
            setattr(model.config.visreg, name, value)
        history_path = OUTPUT_DIR / "history.json"
        history = None
        if history_path.exists():
            with open(history_path) as handle:
                history = json.load(handle)
    else:
        model = build_model(tokenizer)
        history = None

    # Runtime training knob (not saved in the config), so set it for both a
    # fresh model and a resumed one.
    model.decoder_hindsight_prob = DECODER_HINDSIGHT_PROB
    if args.jepa_only:
        model.skip_decoder()

    print("Model loaded")
    if args.jepa_only:
        print("Mode: JEPA only (encoder + predictor); the decoder is not run")
    else:
        print(
            f"Decoder window: {model.decoder.sliding_window} | "
            f"hindsight waypoint prob: {model.decoder_hindsight_prob}"
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
