"""Train the autoregressive decoder against a frozen JEPA encoder/predictor.

Loads a JEPA checkpoint, freezes its encoder and predictor, attaches a fresh
decoder, and runs the same :class:`jepa.JepaTrainer` as ``python -m jepa train``
in frozen mode: only the decoder loss is computed and only the decoder is
optimized.  The decoder is conditioned on the real encoder latent at the block
start and a frozen predictor candidate at the block end (the hindsight-best one
during training, the score-selected one in validation; see
:class:`jepa.models.stack.JepaDecodeModel`).

The saved checkpoint is a complete :class:`JepaDecodeModel`, so it can be used
with ``python -m jepa generate --model <output-dir>``.

By default it reads and writes ``runs/jepa``: the checkpoint is rewritten with
the same (frozen, unchanged) encoder/predictor and the new decoder, replacing
the jointly trained one.  The loss history goes to ``history_decoder.json`` so
the joint run's ``history.json`` (read by ``train --resume``) is kept.

Examples::

    python -m jepa train-decoder
    python -m jepa train-decoder --resume --epochs 2   # keep training that decoder
    python -m jepa train-decoder --model-dir runs/jepa --output-dir /tmp/jepa-dec
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import set_seed

from ..config import DecoderConfig, JepaConfig
from ..models.encoder import EncodeTransformer
from ..models.predictor import PredictorModel
from ..models.stack import JepaDecodeModel
from ..paths import DEFAULT_RUN_DIR
from .train import (
    AMP_DTYPE,
    COMPILE_MODE,
    COMPILE_MODEL,
    DECODER_LOSS_CHUNK_SIZE,
    DECODER_LOSS_WEIGHT,
    LOG_EVERY,
    MAX_GRAD_NORM,
    PAD_TO_MULTIPLE_OF,
    RIGHT_PADDED,
    SAVE_EVERY_STEPS,
    SEED,
    USE_AMP,
    WEIGHT_DECAY,
    add_dataset_args,
    load_datasets,
    load_training_tokenizer,
)
from .trainer import JepaTrainer, TrainConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Same directory as the joint run (see the module docstring).
DEFAULT_MODEL_DIR = DEFAULT_RUN_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR
HISTORY_FILE = "history_decoder.json"

# Decoder-only defaults.  The frozen backbone removes most of the memory cost,
# so the joint run's batch size is safe; the LR is the batch-24 value from
# train.py's LEARNING_RATE comment, and one epoch over the prepared data is
# usually enough to adapt a fresh decoder.
DECODER_BATCH_SIZE = 24
DECODER_LEARNING_RATE = 6e-4
DECODER_EPOCHS = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jepa train-decoder", description=__doc__
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Existing JEPA checkpoint containing encoder/, predictor/ and config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write the complete checkpoint with the new decoder.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep training the decoder already in --model-dir (instead of "
        "attaching a fresh one), continuing its history_decoder.json.  Like "
        "`train --resume`, the LR schedule restarts.",
    )
    add_dataset_args(parser, batch_size=DECODER_BATCH_SIZE, epochs=DECODER_EPOCHS)
    parser.add_argument("--lr", type=float, default=DECODER_LEARNING_RATE)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, ...")
    return parser.parse_args(argv)


def load_model_for_resume(model_dir: Path) -> JepaDecodeModel:
    """Load the full checkpoint, keeping its decoder, with the JEPA part frozen.

    Unlike :func:`build_model` the decoder is not replaced, and its config
    (layers, window, ...) comes from the checkpoint.
    """
    model = JepaDecodeModel.from_pretrained(str(model_dir))
    model.freeze_jepa()
    return model


def build_model(model_dir: Path) -> JepaDecodeModel:
    """Load the checkpoint's encoder/predictor and attach a fresh decoder."""
    encoder = EncodeTransformer.from_pretrained(str(model_dir / "encoder"))
    predictor = PredictorModel.from_pretrained(str(model_dir / "predictor"))
    base_config = JepaConfig.from_json_file(str(model_dir / "config.json"))

    decoder_config = DecoderConfig.from_encoder(encoder.config)
    config = JepaConfig.from_encoder(
        encoder.config,
        predictor=predictor.config,
        decoder=decoder_config,
        visreg=base_config.visreg,
        pred_horizon=base_config.pred_horizon,
    )
    model = JepaDecodeModel(config, encoder=encoder, predictor=predictor)
    model.freeze_jepa()
    return model


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(SEED)

    tokenizer = load_training_tokenizer(args.tokenizer)
    train_dataset, val_dataset = load_datasets(args.fineweb_dir, args.subset)
    print(
        f"train examples: {len(train_dataset):,} | "
        f"val examples: {len(val_dataset):,}"
    )

    history = None
    if args.resume:
        model = load_model_for_resume(args.model_dir)
        history_path = args.model_dir / HISTORY_FILE
        if history_path.exists():
            history = json.loads(history_path.read_text())
    else:
        model = build_model(args.model_dir)
    if model.config.encoder.vocab_size != len(tokenizer):
        raise ValueError(
            f"{args.model_dir} was trained with vocab_size="
            f"{model.config.encoder.vocab_size}, but {args.tokenizer} has "
            f"{len(tokenizer)} tokens. Pass the matching --tokenizer."
        )
    print(
        "Decoder parameters: "
        f"{sum(p.numel() for p in model.decoder.parameters()):,}"
    )
    print(
        "Encoder/predictor: frozen | decoder: "
        + ("resumed from checkpoint" if args.resume else "fresh")
    )

    trainer = JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(args.output_dir),
            learning_rate=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            weight_decay=WEIGHT_DECAY,
            seed=SEED,
            device=args.device,
            history_file=HISTORY_FILE,
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        history=history,
        decoder_loss_weight=DECODER_LOSS_WEIGHT,
        max_grad_norm=MAX_GRAD_NORM,
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
    print(f"Saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
