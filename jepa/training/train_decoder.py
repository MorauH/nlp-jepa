"""Train the autoregressive decoder against a frozen JEPA encoder.

This is the decoder-only counterpart of ``jepa.training.train``.  It loads the encoder
and predictor from an existing JEPA checkpoint, freezes them, encodes each
batch once, and trains only the decoder on next-token prediction.  The decoder
is conditioned on the real encoder latent at the block start and the frozen
predictor's selected waypoint at the block end.  This matches inference and
avoids leaking the final target token.

The saved checkpoint is a complete :class:`JepaDecodeModel`, so it can be used
with ``python -m jepa generate --model <output-dir>``.

Examples::

    python -m jepa train-decoder --model-dir runs/jepa --output-dir runs/jepa-decoder
    python -m jepa train-decoder --batch-size 8 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from functools import partial
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import set_seed

from ..config import DecoderConfig, JepaConfig
from ..data.tokenizer import load_tokenizer
from ..models.encoder import EncodeTransformer
from ..models.predictor import PredictorModel
from ..models.stack import JepaDecodeModel, build_block_hints, select_waypoints
from ..paths import (
    DEFAULT_DECODER_RUN_DIR,
    DEFAULT_RUN_DIR,
    FINEWEB_DIR,
    TOKENIZER_DIR,
)
from ..runtime import available_cpu_count, resolve_amp_dtype, select_device
from .data import LengthBucketBatchSampler, example_lengths, tokenize_batch
from .optim import parameter_groups

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = DEFAULT_RUN_DIR
DEFAULT_OUTPUT_DIR = DEFAULT_DECODER_RUN_DIR

# ---------------------------------------------------------------------------
# Decoder / training shape
# ---------------------------------------------------------------------------
MODEL_MAX_LENGTH = 1024
DECODER_HIDDEN_LAYERS = 2
DECODER_SLIDING_WINDOW = 96
DECODER_LOSS_CHUNK_SIZE = 256
PAD_TO_MULTIPLE_OF = 64

USE_AMP = True
AMP_DTYPE = "auto"
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 42
LOG_EVERY = 10
RIGHT_PADDED = True

N_TRAIN = 4
N_VAL = 2


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, ...")
    return parser.parse_args(argv)


def load_data(args: argparse.Namespace, tokenizer):
    if not args.fineweb_dir.is_dir():
        raise FileNotFoundError(
            f"{args.fineweb_dir} not found. "
            "Run `python -m jepa prepare-data` first."
        )
    dataset = load_from_disk(str(args.fineweb_dir))
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    if args.subset:
        train_dataset = train_dataset.select(
            range(min(N_TRAIN, len(train_dataset)))
        )
        val_dataset = val_dataset.select(range(min(N_VAL, len(val_dataset))))

    return train_dataset, val_dataset


def build_decoder_model(args: argparse.Namespace) -> JepaDecodeModel:
    """Load the existing encoder/predictor and attach a fresh decoder."""
    encoder = EncodeTransformer.from_pretrained(str(args.model_dir / "encoder"))
    predictor = PredictorModel.from_pretrained(str(args.model_dir / "predictor"))
    base_config = JepaConfig.from_json_file(str(args.model_dir / "config.json"))

    decoder_config = DecoderConfig.from_encoder(
        encoder.config,
        num_hidden_layers=DECODER_HIDDEN_LAYERS,
        sliding_window=DECODER_SLIDING_WINDOW,
    )
    config = JepaConfig.from_encoder(
        encoder.config,
        predictor=predictor.config,
        decoder=decoder_config,
        visreg=base_config.visreg,
        pred_horizon=base_config.pred_horizon,
    )
    model = JepaDecodeModel(config, encoder=encoder, predictor=predictor)

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.predictor.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    model.predictor.eval()
    return model


def encode_latents(model: JepaDecodeModel, labels, attention_mask):
    """Run the frozen encoder once and return detached block hints."""
    enc_mask = None if RIGHT_PADDED and model.encoder.config.is_causal else attention_mask
    with torch.no_grad():
        latents = model.encoder(labels, attention_mask=enc_mask)
    return latents.detach()


@torch.no_grad()
def predict_waypoints(model: JepaDecodeModel, latents, horizon: int):
    """Frozen predictor waypoint at each base position."""
    candidates, scores = model.predictor(
        latents[:, :-horizon, :],
        return_all_timesteps=True,
    )
    return select_waypoints(candidates, scores)


def decoder_loss(model: JepaDecodeModel, labels, attention_mask, latents):
    predicted_future = predict_waypoints(model, latents, model.pred_horizon)
    z_t, z_future = build_block_hints(
        latents,
        model.pred_horizon,
        predicted_future=predicted_future,
    )
    return model.decoder(
        z_t=z_t,
        z_future=z_future,
        labels=labels,
        training=True,
        loss_chunk_size=DECODER_LOSS_CHUNK_SIZE,
        attention_mask=attention_mask,
    )["loss"]


def make_loader(dataset, batch_size: int, shuffle: bool, device, tokenizer):
    num_workers = min(8, available_cpu_count())
    collate = partial(
        tokenize_batch,
        tokenizer=tokenizer,
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
    )
    kwargs = dict(
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    if shuffle:
        # Bucket by length to avoid padding every batch to 1024.
        return DataLoader(
            dataset,
            batch_sampler=LengthBucketBatchSampler(
                example_lengths(dataset), batch_size, seed=SEED
            ),
            **kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **kwargs,
    )


@torch.no_grad()
def evaluate(model: JepaDecodeModel, loader, device, amp_dtype) -> float:
    model.decoder.eval()
    total = torch.zeros((), device=device)
    n_batches = 0
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else torch.autocast(device_type=device.type, enabled=False)
    )

    for labels, attention_mask in loader:
        labels = labels.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        if labels.shape[1] <= model.pred_horizon:
            continue
        with autocast:
            latents = encode_latents(model, labels, attention_mask)
            loss = decoder_loss(model, labels, attention_mask, latents)
        total += loss.detach()
        n_batches += 1

    model.decoder.train()
    return float(total / max(1, n_batches))


def save_checkpoint(output_dir: Path, model: JepaDecodeModel, history: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    model.encoder.save_pretrained(output_dir / "encoder")
    model.predictor.save_pretrained(output_dir / "predictor")
    model.decoder.save_pretrained(output_dir / "decoder")
    with open(output_dir / "history.json", "w") as handle:
        json.dump(history, handle, indent=4)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(SEED)
    device = select_device(args.device)

    tokenizer = load_tokenizer(args.tokenizer)
    tokenizer.model_max_length = min(
        tokenizer.model_max_length or MODEL_MAX_LENGTH, MODEL_MAX_LENGTH
    )
    tokenizer("test")  # warm any encoding cache

    train_dataset, val_dataset = load_data(args, tokenizer)
    print(
        f"train examples: {len(train_dataset):,} | "
        f"val examples: {len(val_dataset):,}"
    )

    model = build_decoder_model(args).to(device)
    if model.config.encoder.vocab_size != len(tokenizer):
        raise ValueError(
            f"{args.model_dir} was trained with vocab_size="
            f"{model.config.encoder.vocab_size}, but {args.tokenizer} has "
            f"{len(tokenizer)} tokens. Pass the matching --tokenizer."
        )
    trainable = sum(p.numel() for p in model.decoder.parameters())
    print(f"Decoder parameters: {trainable:,}")
    print("Encoder/predictor: frozen")

    optimizer_kwargs = {}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(
            parameter_groups(model.decoder, WEIGHT_DECAY),
            lr=args.lr,
            **optimizer_kwargs,
        )
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(
            parameter_groups(model.decoder, WEIGHT_DECAY),
            lr=args.lr,
            **optimizer_kwargs,
        )

    amp_dtype = resolve_amp_dtype(device, AMP_DTYPE) if USE_AMP else None
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(amp_dtype == torch.float16)
    )
    print(
        f"AMP: {str(amp_dtype).replace('torch.', '') if amp_dtype else 'off'} | "
        f"batch: {args.batch_size} | device: {device}"
    )

    train_loader = make_loader(
        train_dataset, args.batch_size, True, device, tokenizer
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, device, tokenizer
    )

    total_steps = max(1, len(train_loader) * max(1, args.epochs))
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
        warnings.filterwarnings(
            "ignore", message="Detected call of `lr_scheduler.step"
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_loss": [], "val_loss": []}
    autocast = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None
        else torch.autocast(device_type=device.type, enabled=False)
    )

    for epoch in range(args.epochs):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if isinstance(batch_sampler, LengthBucketBatchSampler):
            batch_sampler.set_epoch(epoch)
        model.decoder.train()
        running = torch.zeros((), device=device)
        n_batches = 0
        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{args.epochs}",
        )

        for labels, attention_mask in pbar:
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            if labels.shape[1] <= model.pred_horizon:
                continue

            with autocast:
                latents = encode_latents(model, labels, attention_mask)
                loss = decoder_loss(model, labels, attention_mask, latents)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if MAX_GRAD_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.decoder.parameters(), MAX_GRAD_NORM
                )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.detach()
            n_batches += 1

            if n_batches % LOG_EVERY == 0:
                pbar.set_postfix(
                    loss=f"{float(running / n_batches):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

        train_loss = float(running / max(1, n_batches))
        val_loss = evaluate(model, val_loader, device, amp_dtype)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )
        save_checkpoint(args.output_dir, model, history)

    print(f"Saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
