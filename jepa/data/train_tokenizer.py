"""Train the 32k byte-level BPE tokenizer used by the JEPA stack.

Streams English FineWeb-Edu text and trains a GPT-2-style byte-level BPE with
the HuggingFace ``tokenizers`` (Rust) implementation.  The result is saved as a
``PreTrainedTokenizerFast`` directory, so it loads with
``AutoTokenizer.from_pretrained``.

The tokenizer reserves ``<pad>`` (id 0), ``<bos>`` and ``<eos>``.  ``<pad>``
must stay distinct from ``<eos>``: :mod:`jepa.decoder` ignores
``pad_token_id`` in the loss.

Usage:
    python -m jepa train-tokenizer
    python -m jepa train-tokenizer --megabytes 500 --vocab-size 32000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from ..paths import TOKENIZER_DIR

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jepa train-tokenizer", description=__doc__
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TOKENIZER_DIR,
        help="Directory for tokenizer.json and tokenizer_config.json.",
    )
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument(
        "--megabytes",
        type=int,
        default=500,
        help="Millions of characters of FineWeb-Edu text to stream. 500 MB is "
        "enough to train the 32k vocabulary; pass ~2300 to also save enough "
        "text for a 500M-token dataset.",
    )
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--model-max-length", type=int, default=1024)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only stream and save the corpus (requires --save-text); skip "
        "tokenizer training. Use to extend a sample without retraining.",
    )
    parser.add_argument("--heldout-docs", type=int, default=100)
    parser.add_argument(
        "--save-text",
        type=Path,
        default=None,
        help="Also write the training documents to this file (one per line) "
        "so `python -m jepa prepare-data --text-file ...` can reuse them "
        "without streaming FineWeb-Edu again.",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.vocab_size < 512:
        raise SystemExit("--vocab-size must be at least 512")

    if args.text_only and args.save_text is None:
        raise SystemExit("--text-only requires --save-text")
    if args.megabytes < 1:
        raise SystemExit("--megabytes must be positive")

    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.config,
        split="train",
        streaming=True,
    ).shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    documents = iter(stream)

    # Hold out whole documents so the reported compression is on unseen text.
    # Not needed when only saving the corpus.
    heldout = []
    if not args.text_only:
        heldout = [next(documents)["text"] for _ in range(args.heldout_docs)]

    target_chars = args.megabytes * 1_000_000
    stats = {"docs": 0, "chars": 0}

    saved_text = None
    if args.save_text is not None:
        args.save_text.parent.mkdir(parents=True, exist_ok=True)
        saved_text = open(args.save_text, "w", encoding="utf-8")

    def train_texts():
        started = time.time()
        for example in documents:
            text = example["text"]
            stats["docs"] += 1
            stats["chars"] += len(text)
            if saved_text is not None:
                # One document per line for `jepa prepare-data --text-file`.
                saved_text.write(text.replace("\n", " ").replace("\r", " ") + "\n")
            yield text
            if stats["docs"] % 2_000 == 0:
                print(
                    f"  streamed {stats['chars'] / 1e6:7.1f} / "
                    f"{args.megabytes} MB "
                    f"({stats['docs']:,} docs, {time.time() - started:.0f}s)",
                    flush=True,
                )
            if stats["chars"] >= target_chars:
                return

    if args.text_only:
        print(
            f"Saving {args.megabytes} MB of FineWeb-Edu [{args.config}] "
            f"to {args.save_text}"
        )
        started = time.time()
        try:
            for _ in train_texts():
                pass
        finally:
            if saved_text is not None:
                saved_text.close()
        print(
            f"wrote {stats['chars'] / 1e6:.0f} MB / {stats['docs']:,} docs "
            f"in {time.time() - started:.0f}s"
        )
        return

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    print(f"Training {args.vocab_size:,}-token BPE on FineWeb-Edu [{args.config}]")
    started = time.time()
    try:
        tokenizer.train_from_iterator(train_texts(), trainer=trainer)
    finally:
        if saved_text is not None:
            saved_text.close()
    train_seconds = time.time() - started

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        model_max_length=args.model_max_length,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(args.output)

    # Sanity check: byte-level BPE round-trips every string, including the
    # characters FineWeb-Edu contains but the corpus may only rarely show.
    probe = "Hello world — naïve café 42\n\t<>&"
    assert fast.decode(fast.encode(probe, add_special_tokens=False)) == probe

    heldout_chars = sum(len(text) for text in heldout)
    heldout_tokens = sum(
        len(tokenizer.encode(text, add_special_tokens=False).ids)
        for text in heldout
    )
    metadata = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": args.config,
        "vocab_size": len(fast),
        "pad_token_id": fast.pad_token_id,
        "bos_token_id": fast.bos_token_id,
        "eos_token_id": fast.eos_token_id,
        "min_frequency": args.min_frequency,
        "train_documents": stats["docs"],
        "train_chars": stats["chars"],
        "train_seconds": round(train_seconds, 1),
        "text_file": str(args.save_text) if args.save_text else None,
        "heldout_documents": len(heldout),
        "heldout_chars_per_token": round(heldout_chars / heldout_tokens, 3),
        "shuffle_buffer": args.shuffle_buffer,
        "seed": args.seed,
    }
    with open(args.output / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
