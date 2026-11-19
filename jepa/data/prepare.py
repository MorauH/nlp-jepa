"""Prepare a ~500M-token FineWeb-Edu subset for JEPA training.

Streams FineWeb-Edu, tokenizes with the 32k tokenizer from
:mod:`jepa.data.train_tokenizer`, splits long documents into non-overlapping
chunks, and writes a tokenized DatasetDict with train and validation splits.
No BOS/EOS tokens are added.

Usage:
    python -m jepa prepare-data
    python -m jepa prepare-data --target-tokens 500_000_000 --output data/fineweb_edu_500m_32k
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow.compute as pc
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_dataset,
)

from ..paths import FINEWEB_DIR, TOKENIZER_DIR
from .tokenizer import load_tokenizer


def _chunks(ids, max_length, min_length):
    """Yield non-overlapping chunks of at most ``max_length`` tokens."""
    for start in range(0, len(ids), max_length):
        chunk = ids[start : start + max_length]
        if len(chunk) >= min_length:
            yield chunk


def _read_documents(path):
    """Yield one-document-per-line examples from a plain text file."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield {"text": line}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m jepa prepare-data", description=__doc__
    )
    parser.add_argument("--output", type=Path, default=FINEWEB_DIR)
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=500_000_000,
        help="Stop after approximately this many tokens (default: 500M).",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--min-chunk-tokens", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    parser.add_argument(
        "--text-file",
        type=Path,
        default=None,
        help="Read documents from a local one-document-per-line file instead "
        "of streaming FineWeb-Edu, e.g. the --save-text output of "
        "`python -m jepa train-tokenizer`. The file must hold enough text for "
        "--target-tokens (~2.3 GB for 500M tokens).",
    )
    args = parser.parse_args(argv)

    if args.target_tokens < 1:
        parser.error("--target-tokens must be positive")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    if not 0 < args.min_chunk_tokens <= args.max_length:
        parser.error("--min-chunk-tokens must be in (0, max-length]")

    tokenizer = load_tokenizer(args.tokenizer)
    stream = None
    if args.text_file is not None:
        if not args.text_file.is_file():
            parser.error(f"--text-file {args.text_file} does not exist")
        print(f"Reading documents from {args.text_file}")
    else:
        print(f"Streaming HuggingFaceFW/fineweb-edu [{args.config}]")
        stream = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=args.config,
            split="train",
            streaming=True,
        ).shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    stats = {"docs": 0, "chunks": 0, "tokens": 0}
    # Hold out whole documents, not individual chunks, so validation does not
    # contain continuations of training chunks.
    val_every = max(2, round(1 / args.val_fraction))
    started = time.time()

    def generate():
        if args.text_file is not None:
            examples = _read_documents(args.text_file)
        else:
            examples = stream
        for example in examples:
            stats["docs"] += 1
            is_val = int(stats["docs"] % val_every == 0)
            ids = tokenizer._encode_text(example["text"])
            for chunk in _chunks(ids, args.max_length, args.min_chunk_tokens):
                yield {"input_ids": chunk, "split": is_val}
                stats["chunks"] += 1
                stats["tokens"] += len(chunk)
                if stats["tokens"] >= args.target_tokens:
                    print(
                        f"reached target: docs={stats['docs']:,} "
                        f"chunks={stats['chunks']:,} "
                        f"tokens={stats['tokens']:,} "
                        f"elapsed={time.time() - started:.0f}s",
                        flush=True,
                    )
                    return
            if stats["docs"] % 2000 == 0:
                print(
                    f"docs={stats['docs']:,} "
                    f"chunks={stats['chunks']:,} "
                    f"tokens={stats['tokens']:,} "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
        print(
            f"stream exhausted: docs={stats['docs']:,} "
            f"chunks={stats['chunks']:,} tokens={stats['tokens']:,}",
            flush=True,
        )

    print("Tokenizing and writing chunks...")
    dataset = Dataset.from_generator(
        generate,
        features=Features(
            {
                "input_ids": Sequence(Value("int32")),
                "split": Value("int8"),
            }
        ),
        writer_batch_size=1000,
    )
    token_count = pc.sum(
        pc.list_value_length(dataset.data.column("input_ids"))
    ).as_py()
    print(
        f"wrote {len(dataset):,} chunks / {token_count:,} tokens; "
        f"holding out every {val_every}th document for validation"
    )

    train = dataset.filter(lambda example: example["split"] == 0)
    validation = dataset.filter(lambda example: example["split"] == 1)
    train = train.remove_columns("split")
    validation = validation.remove_columns("split")
    output = DatasetDict({"train": train, "validation": validation})
    args.output.mkdir(parents=True, exist_ok=True)
    output.save_to_disk(str(args.output))

    metadata = {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": args.config,
        "tokenizer": str(args.tokenizer),
        "text_file": str(args.text_file) if args.text_file else None,
        "max_length": args.max_length,
        "min_chunk_tokens": args.min_chunk_tokens,
        "target_tokens": args.target_tokens,
        "tokens": token_count,
        "chunks": len(dataset),
        "train_chunks": len(output["train"]),
        "validation_chunks": len(output["validation"]),
        "val_fraction": args.val_fraction,
        "val_every_documents": val_every,
        "shuffle_buffer": args.shuffle_buffer,
        "seed": args.seed,
    }
    with open(args.output / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    print(json.dumps(metadata, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
