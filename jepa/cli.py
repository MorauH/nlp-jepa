"""Unified command-line interface: ``python -m jepa <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from .data.prepare import main as prepare_data
from .data.train_tokenizer import main as train_tokenizer
from .evaluation.verify import main as verify
from .inference.generate import main as generate
from .training.train import main as train
from .training.train_decoder import main as train_decoder

Command = Callable[[list[str] | None], None]

COMMANDS: dict[str, tuple[Command, str]] = {
    "train": (train, "Train the full JEPA encoder + predictor + decoder."),
    "train-decoder": (
        train_decoder,
        "Fine-tune the decoder against a frozen JEPA stack.",
    ),
    "generate": (generate, "Generate text from a trained checkpoint."),
    "prepare-data": (prepare_data, "Tokenize FineWeb-Edu into training chunks."),
    "train-tokenizer": (train_tokenizer, "Train the 32k byte-level BPE tokenizer."),
    "verify": (verify, "Run collapse and latent-usage checks."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jepa",
        description="JEPA language model command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_, help_text) in COMMANDS.items():
        # The command module owns its own --help; this parser only lists it.
        subparsers.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0

    command, *rest = argv
    if command not in COMMANDS:
        print(f"unknown command: {command!r}", file=sys.stderr)
        build_parser().print_help(sys.stderr)
        return 2

    COMMANDS[command][0](rest)
    return 0
