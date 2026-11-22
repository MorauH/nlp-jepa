"""Tokenizer loading for the JEPA stack.

``jepa`` trains its own 32k byte-level BPE on FineWeb-Edu (see
:mod:`jepa.data.train_tokenizer`).  The tokenizer is saved as a directory by
``PreTrainedTokenizerFast.save_pretrained`` and reloaded with
``AutoTokenizer.from_pretrained``.

:class:`HFTokenizer` exposes the small interface the training code relies on,
so the rest of ``jepa`` does not depend on the HuggingFace wrapper details.
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..paths import TOKENIZER_DIR


class HFTokenizer:
    """Adapter giving a HuggingFace tokenizer the small JEPA training interface.

    The training code only needs ``len``, ``pad_token_id``, ``eos_token_id``,
    ``model_max_length``, ``_encode_text``, ``__call__`` and ``decode``.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        # jepa feeds right-padded batches to a causal encoder.
        tokenizer.padding_side = "right"

    def __len__(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def model_max_length(self) -> int:
        return self.tokenizer.model_max_length

    @model_max_length.setter
    def model_max_length(self, value: int) -> None:
        self.tokenizer.model_max_length = value

    def _encode_text(self, text: str) -> list[int]:
        """Encode one text without BOS/EOS, as ``prepare_data`` expects.

        Uses the Rust backend directly: whole FineWeb documents are usually
        longer than ``model_max_length`` (1024), and the HF wrapper warns about
        that even though nothing is truncated here.
        """
        return self.tokenizer.backend_tokenizer.encode(
            text, add_special_tokens=False
        ).ids

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        return self.tokenizer(
            texts,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
        )

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids)


def load_tokenizer(path: str | Path = TOKENIZER_DIR) -> HFTokenizer:
    """Load a tokenizer directory saved by ``save_pretrained``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m jepa train-tokenizer` to create "
            f"{TOKENIZER_DIR}."
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
    # The causal encoder relies on right-padded batches: no real query may
    # attend a pad, which lets attention drop the mask and use fused SDPA.
    tokenizer.padding_side = "right"
    return HFTokenizer(tokenizer)


def eos_token_id(tokenizer) -> int | None:
    """End-of-text id for the tokenizer."""
    return tokenizer.eos_token_id
