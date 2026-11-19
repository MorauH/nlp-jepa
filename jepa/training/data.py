"""Dataset collation and length-bucketed batching."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

# Number of batches' worth of examples to sort together when bucketing by
# length. Larger windows pad less but correlate batch content more.
LENGTH_BUCKET_WINDOW = 100


def example_lengths(dataset) -> list[int]:
    """Per-example lengths for length bucketing.

    Prepared FineWeb chunks carry ``input_ids``; raw text corpora only have
    ``text``, where the character count is a good enough proxy for the token
    length.
    """
    column = "input_ids" if "input_ids" in dataset.column_names else "text"
    try:
        import pyarrow.compute as pc

        values = dataset.data.column(column)
        # ``select``/``filter`` keep the selection in ``_indices``; ``data`` is
        # the unfiltered table, so apply it before measuring.
        indices = getattr(dataset, "_indices", None)
        if indices is not None:
            values = values.take(indices.column(0))
        if column == "input_ids":
            return pc.list_value_length(values).to_pylist()
        return pc.utf8_length(values).to_pylist()
    except Exception:
        return [len(dataset[i][column]) for i in range(len(dataset))]


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Yield batches of similarly long examples to avoid padding.

    FineWeb chunks vary from 64 to 1024 tokens: a document's last chunk is
    usually short, and :func:`tokenize_batch` pads every batch to its longest
    sample. Sorting within a window of ``window_batches`` batches keeps the
    data order shuffled overall while making each batch nearly
    length-homogeneous, which cuts the padded positions substantially.

    Call :meth:`set_epoch` before each epoch to reshuffle.
    """

    def __init__(
        self,
        lengths,
        batch_size: int,
        *,
        drop_last: bool = False,
        seed: int = 0,
        window_batches: int = LENGTH_BUCKET_WINDOW,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.window = max(batch_size, batch_size * window_batches)
        self.epoch = 0
        self._batches: list[list[int]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._batches = None

    def _build(self) -> None:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        lengths = self.lengths
        batches: list[list[int]] = []
        for start in range(0, len(order), self.window):
            window = order[start : start + self.window]
            # Sort by length; equal lengths keep the shuffled order (stable).
            window.sort(key=lengths.__getitem__)
            for begin in range(0, len(window), self.batch_size):
                batch = window[begin : begin + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        # Materialize in a shuffled batch order so consecutive steps do not
        # see monotonically increasing lengths.
        permutation = torch.randperm(len(batches), generator=generator).tolist()
        self._batches = [batches[i] for i in permutation]

    def __iter__(self):
        if self._batches is None:
            self._build()
        return iter(self._batches)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)


def tokenize_batch(batch, tokenizer, pad_to_multiple_of: int = 1):
    """Collate examples into ``(labels, attention_mask)`` tensors.

    Accepts either raw ``text`` examples or pre-tokenized ``input_ids``
    examples (used by the FineWeb-Edu preparation script).  Padding is
    right-aligned and ignored by the loss.
    """
    if "input_ids" in batch[0]:
        sequences = [list(example["input_ids"]) for example in batch]
        max_len = max(len(ids) for ids in sequences)
        if pad_to_multiple_of > 1:
            max_len = (
                (max_len + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of

        pad_id = tokenizer.pad_token_id
        input_ids = [
            ids + [pad_id] * (max_len - len(ids)) for ids in sequences
        ]
        attention_mask = [
            [1] * len(ids) + [0] * (max_len - len(ids)) for ids in sequences
        ]
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    encoded = tokenizer(
        [example["text"] for example in batch],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    if pad_to_multiple_of > 1:
        extra = (-input_ids.shape[1]) % pad_to_multiple_of
        if extra:
            input_ids = F.pad(
                input_ids, (0, extra), value=tokenizer.pad_token_id
            )
            attention_mask = F.pad(attention_mask, (0, extra), value=0)

    return input_ids, attention_mask
