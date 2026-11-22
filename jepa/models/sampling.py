"""Sampling helpers shared by the commit policies and the token decoder."""

from __future__ import annotations

from torch import Tensor


def top_k_mask(logits: Tensor, k: int | None) -> Tensor:
    """Keep the ``k`` highest logits, masking the rest with ``-inf``.

    ``None`` or a non-positive ``k`` disables the cutoff and returns ``logits``
    unchanged.  ``k`` larger than the last dimension is clamped to it.
    """
    if k is None or k <= 0:
        return logits
    k = min(int(k), logits.shape[-1])
    kth = logits.topk(k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))
