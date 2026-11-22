"""Device and runtime helpers shared by the CLI commands."""

from __future__ import annotations

import os

import torch


def available_cpu_count() -> int:
    """Return the number of CPUs this process is actually allowed to use."""
    if hasattr(os, "process_cpu_count"):  # Python >= 3.13
        count = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    return max(1, count or 1)


def select_device(name: str | None = None) -> torch.device:
    """Return an explicit device, or the best available one."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_amp_dtype(
    device: torch.device, requested: str | None = "auto"
) -> torch.dtype | None:
    """Pick the autocast dtype, or ``None`` when AMP is unavailable."""
    if device.type != "cuda":
        return None

    requested = (requested or "auto").lower()
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested in ("bf16", "bfloat16"):
        return torch.bfloat16
    if requested in ("fp16", "float16", "half"):
        return torch.float16
    if requested in ("none", "off", "false"):
        return None
    raise ValueError(
        "amp_dtype must be 'auto', 'bf16', 'fp16' or 'none'; "
        f"got {requested!r}"
    )
