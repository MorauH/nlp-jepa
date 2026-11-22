"""Device helpers shared by the CLI commands."""

from __future__ import annotations

import torch


def select_device(name: str | None = None) -> torch.device:
    """Return an explicit device, or the best available one."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
