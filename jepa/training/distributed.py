"""Optional ``torch.distributed`` plumbing for multi-GPU (DDP) training.

Distributed mode is detected from the environment variables ``torchrun``
exports (``WORLD_SIZE`` / ``LOCAL_RANK``), never from a CLI flag, so the same
command line works for one process or many::

    python -m jepa train                              # single process
    torchrun --standalone --nproc_per_node=4 -m jepa train

Every helper degrades to a no-op when the job is not distributed, so the
single-process path never initializes a process group and behaves exactly as
before.  A ``torchrun --nproc_per_node=1`` launch also takes that path: with
``WORLD_SIZE=1`` there is nothing to coordinate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on bad values."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_world_size() -> int:
    """Number of processes in the job (1 when the launcher set nothing)."""
    return max(1, _env_int("WORLD_SIZE", 1))


def env_local_rank() -> int:
    """Rank of this process on its node (0 when the launcher set nothing)."""
    return _env_int("LOCAL_RANK", 0)


def env_local_world_size() -> int:
    """Number of processes on this node."""
    return max(1, _env_int("LOCAL_WORLD_SIZE", env_world_size()))


@dataclass(frozen=True)
class DistributedInfo:
    """Where this process sits in the job, and whether there is a job at all."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1
    backend: str | None = None

    @property
    def is_distributed(self) -> bool:
        """True when more than one process trains (DDP is needed)."""
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """True on the process that prints and writes checkpoints."""
        return self.rank == 0


def setup_distributed(use_cpu: bool = False) -> DistributedInfo:
    """Initialize the process group for this job and return its layout.

    Call this *before* building HF ``TrainingArguments``: accelerate's
    ``PartialState`` initializes a group itself when it sees ``LOCAL_RANK``,
    so creating ours first means it finds ``dist.is_initialized()`` true and
    reuses the group (with our backend and our device selection) instead of
    initializing a second one.
    """
    world_size = env_world_size()
    if world_size <= 1:
        return DistributedInfo()

    local_rank = env_local_rank()
    # NCCL moves CUDA tensors between ranks; CPU-only runs (and the CPU test
    # path) use gloo, which needs no CUDA at all.
    backend = "gloo" if use_cpu or not torch.cuda.is_available() else "nccl"
    if backend == "nccl":
        if local_rank < 0:
            # Launchers that do not export LOCAL_RANK (plain `srun`, say) still
            # export the global rank; modulo the visible device count that is
            # the device this process should own.
            local_rank = _env_int("RANK", 0) % max(1, torch.cuda.device_count())
        # Pin this process to its GPU before anything else touches CUDA: later
        # `"cuda"` tensors (model, batches, optimizer state) then land on the
        # right device without threading an index through the whole trainer.
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        # torchrun exports RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT, so
        # init_process_group reads the topology from the environment.
        dist.init_process_group(backend=backend)

    return DistributedInfo(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        local_world_size=env_local_world_size(),
        backend=dist.get_backend(),
    )


def cleanup_distributed() -> None:
    """Tear the process group down (no-op when it was never created)."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.destroy_process_group()
    except Exception:  # pragma: no cover - shutdown races with a dying peer
        # A peer that already exited can make the teardown raise; there is
        # nothing left to clean up at that point, and masking the original
        # training error would be worse than leaking a handle at process exit.
        pass


def barrier() -> None:
    """Wait for every rank (no-op when not distributed)."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


def reduce_totals(
    totals: torch.Tensor, count: int
) -> tuple[torch.Tensor, int]:
    """Sum per-rank loss totals and sample counts in a single collective.

    Used at epoch/log boundaries only, never per step: the trainers accumulate
    losses on the device and call this when they need a global average.
    """
    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return totals, count
    buffer = torch.cat(
        (
            totals.detach().reshape(-1),
            torch.tensor(
                [float(count)], device=totals.device, dtype=totals.dtype
            ),
        )
    )
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[:-1], int(round(buffer[-1].item()))
