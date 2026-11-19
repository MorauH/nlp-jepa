"""Tests for multi-GPU (DDP) support.

The sampler tests are pure CPU bookkeeping.  The real end-to-end check launches
``tests/ddp_worker.py`` under ``torchrun`` with two CPU (gloo) processes, which
is the only way to exercise ``DistributedDataParallel`` on a single-GPU box.
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from jepa.training import train
from jepa.training.data import LengthBucketBatchSampler, ShardedBatchSampler
from jepa.training.distributed import setup_distributed

REPO_ROOT = Path(__file__).resolve().parent.parent

requires_gloo = pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason="torch.distributed with the gloo backend is required",
)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------


def legacy_batches(lengths, batch_size, *, drop_last, seed, epoch, window_batches):
    """The pre-DDP batch construction, frozen as a regression reference."""
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(lengths), generator=generator).tolist()
    window = max(batch_size, batch_size * window_batches)
    batches = []
    for start in range(0, len(order), window):
        chunk = order[start : start + window]
        chunk.sort(key=lengths.__getitem__)
        for begin in range(0, len(chunk), batch_size):
            batch = chunk[begin : begin + batch_size]
            if len(batch) < batch_size and drop_last:
                continue
            batches.append(batch)
    permutation = torch.randperm(len(batches), generator=generator).tolist()
    return [batches[i] for i in permutation]


@pytest.mark.parametrize("n", [1, 7, 10, 37, 100, 101])
@pytest.mark.parametrize("batch_size", [1, 3, 8])
@pytest.mark.parametrize("drop_last", [False, True])
def test_single_process_sampler_is_unchanged(n, batch_size, drop_last):
    """world_size=1 must reproduce the old batch list exactly."""
    lengths = [(i * 13) % 97 for i in range(n)]
    sampler = LengthBucketBatchSampler(
        lengths,
        batch_size,
        drop_last=drop_last,
        seed=3,
        window_batches=2,
    )
    for epoch in range(3):
        sampler.set_epoch(epoch)
        expected = legacy_batches(
            lengths,
            batch_size,
            drop_last=drop_last,
            seed=3,
            epoch=epoch,
            window_batches=2,
        )
        assert list(sampler) == expected
        # The LR schedule uses len(); it must match what iteration yields.
        assert len(sampler) == len(expected)


@pytest.mark.parametrize("world_size", [2, 3, 4])
@pytest.mark.parametrize("n", [10, 37, 100, 101])
@pytest.mark.parametrize("batch_size", [2, 4])
def test_sampler_shards_disjoint_equal_counts(world_size, n, batch_size):
    """Each rank gets the same number of steps and a disjoint slice."""
    lengths = [(i * 7) % 31 for i in range(n)]
    epoch = 1

    def batches(rank, replicas):
        sampler = LengthBucketBatchSampler(
            lengths,
            batch_size,
            seed=5,
            window_batches=2,
            num_replicas=replicas,
            rank=rank,
        )
        sampler.set_epoch(epoch)
        return list(sampler), len(sampler)

    global_sampler = LengthBucketBatchSampler(
        lengths, batch_size, seed=5, window_batches=2
    )
    global_sampler.set_epoch(epoch)
    global_batches = list(global_sampler)
    usable = (len(global_batches) // world_size) * world_size

    shards = []
    for rank in range(world_size):
        shard, length = batches(rank, world_size)
        assert length == len(shard) == len(global_batches) // world_size
        # Deterministic stride of the shared global order.
        assert shard == global_batches[:usable][rank::world_size]
        shards.append(shard)

    assert len({len(shard) for shard in shards}) == 1
    seen = set()
    for shard in shards:
        indices = {index for batch in shard for index in batch}
        assert not indices & seen, "ranks must not share examples"
        seen |= indices

    global_set = {tuple(batch) for batch in global_batches}
    for shard in shards:
        for batch in shard:
            assert tuple(batch) in global_set

    # Deterministic for a given epoch (same seed and epoch on every rank).
    assert batches(0, world_size)[0] == shards[0]


def _batches_at_epoch(lengths, batch_size, world_size, rank, epoch):
    sampler = LengthBucketBatchSampler(
        lengths,
        batch_size,
        seed=5,
        window_batches=2,
        num_replicas=world_size,
        rank=rank,
    )
    sampler.set_epoch(epoch)
    return list(sampler)


def test_sampler_reshuffles_between_epochs():
    lengths = [(i * 7) % 31 for i in range(101)]
    first = _batches_at_epoch(lengths, 4, 2, 0, 0)
    assert first == _batches_at_epoch(lengths, 4, 2, 0, 0)
    assert first != _batches_at_epoch(lengths, 4, 2, 0, 1)


def test_sharded_validation_sampler_covers_every_example_once():
    """Validation sharding drops nothing and matches the sequential batches."""
    n, batch_size = 10, 3
    expected = [
        list(range(start, min(start + batch_size, n)))
        for start in range(0, n, batch_size)
    ]
    for world_size in (1, 2, 3, 4):
        union = []
        for rank in range(world_size):
            sampler = ShardedBatchSampler(
                n, batch_size, num_replicas=world_size, rank=rank
            )
            assert len(sampler) == len(list(sampler))
            union.extend(sampler)
        assert sorted(union) == sorted(expected)
        assert len(union) == math.ceil(n / batch_size)


# ---------------------------------------------------------------------------
# DDP assumptions
# ---------------------------------------------------------------------------


def test_setup_distributed_is_inert_without_a_launcher(monkeypatch):
    """No torchrun env = no process group, exactly like before DDP support."""
    for name in ("WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "RANK"):
        monkeypatch.delenv(name, raising=False)

    info = setup_distributed()

    assert not info.is_distributed
    assert info.is_main and info.world_size == 1 and info.backend is None
    assert not torch.distributed.is_initialized()


def test_default_config_gives_every_parameter_a_gradient():
    """``find_unused_parameters=False`` is only valid if nothing is unused.

    DDP errors out ("Expected to have finished reduction in the prior
    iteration") when a parameter that ``requires_grad`` sees no gradient, so
    the default stack in ``jepa/training/train.py`` is checked here: one
    forward/backward with both losses must touch every parameter.
    """

    class FakeTokenizer:
        pad_token_id = 0
        model_max_length = train.MODEL_MAX_LENGTH

        def __len__(self):
            return 32_000

    torch.manual_seed(0)
    model = train.build_model(FakeTokenizer())
    labels = torch.randint(1, 32_000, (2, 32))
    attention_mask = torch.ones_like(labels)

    jepa_out, decoder_out = model(
        labels=labels,
        attention_mask=attention_mask,
        training=True,
        right_padded=train.RIGHT_PADDED,
        loss_chunk_size=64,
    )
    loss = (
        train.JEPA_LOSS_WEIGHT * jepa_out["loss"]
        + train.DECODER_LOSS_WEIGHT * decoder_out["loss"]
    )
    loss.backward()

    unused = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert unused == []
    assert train.DDP_FIND_UNUSED_PARAMETERS is False


# ---------------------------------------------------------------------------
# Real two-process run (CPU / gloo)
# ---------------------------------------------------------------------------


def run_ddp_worker(tmp_path, *extra_args, timeout=900):
    """Launch tests/ddp_worker.py under torchrun with two CPU processes."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(REPO_ROOT / "tests" / "ddp_worker.py"),
        "--output-dir",
        str(tmp_path / "ckpt"),
        "--dump-dir",
        str(tmp_path / "dump"),
        *extra_args,
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def check_ddp_run(tmp_path, process, epochs=1):
    assert process.returncode == 0, (
        f"torchrun failed ({process.returncode})\n"
        f"--- stdout ---\n{process.stdout[-4000:]}\n"
        f"--- stderr ---\n{process.stderr[-4000:]}"
    )

    dump = tmp_path / "dump"
    dumps = {
        rank: json.loads((dump / f"rank{rank}.json").read_text())
        for rank in (0, 1)
    }
    assert dumps[0]["world_size"] == dumps[1]["world_size"] == 2
    # Rank 0 owns the checkpoint: one save per step (save_every_steps=1) plus
    # the end-of-epoch save; the other rank never writes.
    assert dumps[0]["saves"] > 0
    assert dumps[1]["saves"] == 0
    # Rank 0 loads history.json on --resume, the others only track the current
    # run, so compare the epoch that just finished.
    assert len(dumps[0]["history"]["train_loss"]) == epochs
    assert len(dumps[1]["history"]["train_loss"]) == 1
    assert (
        dumps[0]["history"]["train_loss"][-1]
        == dumps[1]["history"]["train_loss"][-1]
    )
    assert (
        dumps[0]["history"]["val_loss"][-1] == dumps[1]["history"]["val_loss"][-1]
    )

    # DDP keeps the replicas bitwise identical.
    params = {rank: torch.load(dump / f"rank{rank}.pt") for rank in (0, 1)}
    assert sorted(params[0]) == sorted(params[1])
    mismatched = [
        key
        for key in params[0]
        if not torch.equal(params[0][key], params[1][key])
    ]
    assert mismatched == []

    # Only rank 0 wrote a checkpoint, and it loads back.
    ckpt = tmp_path / "ckpt"
    assert (ckpt / "config.json").is_file()
    history = json.loads((ckpt / "history.json").read_text())
    assert len(history["train_loss"]) == epochs
    assert history["last_checkpoint"]["end_of_epoch"] is True

    from jepa.models.stack import JepaDecodeModel

    reloaded, info = JepaDecodeModel.from_pretrained(
        str(ckpt), output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    saved = reloaded.state_dict()
    assert all(
        torch.equal(value, params[0][key])
        for key, value in saved.items()
        if key in params[0]
    )


@requires_gloo
def test_two_process_cpu_training(tmp_path):
    """Two gloo ranks train, only rank 0 saves, the replicas stay identical."""
    process = run_ddp_worker(tmp_path)
    check_ddp_run(tmp_path, process)


@requires_gloo
def test_two_process_cpu_resume(tmp_path):
    """--resume loads on every rank and appends to rank 0's history."""
    check_ddp_run(tmp_path, run_ddp_worker(tmp_path), epochs=1)
    process = run_ddp_worker(tmp_path, "--resume")
    check_ddp_run(tmp_path, process, epochs=2)


@requires_gloo
@pytest.mark.skipif(
    not os.environ.get("JEPA_TEST_DDP_COMPILE"),
    reason="inductor compile takes ~90s and needs a C++ compiler; "
    "set JEPA_TEST_DDP_COMPILE=1 to run it",
)
def test_two_process_cpu_training_with_compile(tmp_path):
    """DDP + torch.compile (wrap DDP first, then compile) still trains."""
    process = run_ddp_worker(tmp_path, "--compile")
    check_ddp_run(tmp_path, process)
