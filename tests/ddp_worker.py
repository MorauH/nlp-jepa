"""Tiny two-process DDP run used by ``tests/test_ddp.py``.

Launched from the repository root with the ``torchrun`` entry point (the test
uses ``python -m torch.distributed.run``, which is the same launcher)::

    torchrun --standalone --nproc_per_node=2 tests/ddp_worker.py \
        --output-dir /tmp/out --dump-dir /tmp/dump

It trains a tiny :class:`JepaDecodeModel` on synthetic token ids on CPU (gloo,
via ``use_cpu=True``), then writes, per rank, the final parameters and the
number of times ``JepaTrainer._save`` ran.  The pytest test uses those files to
check that:

* every rank finished with identical parameters,
* only rank 0 saved checkpoints/history,
* the checkpoint rank 0 wrote reloads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

# `torchrun` puts this script's directory first on sys.path, not the CWD, so
# add the repository root before importing the package (see AGENTS.md).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from transformers import TrainingArguments

import jepa.training.trainer as trainer_module
from jepa.models.stack import JepaDecodeModel
from jepa.training.distributed import cleanup_distributed, setup_distributed
from jepa.training.trainer import JepaTrainer
from tests.test_smoke import tiny_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also exercise DDP + torch.compile (needs a CPU C++ compiler).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the checkpoint in --output-dir instead of a fresh model.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    info = setup_distributed(use_cpu=True)
    try:
        # Keep the test fast: the heuristic would otherwise start up to 8
        # dataloader workers per rank for a handful of batches.
        trainer_module.available_cpu_count = lambda: 2

        # Identical data on every rank (same generator seed), like a real
        # dataset read from disk.
        generator = torch.Generator().manual_seed(1234)
        rows = [
            torch.randint(1, 64, (12,), generator=generator).tolist()
            for _ in range(12)
        ]
        dataset = Dataset.from_dict({"input_ids": rows})

        if args.resume:
            # Every rank loads the checkpoint, like train.py does; only rank 0
            # reads the history it is also going to write back.
            model = JepaDecodeModel.from_pretrained(args.output_dir)
            history = None
            history_path = Path(args.output_dir) / "history.json"
            if info.is_main and history_path.exists():
                history = json.loads(history_path.read_text())
        else:
            torch.manual_seed(0)
            model = JepaDecodeModel(
                tiny_config(
                    trunk_layers=1, tie_embeddings=True, hint_every_layer=True
                )
            )
            history = None
        train_args = TrainingArguments(
            output_dir=args.output_dir,
            optim="adamw_torch",
            eval_strategy="epoch",
            learning_rate=1e-3,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            report_to=[],
            seed=0,
            use_cpu=True,
        )
        trainer = JepaTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            eval_dataset=dataset.select(range(6)),
            processing_class=SimpleNamespace(pad_token_id=0),
            history=history,
            use_amp=False,
            compile_model=args.compile,
            decoder_loss_chunk_size=8,
            pad_to_multiple_of=8,
            log_every=1,
            save_every_steps=1,
        )

        saves = []
        original_save = trainer._save

        def recording_save(output_dir):
            saves.append(1)
            original_save(output_dir)

        trainer._save = recording_save
        history = trainer.train()

        dump_dir = Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        # The raw (uncompiled, unwrapped) module: same keys the checkpoint has.
        parameters = {
            key: value.detach().cpu()
            for key, value in trainer._raw_model.state_dict().items()
        }
        torch.save(parameters, dump_dir / f"rank{info.rank}.pt")
        (dump_dir / f"rank{info.rank}.json").write_text(
            json.dumps(
                {
                    "rank": info.rank,
                    "world_size": info.world_size,
                    "saves": len(saves),
                    "history": history,
                }
            )
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
