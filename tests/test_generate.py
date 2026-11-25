"""Tests for decoded latent plans (jepa/inference/generate.py)."""

import json

import pytest
import torch
from datasets import Dataset, DatasetDict

from jepa import (
    DecoderConfig,
    EncoderConfig,
    JepaConfig,
    JepaModel,
    PredictorConfig,
    VISRegConfig,
)
from jepa.inference import generate

DEPTH = 3
HORIZON = 2


def tiny_model(prediction_count: int = 4, horizon_range=(HORIZON, HORIZON)) -> JepaModel:
    torch.manual_seed(0)
    encoder = EncoderConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    config = JepaConfig.from_encoder(
        encoder,
        predictor=PredictorConfig.from_encoder(
            encoder, prediction_count=prediction_count, num_hidden_layers=1
        ),
        decoder=DecoderConfig.from_encoder(
            encoder, sliding_window=8, num_hidden_layers=1
        ),
        horizon_range=list(horizon_range),
        visreg=VISRegConfig(num_slices=16),
    )
    return JepaModel(config).eval()


PROMPT = [5, 9, 13, 2, 40, 7]


def test_plan_decodes_every_condition_for_depth_blocks_of_k_tokens():
    model = tiny_model()

    def run():
        return generate.plan_prompt(
            model, PROMPT, depth=DEPTH, horizon=HORIZON, samples=2, generator_seed=0
        )

    records = run()

    conditions = [(r["condition"], r["sample"]) for r in records]
    assert conditions == [
        ("pi_argmax", 0),
        ("pi_sample", 0),
        ("pi_sample", 1),
        ("mu", 0),
    ]
    for record in records:
        assert len(record["token_ids"]) == DEPTH
        assert all(len(block) == HORIZON for block in record["token_ids"])
    # Greedy token decoding and a seeded commit sampler: same plans twice.
    assert run() == records


def test_plan_encodes_the_prompt_exactly_once():
    model = tiny_model()
    calls = []
    model.encoder.register_forward_hook(lambda *args: calls.append(1))

    generate.plan_prompt(
        model, PROMPT, depth=DEPTH, horizon=HORIZON, samples=2, generator_seed=0
    )

    assert len(calls) == 1


class WordTokenizer:
    """Token ``i`` is the word ``w{i}``; enough for the CLI."""

    pad_token_id = 0
    eos_token_id = None

    def __len__(self) -> int:
        return 64

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False):
        ids = [[int(word[1:]) for word in text.split()] for text in texts]
        return {"input_ids": torch.tensor(ids)}

    def decode(self, ids) -> str:
        return "".join(f" w{int(i)}" for i in ids)


@pytest.fixture()
def saved(tmp_path):
    model_dir = tmp_path / "model"
    tiny_model().save_pretrained(model_dir)
    baseline_dir = tmp_path / "baseline"
    tiny_model(prediction_count=1).save_pretrained(baseline_dir)

    rows = [torch.randint(1, 64, (n,)).tolist() for n in (20, 20, 20, 5)]
    data_dir = tmp_path / "data"
    DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": rows}),
            "validation": Dataset.from_dict({"input_ids": rows}),
        }
    ).save_to_disk(str(data_dir))
    return model_dir, baseline_dir, data_dir


def test_plan_cli_writes_records_with_references(saved, tmp_path, monkeypatch):
    model_dir, baseline_dir, data_dir = saved
    monkeypatch.setattr(generate, "load_tokenizer", lambda path: WordTokenizer())
    out_json, out_html = tmp_path / "plans.json", tmp_path / "plans.html"

    generate.main([
        "--model", str(model_dir),
        "--baseline-model", str(baseline_dir),
        "--fineweb-dir", str(data_dir),
        "--plan-depth", str(DEPTH),
        "--samples", "2",
        "--num-prompts", "2",
        "--prompt-length", "8",
        "--prompt", "w3 w4 w5",
        "--json", str(out_json),
        "--html", str(out_html),
        "--device", "cpu",
    ])

    plans = json.loads(out_json.read_text())
    records = plans["records"]
    prompts = {r["prompt_index"] for r in records}
    # Two validation chunks long enough for prompt + depth * k, plus --prompt.
    assert prompts == {0, 1, 2}
    by_prompt = {i: [r for r in records if r["prompt_index"] == i] for i in prompts}
    for index in (0, 1):
        rows = by_prompt[index]
        assert {r["condition"] for r in rows} == {
            "pi_argmax", "pi_sample", "mu", "baseline"
        }
        assert all(r["source"] == "validation" for r in rows)
        assert all(len(r["reference"]) == DEPTH for r in rows)
        assert all(len(r["prompt"].split()) == 8 for r in rows)
        assert all(len(r["blocks"]) == DEPTH for r in rows)
    assert all(r["reference"] is None for r in by_prompt[2])
    assert by_prompt[2][0]["prompt"] == "w3 w4 w5"
    assert plans["context"]["temperature"] == 0.0
    assert "<table" in out_html.read_text()


# ---------------------------------------------------------------------------
# Random horizon: one k per generated sequence (ADR 0003)
# ---------------------------------------------------------------------------


def test_generate_uses_one_k_for_every_block_and_repeats_with_the_seed(monkeypatch):
    model = tiny_model(horizon_range=(2, 6))
    used = []
    original = model.decoder.generate

    def record(*args, k, **kwargs):
        used.append(k)
        return original(*args, k=k, **kwargs)

    monkeypatch.setattr(model.decoder, "generate", record)

    def run(seed):
        used.clear()
        _, ids, horizon = generate.generate(
            model, WordTokenizer(), "w3 w4 w5", 3, seed=seed, temperature=0.0,
            device=torch.device("cpu"),
        )
        assert used == [horizon] * 3 and len(ids) == 3 * horizon
        return horizon

    horizons = [run(seed) for seed in range(20)]
    assert horizons == [run(seed) for seed in range(20)]
    assert set(horizons) == {2, 3, 4, 5, 6}
    assert 2 <= run(None) <= 6


@pytest.fixture()
def random_saved(tmp_path):
    """Random-horizon checkpoints; chunks of exactly prompt + DEPTH * hi tokens."""
    tmp_path = tmp_path / "random"
    model_dir, baseline_dir = tmp_path / "model", tmp_path / "baseline"
    tiny_model(horizon_range=(2, 6)).save_pretrained(model_dir)
    tiny_model(prediction_count=1, horizon_range=(2, 6)).save_pretrained(baseline_dir)
    # 8 + 3 * 6 = 26 tokens are enough for every k; 25 are not.
    rows = [torch.randint(1, 64, (n,)).tolist() for n in (26,) * 8 + (25,) * 8]
    data_dir = tmp_path / "data"
    DatasetDict(
        {name: Dataset.from_dict({"input_ids": rows}) for name in ("train", "validation")}
    ).save_to_disk(str(data_dir))
    return model_dir, baseline_dir, data_dir, rows


def test_plans_share_one_k_per_prompt(random_saved, tmp_path, monkeypatch):
    model_dir, baseline_dir, data_dir, rows = random_saved
    monkeypatch.setattr(generate, "load_tokenizer", lambda path: WordTokenizer())

    def run(name, seed):
        out, page = tmp_path / f"{name}.json", tmp_path / f"{name}.html"
        generate.main([
            "--model", str(model_dir), "--baseline-model", str(baseline_dir),
            "--fineweb-dir", str(data_dir), "--plan-depth", str(DEPTH),
            "--samples", "2", "--num-prompts", "16", "--prompt-length", "8",
            "--prompt", "w3 w4 w5", "--seed", str(seed),
            "--json", str(out), "--html", str(page), "--device", "cpu",
        ])
        return json.loads(out.read_text()), page.read_text()

    document, page = run("a", 1)
    records = document["records"]
    assert document["context"]["horizon_range"] == [2, 6]
    by_prompt = {}
    for record in records:
        by_prompt.setdefault(record["prompt_index"], []).append(record)
    # The eight long chunks plus --prompt; the 25-token chunks have no room.
    assert len(by_prompt) == 9

    horizons = []
    for prompt_records in by_prompt.values():
        (k,) = {r["horizon"] for r in prompt_records}
        horizons.append(k)
        assert f"k = {k}" in page
        assert {r["condition"] for r in prompt_records} == {
            "pi_argmax", "pi_sample", "mu", "baseline"
        }
        for r in prompt_records:
            assert [len(block) for block in r["token_ids"]] == [k] * DEPTH
            if r["reference"] is None:
                continue
            assert [len(block.split()) for block in r["reference"]] == [k] * DEPTH
            # The reference is the DEPTH * k tokens right after the prompt.
            words = r["prompt"].split() + " ".join(r["reference"]).split()
            assert [int(w[1:]) for w in words] in [
                row[: 8 + DEPTH * k] for row in rows
            ]
    assert len(set(horizons)) > 1

    assert run("b", 1)[0]["records"] == records
    other = {r["prompt_index"]: r["horizon"] for r in run("c", 2)[0]["records"]}
    assert list(other.values()) != horizons


def test_the_baseline_must_have_the_same_horizon_range(saved, random_saved, monkeypatch):
    model_dir = random_saved[0]
    _, fixed_baseline, data_dir = saved
    monkeypatch.setattr(generate, "load_tokenizer", lambda path: WordTokenizer())
    with pytest.raises(ValueError, match="horizon_range"):
        generate.main([
            "--model", str(model_dir), "--baseline-model", str(fixed_baseline),
            "--fineweb-dir", str(data_dir), "--plan-depth", "2", "--device", "cpu",
        ])
