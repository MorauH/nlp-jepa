"""Smoke tests for ``python -m jepa measure`` (jepa/evaluation/measure.py)."""

import json

import pytest
import torch
from datasets import Dataset, DatasetDict, load_from_disk

from jepa import (
    DecoderConfig,
    EncoderConfig,
    JepaConfig,
    JepaModel,
    PredictorConfig,
    VISRegConfig,
)
from jepa.evaluation import measure

SECTION_KEYS = {
    "context",
    "representation",
    "loss",
    "heads",
    "policies",
    "rollouts",
    "decoder",
    "verdicts",
}


def tiny_config() -> JepaConfig:
    encoder = EncoderConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    predictor = PredictorConfig.from_encoder(
        encoder,
        prediction_count=4,
        num_hidden_layers=1,
    )
    decoder = DecoderConfig.from_encoder(
        encoder,
        sliding_window=8,
        num_hidden_layers=1,
    )
    return JepaConfig.from_encoder(
        encoder,
        predictor=predictor,
        decoder=decoder,
        pred_horizon=2,
        visreg=VISRegConfig(num_slices=16),
    )


class FakeTokenizer:
    pad_token_id = 0
    model_max_length = 32

    def __len__(self) -> int:
        return 64


@pytest.fixture()
def saved(tmp_path):
    """A tiny checkpoint plus a tiny prepared dataset (train + validation)."""
    torch.manual_seed(0)
    model_dir = tmp_path / "model"
    JepaModel(tiny_config()).save_pretrained(model_dir)

    rows = [torch.randint(1, 64, (12,)).tolist() for _ in range(6)]
    data_dir = tmp_path / "data"
    DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": rows}),
            "validation": Dataset.from_dict({"input_ids": rows[:4]}),
        }
    ).save_to_disk(str(data_dir))
    return model_dir, data_dir


def collect(saved, **overrides):
    model_dir, data_dir = saved
    tokenizer = FakeTokenizer()
    model = JepaModel.from_pretrained(str(model_dir)).eval()
    dataset = load_from_disk(str(data_dir))
    batches = measure.load_query_batches(
        dataset["validation"],
        tokenizer,
        num_examples=4,
        max_length=12,
        batch_size=2,
        seed=0,
    )
    bank = measure.build_bank(
        model,
        dataset["train"],
        tokenizer,
        size=16,
        max_length=12,
        batch_size=2,
        seed=0,
    )
    options = dict(
        depths=2,
        rollout_positions=8,
        seed=0,
        loss_chunk_size=4,
    )
    options.update(overrides)
    return measure.collect_metrics(model, batches, bank, **options)


def test_collect_metrics_has_every_section_and_nested_key(saved):
    metrics = collect(saved)

    assert set(metrics) == SECTION_KEYS
    assert metrics["context"]["examples"] == 4
    assert metrics["context"]["bank_size"] == 16
    assert metrics["context"]["pred_horizon"] == 2
    assert metrics["context"]["candidates"] == 4

    representation = metrics["representation"]
    assert {
        "per_dim_std",
        "mean_vector_norm",
        "effective_rank",
        "temporal_cos",
        "visreg",
    } <= set(representation)
    assert set(representation["visreg"]) == {"scale", "shape", "center"}

    loss = metrics["loss"]
    assert set(loss["terms"]) == {
        "scale",
        "shape",
        "center",
        "move",
        "pi",
        "mu",
        "diversity",
    }
    assert {"assignment_entropy", "kl_to_pi"} <= set(loss["floor"])

    heads = metrics["heads"]
    assert {
        "best_d",
        "candidate_pair_cos",
        "delta_ratio",
        "win_shares",
        "live_slots",
        "effective_slots",
        "pi_nll",
        "constant_nll",
        "pi_gain",
        "pi_accuracy",
        "majority_accuracy",
        "mu_norm",
        "mu_dir_distance_measured",
        "mu_dir_distance_predicted",
        "mu_calibration_error",
    } <= set(heads)
    assert len(heads["win_shares"]) == 4

    policies = metrics["policies"]
    assert {
        "copy",
        "hindsight",
        "pi_argmax",
        "pi_sample",
        "mu_safe",
        "mu_direction",
        "fixed_slot",
        "oracle_gain_over_mu_direction",
        "selection_headroom",
    } <= set(policies)
    assert set(policies["fixed_slot"]) == {"index", "distance"}

    rollouts = metrics["rollouts"]
    assert rollouts["depths"] == 2
    assert set(rollouts["error"]) == {
        "hindsight",
        "pi_argmax",
        "pi_sample",
        "mu",
    }
    assert set(rollouts["manifold"]) == {
        "hindsight",
        "pi_argmax",
        "pi_sample",
        "mu",
        "real",
    }
    for row in (*rollouts["error"].values(), *rollouts["manifold"].values()):
        assert len(row) == 2

    decoder = metrics["decoder"]
    assert set(decoder["ce"]) == {
        "pi_argmax",
        "pi_sample",
        "mu_safe",
        "hindsight",
        "real",
        "shuffled",
        "null",
    }
    assert "hint_use" in decoder and "selection_headroom" in decoder
    assert set(decoder["specificity"]) == {
        "shuffled_minus_hindsight",
        "pi_argmax_minus_hindsight",
    }

    verdicts = metrics["verdicts"]
    assert verdicts and all(
        set(verdict)
        == {"name", "metric", "value", "threshold", "comparison", "ok"}
        for verdict in verdicts
    )
    names = {verdict["name"] for verdict in verdicts}
    assert {
        "pi_uses_z_t",
        "candidates_beat_mean",
        "decoder_uses_commitment",
        "mu_calibrated",
        "encoder_center",
        "encoder_effective_rank",
        "committed_on_manifold_depth_2",
    } <= names


def test_loss_total_is_the_weighted_sum_of_its_terms(saved):
    loss = collect(saved)["loss"]

    weighted = sum(
        term["weight"] * term["value"] for term in loss["terms"].values()
    )
    torch.testing.assert_close(
        torch.tensor(loss["total"]), torch.tensor(weighted)
    )
    for term in loss["terms"].values():
        torch.testing.assert_close(
            torch.tensor(term["weighted"]),
            torch.tensor(term["weight"] * term["value"]),
        )


def test_metrics_are_json_serializable(saved):
    metrics = collect(saved)
    # Plain floats/ints/lists only: no tensors, no numpy scalars.
    assert json.loads(json.dumps(metrics)) == metrics


def test_measure_cli_writes_the_same_dict(saved, monkeypatch, tmp_path):
    model_dir, data_dir = saved
    tokenizer = FakeTokenizer()
    monkeypatch.setattr(measure, "load_tokenizer", lambda path: tokenizer)

    output = tmp_path / "metrics.json"
    measure.main(
        [
            "--model",
            str(model_dir),
            "--fineweb-dir",
            str(data_dir),
            "--num-examples",
            "2",
            "--max-length",
            "12",
            "--batch-size",
            "2",
            "--bank-size",
            "8",
            "--rollout-positions",
            "2",
            "--depths",
            "2",
            "--loss-chunk-size",
            "4",
            "--device",
            "cpu",
            "--seed",
            "1",
            "--json",
            str(output),
        ]
    )

    metrics = json.loads(output.read_text())
    assert set(metrics) == SECTION_KEYS
    assert metrics["context"]["checkpoint"] == str(model_dir)
    assert metrics["context"]["device"] == "cpu"
    assert metrics["rollouts"]["positions"] == 2
    assert metrics["context"]["bank_size"] == 8
