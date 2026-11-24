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
        "candidate_pair_cos",
        "delta_ratio",
        "hindsight_shares",
        "live_slots",
        "effective_slots",
        "pi_nll",
        "constant_nll",
        "pi_gain",
        "pi_accuracy",
        "majority_accuracy",
        "mu_norm",
        "mu_distance_measured",
        "mu_distance_predicted",
        "mu_calibration_error",
    } <= set(heads)
    assert len(heads["hindsight_shares"]) == 4

    policies = metrics["policies"]
    assert {
        "copy",
        "hindsight",
        "pi_argmax",
        "pi_sample",
        "mu_safe",
        "mu",
        "fixed_slot",
        "hindsight_gain_over_mu",
        "commit_headroom",
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
    assert set(rollouts["best_of_k"]) == {"1", "4", "16"}
    for row in rollouts["best_of_k"].values():
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
        "no_hints",
    }
    # Every hint condition counts the same targets: 12-token rows with k=2 have
    # 11 candidate positions, of which the last partial block (no committed
    # future) is excluded, so 10 per row over 4 examples.
    assert decoder["tokens"] == 40
    assert metrics["context"]["tokens"] == 48
    assert "hint_use" in decoder and "commit_headroom" in decoder
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
        "best_of_16_beats_mu",
    } <= names


def test_best_of_k_is_non_increasing_in_k(saved):
    """Nested draws: best-of-K takes the first K of the same 16 rollouts."""
    best = collect(saved, depths=3)["rollouts"]["best_of_k"]

    for depth in range(3):
        values = [best[k][depth] for k in ("1", "4", "16")]
        assert values[0] >= values[1] >= values[2], (depth, values)


def test_best_of_accepts_a_single_k(saved):
    """K = 1 alone: one π-sample rollout scored by its mean error over 1..i."""
    metrics = collect(saved, depths=3, best_of=(1,))
    rollouts = metrics["rollouts"]
    assert set(rollouts["best_of_k"]) == {"1"}
    assert "best_of_1_beats_mu" in {v["name"] for v in metrics["verdicts"]}
    for value in rollouts["best_of_k"]["1"]:
        assert 0.0 <= value <= 4.0


def test_best_of_rejects_non_positive_k(saved):
    with pytest.raises(ValueError, match="best_of"):
        collect(saved, best_of=(0, 4))


def test_report_prints_best_of_k_table(saved, capsys):
    measure.print_report(collect(saved))
    out = capsys.readouterr().out
    table = out.split("best-of-K")[1].split("manifold distance")[0]
    for row in ("K=1", "K=4", "K=16", "mu", "hindsight"):
        assert f"\n{row} " in table, row


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


def test_single_candidate_reports_choice_metrics_as_na(saved, tmp_path, capsys):
    """P = 1 (the baseline): π, live slots and pair cosine are n/a."""
    from jepa import apply_overrides

    _, data_dir = saved
    model_dir = tmp_path / "p1"
    torch.manual_seed(0)
    config = apply_overrides(tiny_config(), {"predictor": {"prediction_count": 1}})
    JepaModel(config).save_pretrained(model_dir)

    metrics = collect((model_dir, data_dir))
    heads = metrics["heads"]
    assert metrics["context"]["candidates"] == 1
    for key in ("candidate_pair_cos", "live_slots", "pi_nll", "pi_gain", "pi_accuracy"):
        assert heads[key] is None, key
    assert "pi_uses_z_t" not in {verdict["name"] for verdict in metrics["verdicts"]}
    assert json.loads(json.dumps(metrics)) == metrics

    measure.print_report(metrics)
    assert "n/a" in capsys.readouterr().out


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
            "--best-of",
            "1",
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
    assert set(metrics["rollouts"]["best_of_k"]) == {"1", "2"}
    assert metrics["context"]["bank_size"] == 8


def _circle(angles):
    """Unit latents ``(1, T, 2)`` at the given angles, with a full mask."""
    z = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)[None]
    return [z], [torch.ones(1, z.shape[1], dtype=torch.long)]


def test_periodic_latent_rebounds_at_the_horizon():
    """Period 10 with k = 20: cos dips to -1 at lag 5 and returns to 1 at lag 20."""
    t = torch.arange(200, dtype=torch.float64)
    latents, masks = _circle(2 * torch.pi * t / 10)

    profile = measure.lag_profile(latents, masks, range(1, 21))

    assert profile[5] == pytest.approx(-1.0, abs=1e-6)
    assert profile[10] == pytest.approx(1.0, abs=1e-6)
    assert measure.periodic_rebound(profile, horizon=20) == pytest.approx(2.0, abs=1e-6)


def test_drifting_latent_does_not_rebound():
    t = torch.arange(200, dtype=torch.float64)
    latents, masks = _circle(0.01 * t)

    profile = measure.lag_profile(latents, masks, range(1, 21))

    assert profile[1] > profile[10] > profile[20]
    assert measure.periodic_rebound(profile, horizon=20) < 0


def test_fast_share_is_one_for_an_oscillation_and_zero_for_drift():
    """A centred window of one period removes the oscillation and keeps drift."""
    t = torch.arange(200, dtype=torch.float64)

    oscillating = measure.fast_share(*_circle(2 * torch.pi * t / 10), window=10)
    drifting = measure.fast_share(*_circle(0.01 * t), window=10)

    assert oscillating == pytest.approx(1.0, abs=1e-6)
    assert drifting == pytest.approx(0.0, abs=1e-3)


def test_collect_metrics_reports_the_lag_profile_and_hint_channels(saved):
    metrics = collect(saved)
    representation, decoder = metrics["representation"], metrics["decoder"]

    # k = 2: every lag 1..k, then 2k, 5k, 10k, 20k (none of the last three fit
    # in 12-token rows).
    assert list(representation["lag_cos"]) == ["1", "2", "4", "10", "20", "40"]
    assert representation["lag_cos"]["2"] == pytest.approx(representation["temporal_cos"])
    assert representation["lag_cos"]["40"] is None
    assert representation["periodic_rebound"] == pytest.approx(
        representation["lag_cos"]["2"] - representation["lag_cos"]["1"]
    )
    assert 0.0 <= representation["fast_share"] <= 1.0

    ce = decoder["ce"]
    assert "no_hints" in ce
    assert decoder["z_t_use"] == pytest.approx(ce["no_hints"] - ce["null"])
    assert decoder["future_use"] == pytest.approx(ce["null"] - ce["pi_argmax"])

    verdict = next(v for v in metrics["verdicts"] if v["name"] == "no_periodic_shortcut")
    assert verdict["threshold"] == 0.05 and verdict["comparison"] == "<"
    assert verdict["value"] == pytest.approx(representation["periodic_rebound"])


def test_joint_checkpoint_reports_the_weighted_token_term(saved):
    from jepa import apply_overrides

    model_dir, _ = saved
    model = JepaModel.from_pretrained(str(model_dir))
    model.config = apply_overrides(model.config, {"decoder_w": 0.25})
    model.save_pretrained(model_dir)

    terms = collect(saved)["loss"]["terms"]

    assert terms["token"]["weight"] == 0.25
    assert terms["token"]["weighted"] == pytest.approx(0.25 * terms["token"]["value"])


def test_report_prints_the_lag_profile_and_hint_channels(saved, capsys):
    measure.print_report(collect(saved))
    out = capsys.readouterr().out

    assert "lag profile" in out and "periodic rebound" in out and "fast share" in out
    assert "z_t use (no_hints - null)" in out
    assert "future use (null - pi-argmax)" in out
