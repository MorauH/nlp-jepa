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
    "probe",
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
        horizon_range=[2, 2],
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
    probe_batches = measure.load_query_batches(
        dataset["train"], tokenizer, num_examples=6, max_length=12, batch_size=2,
        seed=1,
    )
    options = dict(
        depths=2,
        rollout_positions=8,
        seed=0,
        loss_chunk_size=4,
        probe_batches=probe_batches,
    )
    options.update(overrides)
    return measure.collect_metrics(model, batches, bank, **options)


def test_collect_metrics_has_every_section_and_nested_key(saved):
    metrics = collect(saved)

    assert set(metrics) == SECTION_KEYS
    assert metrics["context"]["examples"] == 4
    assert metrics["context"]["bank_size"] == 16
    assert metrics["context"]["horizon_range"] == [2, 2]
    assert metrics["context"]["horizons"] == [2]
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
        "real_noisy",
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
        "hindsight_carries_content",
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


@pytest.mark.parametrize("period", [7, 10])
def test_max_rebound_detects_a_period_that_need_not_divide_k(period):
    t = torch.arange(400, dtype=torch.float64)
    profile = measure.lag_profile(*_circle(2 * torch.pi * t / period), range(1, 31))

    # cos returns to 1 after one period, from -1 (10) or cos(6/7 pi) (7).
    lowest = {7: -0.9009688679, 10: -1.0}[period]
    assert measure.max_rebound(profile, 30) == pytest.approx(1 - lowest, abs=1e-5)


def test_max_rebound_ignores_a_plain_drift_and_missing_lags():
    t = torch.arange(200, dtype=torch.float64)
    profile = measure.lag_profile(*_circle(0.01 * t), range(1, 31))
    assert profile[1] > profile[10] > profile[30]
    assert measure.max_rebound(profile, 30) < 0

    assert measure.max_rebound({1: 0.9, 2: None, 3: 0.95}, 3) == pytest.approx(0.05)
    assert measure.max_rebound({1: 0.9, 2: None}, 2) is None
    assert measure.max_rebound({1: 0.9}, 1) is None


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

    # Every lag up to the longest horizon, then the fixed long lags (none of
    # which fit in 12-token rows).
    assert list(representation["lag_cos"]) == ["1", "2", "40", "100", "200", "400"]
    assert representation["lag_cos"]["2"] == pytest.approx(representation["temporal_cos"])
    assert representation["lag_cos"]["40"] is None
    assert representation["max_rebound"] == pytest.approx(
        representation["lag_cos"]["2"] - representation["lag_cos"]["1"]
    )
    assert 0.0 <= representation["fast_share"] <= 1.0

    ce = decoder["ce"]
    assert "no_hints" in ce
    assert decoder["z_t_use"] == pytest.approx(ce["no_hints"] - ce["null"])
    assert decoder["future_use"] == pytest.approx(ce["null"] - ce["pi_argmax"])

    verdict = next(v for v in metrics["verdicts"] if v["name"] == "no_periodic_shortcut")
    assert verdict["threshold"] == 0.05 and verdict["comparison"] == "<"
    assert verdict["value"] == pytest.approx(representation["max_rebound"])
    assert verdict["metric"] == "representation.max_rebound"


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

    assert "lag profile" in out and "max rebound" in out and "fast share" in out
    assert "z_t use (no_hints - null)" in out
    assert "future use (null - pi-argmax)" in out


# ---------------------------------------------------------------------------
# Random horizon: scored over fixed evaluation horizons (ADR 0003)
# ---------------------------------------------------------------------------


def test_evaluation_horizons_cover_the_range_with_distinct_residues():
    assert measure.evaluation_horizons([10, 30]) == [11, 14, 17, 20, 23, 26, 29]
    assert measure.evaluation_horizons([20, 20]) == [20]
    assert measure.evaluation_horizons([2, 6]) == [2, 3, 4, 5, 6]
    for lo in (1, 5, 10):
        for hi in range(lo, 70):
            horizons = measure.evaluation_horizons([lo, hi])
            assert lo <= horizons[0] and horizons[-1] <= hi
            assert len({k % 10 for k in horizons}) == len(horizons)
            # Not bunched up at one end or in the middle of the range.
            assert horizons[-1] - horizons[0] >= 0.75 * (hi - lo) or hi - lo < 12


def test_slot_horizon_information_is_zero_or_maximal():
    import math

    horizons, slots = 7, 8
    ignoring_k = torch.tensor([[50, 10, 5, 5, 10, 10, 5, 5]] * horizons)
    assert measure.slot_horizon_information(ignoring_k) == pytest.approx(0.0, abs=1e-9)

    by_phase = torch.zeros(horizons, slots, dtype=torch.long)
    for k in range(horizons):
        by_phase[k, (3 * k) % slots] = 100
    assert measure.slot_horizon_information(by_phase) == pytest.approx(math.log(7))


def test_document_share_is_one_for_a_document_code_and_zero_without_one():
    torch.manual_seed(0)
    masks = [torch.ones(4, 60, dtype=torch.long)]
    per_document = [torch.randn(4, 1, 8).expand(4, 60, 8)]
    assert measure.document_share(per_document, masks) == pytest.approx(1.0)
    # The same trajectory in every document says nothing about which one it is.
    shared = [torch.randn(1, 60, 8).expand(4, 60, 8)]
    assert measure.document_share(shared, masks) == pytest.approx(0.0, abs=1e-6)
    # Padding and the document start (first 20 positions) are left out.
    noisy = per_document[0].clone()
    noisy[:, :20] += torch.randn(4, 20, 8)
    noisy[1, 50:] += 5
    short = masks[0].clone()
    short[1, 50:] = 0
    assert measure.document_share([noisy], [short]) == pytest.approx(1.0)
    assert measure.document_share(per_document, [masks[0] * 0]) is None


@pytest.fixture()
def random_horizon(tmp_path):
    """A model with horizon_range [2, 6], query batches and a bank."""
    from jepa import apply_overrides

    torch.manual_seed(0)
    config = apply_overrides(tiny_config(), {"horizon_range": [2, 6], "decoder_w": 0.3})
    model = JepaModel(config).eval()
    rows = [
        {"input_ids": torch.randint(1, 64, (n,)).tolist()} for n in (32, 32, 27, 20)
    ]
    from jepa.training.data import tokenize_batch

    batches = [tokenize_batch(rows[i : i + 2], FakeTokenizer()) for i in (0, 2)]
    bank = torch.randn(16, 32)
    return model, batches, bank


def measure_random(random_horizon, **overrides):
    options = dict(depths=2, rollout_positions=8, seed=0, loss_chunk_size=4)
    return measure.collect_metrics(*random_horizon, **{**options, **overrides})


def test_policies_report_each_horizon_and_their_mean(random_horizon, monkeypatch):
    monkeypatch.setattr(measure, "STEADY_STATE_START", 8)
    from jepa.models.commitment import distance

    model, batches, _ = random_horizon
    policies = measure_random(random_horizon)["policies"]

    def expected(k, start):
        """Worked out position by position from the encoder and predictor."""
        rows = []
        for labels, mask in batches:
            with torch.no_grad():
                z = model.encode(labels)
                futures = model.predictor(z)
            for row, length in enumerate(mask.sum(-1).tolist()):
                for t in range(start, length - k):
                    target = z[row, t + k]
                    candidates = distance(futures.candidates[row, t], target)
                    rows.append((
                        distance(z[row, t], target).item(),
                        candidates.min().item(),
                        candidates[futures.pi[row, t].argmax()].item(),
                        distance(futures.mu[row, t], target).item(),
                    ))
        return torch.tensor(rows).mean(dim=0).tolist()

    names = ("copy", "hindsight", "pi_argmax", "mu")
    assert policies["steady_state"]["from_position"] == 8
    for section, start in ((policies, 0), (policies["steady_state"], 8)):
        assert list(section["by_horizon"]) == ["2", "3", "4", "5", "6"]
        for k, row in section["by_horizon"].items():
            assert [row[n] for n in names] == pytest.approx(
                expected(int(k), start), abs=1e-5
            )
        for name in (*names, "pi_sample", "mu_safe", "hindsight_gain_over_mu"):
            values = [row[name] for row in section["by_horizon"].values()]
            assert section[name] == pytest.approx(sum(values) / 5)


def test_short_queries_have_no_steady_state_rows(random_horizon, capsys):
    metrics = measure_random(random_horizon)
    assert metrics["policies"]["steady_state"] is None
    assert metrics["heads"]["steady_state"] is None
    measure.print_report(metrics)
    assert "steady state" in capsys.readouterr().out


def test_every_section_averages_over_the_evaluation_horizons(random_horizon):
    from jepa.models.commitment import Hindsight, distance, rollout

    model, batches, _ = random_horizon
    metrics = measure_random(random_horizon, rollout_positions=10_000)
    horizons = [2, 3, 4, 5, 6]

    assert metrics["context"]["horizon_range"] == [2, 6]
    assert metrics["context"]["horizons"] == horizons
    representation = metrics["representation"]
    lag_cos = representation["lag_cos"]
    assert list(lag_cos)[:6] == ["1", "2", "3", "4", "5", "6"]
    assert representation["temporal_cos"] == pytest.approx(
        sum(lag_cos[str(k)] for k in horizons) / 5
    )

    # Decoder: blocks of k tokens for each k.  A block counts if it ends inside
    # the sequence the batch is padded to and its targets are real.
    tokens = 0
    for k in horizons:
        for labels, mask in batches:
            for length in mask.sum(-1).tolist():
                tokens += sum(
                    i // k * k + k <= labels.shape[1] - 1 for i in range(length - 1)
                )
    assert metrics["decoder"]["tokens"] == tokens

    # Rollouts: the same starts for every k, with room for depths * hi.
    rollouts = metrics["rollouts"]
    assert rollouts["horizons"] == horizons
    lengths = [n for _, mask in batches for n in mask.sum(-1).tolist()]
    starts = [(row, t) for row, n in enumerate(lengths) for t in range(n - 2 * 6)]
    assert rollouts["positions"] == len(starts)
    latents = [z for labels, _ in batches for z in model.encode(labels).detach()]
    errors = []
    for row, t in starts:
        for k in horizons:
            targets = torch.stack([latents[row][t + k], latents[row][t + 2 * k]])
            with torch.no_grad():
                path = rollout(
                    model.predictor, latents[row][t], Hindsight(), 2, targets=targets
                ).future
            errors.append(distance(path, targets))
    assert rollouts["error"]["hindsight"] == pytest.approx(
        torch.stack(errors).mean(dim=0).tolist(), abs=1e-5
    )

    assert metrics["loss"]["terms"]["token"]["weight"] == 0.3
    assert json.loads(json.dumps(metrics)) == metrics


def test_slot_horizon_information_is_reported_only_when_both_can_vary(
    random_horizon, saved, monkeypatch, capsys
):
    import math

    from jepa import apply_overrides

    monkeypatch.setattr(measure, "STEADY_STATE_START", 8)
    metrics = measure_random(random_horizon)
    heads = metrics["heads"]
    # Four slots and five horizons: the slot's entropy is the smaller bound.
    assert heads["slot_horizon_mi_max"] == pytest.approx(math.log(4))
    for section in (heads, heads["steady_state"]):
        assert 0.0 <= section["slot_horizon_mi"] <= math.log(4)
        assert section["live_slots"] >= 1
    measure.print_report(metrics)
    out = capsys.readouterr().out
    assert "slot-horizon MI" in out and "document share" in out

    # Fixed k.
    assert collect(saved)["heads"]["slot_horizon_mi"] is None
    # P = 1.
    model, batches, bank = random_horizon
    single = JepaModel(
        apply_overrides(model.config, {"predictor": {"prediction_count": 1}})
    ).eval()
    heads = measure_random((single, batches, bank))["heads"]
    assert heads["slot_horizon_mi"] is None
    assert heads["steady_state"]["live_slots"] is None


def test_queries_must_exceed_the_longest_horizon(random_horizon):
    model, batches, bank = random_horizon
    short = [(labels[:, :6], mask[:, :6]) for labels, mask in batches]
    with pytest.raises(ValueError, match="horizon"):
        measure_random((model, short, bank))


# ---------------------------------------------------------------------------
# Ticket 008: real-hint decoder conditions, the next-block probe, pooled center
# ---------------------------------------------------------------------------


def test_decoder_reports_the_render_gap_and_the_noisy_real_condition(saved):
    decoder = collect(saved)["decoder"]
    ce = decoder["ce"]
    assert decoder["render_gap"] == pytest.approx(ce["pi_argmax"] - ce["real"])
    assert decoder["commit_headroom"] == pytest.approx(ce["pi_argmax"] - ce["hindsight"])
    # The noise is drawn, so the noisy real hint is a different condition.
    assert ce["real_noisy"] != ce["real"]


def test_real_noisy_is_the_real_hint_without_noise(saved, monkeypatch):
    monkeypatch.setattr(measure, "hint_noise_scale", lambda d: torch.zeros_like(d))
    ce = collect(saved)["decoder"]["ce"]
    assert ce["real_noisy"] == pytest.approx(ce["real"])


def test_encoder_center_verdict_reads_the_pooled_latents(saved):
    model_dir, data_dir = saved
    metrics = collect(saved)
    model = JepaModel.from_pretrained(str(model_dir)).eval()
    tokenizer = FakeTokenizer()
    batches = measure.load_query_batches(
        load_from_disk(str(data_dir))["validation"], tokenizer,
        num_examples=4, max_length=12, batch_size=2, seed=0,
    )
    with torch.no_grad():
        pooled = torch.cat([
            model.encode(labels)[mask.bool()].double() for labels, mask in batches
        ])
    expected = (pooled.mean(dim=0) ** 2).mean().item()

    representation = metrics["representation"]
    assert representation["pooled_center"] == pytest.approx(expected, rel=1e-5)
    verdict = next(v for v in metrics["verdicts"] if v["name"] == "encoder_center")
    assert verdict["metric"] == "representation.pooled_center"
    assert verdict["value"] == pytest.approx(expected, rel=1e-5)


def test_bag_of_words_probe_matches_a_dense_ridge_regression():
    torch.manual_seed(0)
    dim, vocab = 6, 11
    fit_x, test_x = torch.randn(200, dim), torch.randn(50, dim)
    fit_tokens = torch.randint(0, vocab, (200, 3))
    test_tokens = torch.randint(0, vocab, (50, 4))  # rows of another k pool too

    probe = measure._BagOfWordsProbe(dim, vocab, "cpu")
    probe.add(fit_x[:120], fit_tokens[:120])
    probe.add(fit_x[120:], fit_tokens[120:])
    probe.solve()
    errors, reference = probe.errors(test_x, test_tokens)

    def bag(tokens):
        counts = torch.zeros(tokens.shape[0], vocab, dtype=torch.float64)
        counts.scatter_add_(1, tokens, torch.ones_like(tokens, dtype=torch.float64))
        return counts / tokens.shape[1]

    x, y = fit_x.double(), bag(fit_tokens)
    mean_x, mean_y = x.mean(dim=0), y.mean(dim=0)
    xc = x - mean_x
    cxx = xc.T @ xc
    ridge = measure.PROBE_RIDGE * cxx.trace() / dim
    weight = torch.linalg.solve(cxx + ridge * torch.eye(dim, dtype=x.dtype), xc.T @ y)
    target = bag(test_tokens)
    prediction = mean_y + (test_x.double() - mean_x) @ weight
    torch.testing.assert_close(errors, ((target - prediction) ** 2).sum(dim=-1))
    torch.testing.assert_close(reference, ((target - mean_y) ** 2).sum(dim=-1))


def test_probe_section_reports_r2_and_gains(saved, capsys):
    metrics = collect(saved)
    probe = metrics["probe"]

    assert set(probe["r2"]) == {"z_t", "mu", "pi_argmax", "hindsight"}
    for name, gain in probe["gain_over_z_t"].items():
        assert gain == pytest.approx(probe["r2"][name] - probe["r2"]["z_t"])
    assert probe["hindsight_over_mu"] == pytest.approx(
        probe["r2"]["hindsight"] - probe["r2"]["mu"]
    )
    # 6 train rows and 4 query rows of 12 tokens, k = 2: 10 positions each.
    assert probe["fit_positions"] == 60 and probe["positions"] == 40
    verdict = next(
        v for v in metrics["verdicts"] if v["name"] == "hindsight_carries_content"
    )
    assert verdict["value"] == pytest.approx(probe["hindsight_over_mu"])

    measure.print_report(metrics)
    out = capsys.readouterr().out
    assert "next-block probe" in out and "render gap" in out and "pooled center" in out


def test_probe_is_skipped_without_train_batches(saved, capsys):
    metrics = collect(saved, probe_batches=None)
    assert metrics["probe"] is None
    assert "hindsight_carries_content" not in {v["name"] for v in metrics["verdicts"]}
    measure.print_report(metrics)
    assert "next-block probe: n/a" in capsys.readouterr().out


def test_real_hint_checkpoint_measures(saved):
    from jepa import apply_overrides

    model_dir, _ = saved
    model = JepaModel.from_pretrained(str(model_dir))
    model.config = apply_overrides(
        model.config, {"decoder_w": 0.3, "decoder_hint": "real"}
    )
    model.save_pretrained(model_dir)

    metrics = collect(saved)

    assert metrics["loss"]["terms"]["token"]["weight"] == 0.3
    assert "hint_noise" in metrics["loss"]["diagnostics"]
    assert json.loads(json.dumps(metrics)) == metrics
