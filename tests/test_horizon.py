"""Random hidden horizon (ADR 0003): one k per batch, never shown to the predictor."""

import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

from jepa import JepaConfig, JepaModel, apply_overrides, build_block_hints
from jepa.horizon import cycle_horizon, draw_horizon
from jepa.models.commitment import Futures, Hindsight, PiArgmax

from test_smoke import CrashingDataset, tiny_config


def random_horizon_model(lo: int = 2, hi: int = 6, decoder_w: float = 0.5) -> JepaModel:
    torch.manual_seed(0)
    return JepaModel(
        apply_overrides(
            tiny_config(), {"horizon_range": [lo, hi], "decoder_w": decoder_w}
        )
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_the_default_horizon_is_random_and_round_trips(tmp_path):
    config = JepaConfig()
    assert config.horizon_range == [10, 30]

    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["horizon_range"] == [10, 30]
    assert "pred_horizon" not in saved
    assert JepaConfig.from_pretrained(tmp_path).horizon_range == [10, 30]


def test_an_old_checkpoint_config_loads_as_a_fixed_horizon(tmp_path):
    """Old config.json files store pred_horizon and no range."""
    JepaConfig().save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    old = json.loads(path.read_text())
    del old["horizon_range"]
    old["pred_horizon"] = 20
    path.write_text(json.dumps(old))

    config = JepaConfig.from_pretrained(tmp_path)

    assert config.horizon_range == [20, 20]
    assert "pred_horizon" not in config.to_dict()


def test_horizon_range_and_the_legacy_argument_conflict():
    with pytest.raises(ValueError, match="pred_horizon"):
        JepaConfig(horizon_range=[10, 30], pred_horizon=20)


@pytest.mark.parametrize(
    "bad", [[0, 4], [5, 4], [10], [1, 2, 3], [1.5, 4], [True, 4], 20, "ab"]
)
def test_invalid_horizon_ranges_are_rejected(bad):
    with pytest.raises(ValueError, match="horizon_range"):
        JepaConfig(horizon_range=bad)
    with pytest.raises(ValueError, match="horizon_range"):
        apply_overrides(tiny_config(), {"horizon_range": bad})


def test_overrides_set_the_range_and_reject_the_legacy_key():
    assert apply_overrides(
        tiny_config(), {"horizon_range": [20, 20]}
    ).horizon_range == [20, 20]
    with pytest.raises(KeyError, match="pred_horizon"):
        apply_overrides(tiny_config(), {"pred_horizon": 20})


# ---------------------------------------------------------------------------
# Drawing k
# ---------------------------------------------------------------------------


def test_the_draw_is_uniform_and_a_function_of_its_key():
    before = torch.get_rng_state()
    draws = [draw_horizon([10, 30], 1, step) for step in range(21_000)]
    assert torch.equal(before, torch.get_rng_state())

    counts = Counter(draws)
    assert set(counts) == set(range(10, 31))  # inclusive at both ends
    # 1000 expected per value; 5 standard deviations is about 155.
    assert all(abs(count - 1000) < 155 for count in counts.values())

    assert draws[:50] == [draw_horizon([10, 30], 1, step) for step in range(50)]
    assert draws[:50] != [draw_horizon([10, 30], 2, step) for step in range(50)]
    assert {draw_horizon([20, 20], 1, step) for step in range(50)} == {20}


def test_the_validation_cycle_covers_the_range():
    assert [cycle_horizon([2, 4], i) for i in range(7)] == [2, 3, 4, 2, 3, 4, 2]
    assert {cycle_horizon([20, 20], i) for i in range(7)} == {20}


# ---------------------------------------------------------------------------
# Model: every use of the horizon is the batch's k
# ---------------------------------------------------------------------------


def spy(monkeypatch, owner, name):
    """Record every call to ``owner.name`` and pass it through."""
    calls = []
    original = getattr(owner, name)

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, record)
    return calls


@pytest.mark.parametrize("loss", ["jepa_loss", "decoder_loss"])
def test_the_predictor_never_receives_k(monkeypatch, loss):
    """The same tokens give the same predictor call whatever k is."""
    model = random_horizon_model().eval()
    labels = torch.randint(1, 64, (2, 16))
    calls = spy(monkeypatch, model.predictor, "forward")

    for k in (2, 4, 6):
        getattr(model, loss)(
            labels, attention_mask=torch.ones_like(labels), horizon=torch.tensor(k)
        )

    assert len(calls) == 3
    for args, kwargs in calls:
        assert len(args) == 1 and not kwargs
        torch.testing.assert_close(args[0], calls[0][0][0])


def test_the_target_is_the_detached_latent_k_tokens_ahead(monkeypatch):
    model = random_horizon_model().eval()
    labels = torch.randint(1, 64, (2, 16))
    mask = torch.ones_like(labels)
    mask[1, 11:] = 0
    latents = model.encode(labels, mask).detach()
    calls = spy(monkeypatch, model.commitment_loss, "forward")

    for k in (2, 5):
        for attention_mask in (mask, None):
            calls.clear()
            model.jepa_loss(
                labels, attention_mask=attention_mask, horizon=torch.tensor(k)
            )
            (args, kwargs), = calls
            target, valid = args[1], kwargs["mask"]
            assert not target.requires_grad

            # A position counts iff it and the token k ahead are real.
            lengths = [16, 11] if attention_mask is not None else [16, 16]
            expected = torch.tensor([[t + k < n for t in range(16)] for n in lengths])
            assert torch.equal(valid, expected)
            for row in range(2):
                for t in torch.nonzero(valid[row]).flatten().tolist():
                    torch.testing.assert_close(target[row, t], latents[row, t + k])


@pytest.mark.parametrize("loss", ["jepa_loss", "decoder_loss"])
def test_decoder_blocks_and_hindsight_use_the_batch_k(monkeypatch, loss):
    """Block start, block length and the hindsight target all follow k."""
    model = random_horizon_model().eval()
    labels = torch.randint(1, 64, (2, 16))
    mask = torch.ones_like(labels)
    with torch.no_grad():
        latents = model.encode(labels, mask)
        futures = model.predictor(latents)
    calls = spy(monkeypatch, model.decoder, "token_loss")

    for k in (3, 5):
        calls.clear()
        getattr(model, loss)(labels, attention_mask=mask, horizon=torch.tensor(k))
        (args, kwargs), = calls
        z_t, hint, loss_mask = args[0], args[1], kwargs["loss_mask"]
        assert not hint.requires_grad
        for i in range(15):
            start = i // k * k
            counted = bool(loss_mask.expand(2, -1)[0, i])
            assert counted == (start + k <= 15)
            torch.testing.assert_close(z_t[:, i], latents[:, start])
            if counted:
                # The hindsight commit for z_{start+k}, worked out on its own.
                closest = Futures(*(x[:, start] for x in futures)).commit(
                    Hindsight(), latents[:, start + k]
                )
                torch.testing.assert_close(hint[:, i], closest.future)


@pytest.mark.parametrize("k", [2, 3, 5, 6])
def test_a_block_hint_does_not_depend_on_tokens_after_the_block_start(k):
    model = random_horizon_model().eval()
    start = 2 * k
    labels = torch.randint(1, 64, (1, 28))
    changed = labels.clone()
    changed[:, start + 1 :] = (labels[:, start + 1 :] % 63) + 1

    hints = []
    with torch.no_grad():
        for x in (labels, changed):
            latents = model.encode(x)
            committed = model.predictor(latents).commit(PiArgmax()).future
            hints.append(
                build_block_hints(latents, torch.tensor(k), committed=committed)
            )

    block = slice(start, start + k)
    for part in (0, 1):  # z_t and the committed future
        torch.testing.assert_close(hints[0][part][:, block], hints[1][part][:, block])
        assert not torch.allclose(
            hints[0][part][:, start + k], hints[1][part][:, start + k]
        )


@pytest.mark.parametrize("k", [2, 5])
def test_the_loss_equals_the_sliced_fixed_k_formulation(k):
    """Predicting the tail and masking it changes nothing: the fixed-k recipe."""
    model = random_horizon_model(k, k, decoder_w=0.0)
    labels = torch.randint(1, 64, (2, 16))
    mask = torch.ones_like(labels)
    mask[1, 11:] = 0

    torch.manual_seed(3)
    out = model.jepa_loss(labels, attention_mask=mask, horizon=torch.tensor(k))
    rng = torch.get_rng_state()

    torch.manual_seed(3)
    latents = model.encode(labels, mask)
    visreg = model.visreg(latents, mask=mask)
    commitment = model.commitment_loss(
        model.predictor(latents[:, :-k]),
        latents[:, k:].detach(),
        mask=mask[:, :-k].bool() & mask[:, k:].bool(),
    )
    assert torch.equal(rng, torch.get_rng_state())
    for part in (visreg, commitment):
        for name, value in part.terms.items():
            torch.testing.assert_close(out.terms[name], value, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        out.loss, visreg.loss + commitment.loss, atol=1e-5, rtol=0
    )


def test_hints_accept_an_int_or_a_tensor_horizon():
    latents = torch.randn(2, 13, 4)
    for committed in (None, torch.randn(2, 13, 4)):
        for a, b in zip(
            build_block_hints(latents, 5, committed=committed),
            build_block_hints(latents, torch.tensor(5), committed=committed),
        ):
            assert torch.equal(a, b)


def test_a_sequence_must_exceed_the_longest_horizon():
    model = random_horizon_model(2, 6)
    labels = torch.randint(1, 64, (2, 6))
    for loss in (model.jepa_loss, model.decoder_loss):
        with pytest.raises(ValueError, match="horizon"):
            loss(labels, horizon=torch.tensor(2))


# ---------------------------------------------------------------------------
# Trainer: where k is drawn
# ---------------------------------------------------------------------------


def trainer_for(model, tmp_path, *, stage, resume=False, crash_after=None, seed=7):
    from jepa.training.trainer import JepaTrainer, TrainConfig

    generator = torch.Generator().manual_seed(0)
    data = Dataset.from_dict(
        {
            "input_ids": [
                torch.randint(1, 64, (14,), generator=generator).tolist()
                for _ in range(10)
            ]
        }
    )
    return JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(tmp_path), stage=stage, learning_rate=1e-3,
            batch_size=2, epochs=2, device="cpu", num_workers=0, resume=resume,
            save_every_steps=2, seed=seed,
        ),
        train_dataset=data if crash_after is None else CrashingDataset(data, crash_after),
        eval_dataset=data.select(range(8)),
        tokenizer=SimpleNamespace(pad_token_id=0),
    )


def record_horizons(model, stage):
    """Wrap the stage loss; return ``(train, validation)`` lists of k."""
    train, validation = [], []
    name = f"{stage}_loss"
    original = getattr(model, name)

    def record(*args, horizon, **kwargs):
        (train if torch.is_grad_enabled() else validation).append(int(horizon))
        assert isinstance(horizon, torch.Tensor) and horizon.ndim == 0
        return original(*args, horizon=horizon, **kwargs)

    setattr(model, name, record)
    return train, validation


@pytest.mark.parametrize("stage", ["jepa", "decoder"])
def test_a_resumed_run_draws_the_same_horizons(tmp_path, stage):
    reference = random_horizon_model()
    ref_train, ref_val = record_horizons(reference, stage)
    trainer_for(reference, tmp_path / "ref", stage=stage).train()

    # (seed, global_step) alone decides: 5 steps per epoch, 2 epochs, seed 7.
    assert ref_train == [draw_horizon([2, 6], 7, step) for step in range(10)]
    assert len(set(ref_train)) > 1
    # Four validation batches per epoch, the same k values in both epochs.
    assert ref_val == [2, 3, 4, 5] * 2

    # Crash on the batch after the step-6 save, then resume.
    crashed = random_horizon_model()
    crash_train, _ = record_horizons(crashed, stage)
    with pytest.raises(RuntimeError, match="simulated crash"):
        trainer_for(crashed, tmp_path / "run", stage=stage, crash_after=12).train()
    assert crash_train == ref_train[:6]

    resumed = JepaModel.from_pretrained(str(tmp_path / "run"))
    resumed_train, resumed_val = record_horizons(resumed, stage)
    # The seed in the trainer state decides, not the config's.
    trainer_for(resumed, tmp_path / "run", stage=stage, resume=True, seed=99).train()
    assert resumed_train == ref_train[6:]
    assert resumed_val == ref_val[4:]


def test_batches_shorter_than_the_longest_horizon_are_skipped(tmp_path, monkeypatch):
    """The rule is hi, so whether a batch trains never depends on the draw."""
    from jepa.training import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "PAD_TO_MULTIPLE_OF", 1)
    model = random_horizon_model(2, 14)
    train, validation = record_horizons(model, "jepa")
    trainer_for(model, tmp_path, stage="jepa").train()
    assert train == [] and validation == []


# ---------------------------------------------------------------------------
# Why: a periodic latent stops being free to predict
# ---------------------------------------------------------------------------


def test_a_period_10_latent_is_free_at_k_20_and_costly_at_a_random_k():
    from jepa.evaluation.measure import evaluation_horizons
    from jepa.models.commitment import distance

    angles = 2 * torch.pi * torch.arange(400, dtype=torch.float64) / 10
    z = torch.stack([angles.cos(), angles.sin()], dim=-1)

    def copy_distance(k):
        return distance(z[:-k], z[k:]).mean().item()

    fixed = copy_distance(20)
    random = [copy_distance(k) for k in evaluation_horizons([10, 30])]

    assert fixed == pytest.approx(0.0, abs=1e-5)
    # Averaged over a whole cycle of phases, cos is 0 and the distance is 2.
    assert sum(random) / len(random) > 1.0
