"""Joint training: the decoder loss reaches the encoder through the hint z_t."""

import pytest
import torch

from jepa import JepaModel
from jepa.config import apply_overrides

from test_smoke import K, expected_decoder_loss, tiny_config


def joint_model(decoder_w: float = 0.5) -> JepaModel:
    torch.manual_seed(0)
    return JepaModel(apply_overrides(tiny_config(), {"decoder_w": decoder_w}))


def test_joint_loss_adds_the_weighted_decoder_token_term():
    model = joint_model(0.5)
    labels = torch.randint(1, 64, (2, 12))
    mask = torch.ones_like(labels)

    out = model.jepa_loss(labels, attention_mask=mask, horizon=K)

    # The token term is the decoder loss on hindsight hints (the train policy).
    expected, _ = expected_decoder_loss(model, labels, mask, "hindsight")
    assert out.terms["token"].item() == pytest.approx(expected.item(), rel=1e-5)
    weights = {**model.visreg.weights, **model.commitment_loss.weights, "token": 0.5}
    weighted = sum(weights[name] * value for name, value in out.terms.items())
    assert out.loss.item() == pytest.approx(weighted.item(), rel=1e-5)


def test_token_term_trains_the_encoder_but_not_the_predictor():
    """The decoder gradient enters through z_t; the committed future is detached."""
    model = joint_model(0.5)
    labels = torch.randint(1, 64, (2, 12))

    out = model.jepa_loss(labels, attention_mask=torch.ones_like(labels), horizon=K)
    out.terms["token"].backward()

    assert model.encoder.embedding.weight.grad.abs().sum() > 0
    assert all(
        p.grad is None or not p.grad.any() for p in model.predictor.parameters()
    )
    assert model.decoder.lm_head.weight.grad.abs().sum() > 0


def test_hint_z_t_does_not_see_its_block():
    """Changing the tokens a block predicts leaves that block's hint z_t unchanged."""
    from jepa import build_block_hints

    model = joint_model(0.5).eval()
    labels = torch.randint(1, 64, (1, 12))
    changed = labels.clone()
    block_start = 4
    changed[:, block_start + 1 :] = torch.randint(1, 64, (1, 12 - block_start - 1))

    hints = [build_block_hints(model.encode(x), K)[0] for x in (labels, changed)]

    # Positions block_start..block_start+k-1 predict tokens inside the block.
    block = slice(block_start, block_start + K)
    torch.testing.assert_close(hints[0][:, block], hints[1][:, block])
    assert not torch.allclose(hints[0][:, block_start + K], hints[1][:, block_start + K])


def test_joint_jepa_stage_trains_every_module_and_reports_token_perplexity(tmp_path):
    import math
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    model = joint_model(0.5)
    before = {
        name: [p.detach().clone() for p in getattr(model, name).parameters()]
        for name in ("encoder", "predictor", "decoder")
    }
    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(4)]}
    )
    history = JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(tmp_path), stage="jepa", learning_rate=1e-3,
            batch_size=2, epochs=1, device="cpu", num_workers=0,
            save_every_steps=None,
        ),
        train_dataset=data,
        eval_dataset=data.select(range(2)),
        tokenizer=SimpleNamespace(pad_token_id=0),
    ).train()

    for name, params in before.items():
        assert any(
            not torch.equal(p, q)
            for p, q in zip(getattr(model, name).parameters(), params)
        ), name
    assert {"train_jepa_token", "val_jepa_token", "val_jepa_tokens"} <= set(history)
    # One validation batch: perplexity is exp of its token term, not of the
    # total loss (which also holds the JEPA terms).
    assert history["val_perplexity"][0] == pytest.approx(
        math.exp(history["val_jepa_token"][0]), rel=1e-4
    )
    assert history["val_jepa_token"][0] != pytest.approx(history["val_loss"][0])


def test_zero_decoder_weight_has_no_token_term():
    model = joint_model(0.0)
    labels = torch.randint(1, 64, (2, 12))

    out = model.jepa_loss(labels, attention_mask=torch.ones_like(labels), horizon=K)

    assert "token" not in out.terms and "tokens" not in out.diagnostics
    assert "token" not in model.loss_weights


# ---------------------------------------------------------------------------
# decoder_hint = "real" (ADR 0004): the decoder reads the blurred real future
# ---------------------------------------------------------------------------


def real_hint_model(decoder_w: float = 0.5) -> JepaModel:
    torch.manual_seed(0)
    return JepaModel(apply_overrides(
        tiny_config(), {"decoder_w": decoder_w, "decoder_hint": "real"}
    ))


def test_decoder_hint_defaults_to_committed_and_validates():
    from jepa import JepaConfig

    assert tiny_config().decoder_hint == "committed"
    # Old checkpoints have no field and load as "committed".
    serialized = tiny_config().to_dict()
    serialized.pop("decoder_hint")
    assert JepaConfig(**serialized).decoder_hint == "committed"
    with pytest.raises(ValueError, match="decoder_hint"):
        apply_overrides(tiny_config(), {"decoder_w": 0.3, "decoder_hint": "noisy"})
    # The real hint only exists in the joint JEPA stage.
    with pytest.raises(ValueError, match="decoder_w"):
        apply_overrides(tiny_config(), {"decoder_hint": "real"})
    assert real_hint_model().config.decoder_hint == "real"


def test_real_hint_survives_a_save_and_load(tmp_path):
    real_hint_model().save_pretrained(tmp_path)
    assert JepaModel.from_pretrained(str(tmp_path)).config.decoder_hint == "real"


@pytest.mark.parametrize("d", [0.0, 0.05, 0.3, 0.8, 1.0])
def test_blur_puts_the_hint_at_the_hindsight_distance(d):
    from jepa.models.commitment import distance
    from jepa.models.jepa import blur, hint_noise_scale

    torch.manual_seed(0)
    z = torch.randn(64, 32, 448) * 3.0
    scale = hint_noise_scale(torch.tensor(d))
    assert scale.item() == pytest.approx((1 / (1 - d / 2) ** 2 - 1) ** 0.5)

    blurred = blur(z, scale)

    assert distance(blurred, z).mean().item() == pytest.approx(d, abs=0.01)


def test_hint_noise_scale_clamps_the_distance_to_zero_one():
    from jepa.models.jepa import hint_noise_scale

    assert hint_noise_scale(torch.tensor(-0.2)).item() == 0.0
    assert hint_noise_scale(torch.tensor(1.7)).item() == pytest.approx(3**0.5)


def _token_gradient_by_position(model, labels):
    """Gradient of the token term on each latent position, and the output."""
    captured = {}
    encode = model.encode

    def capture(*args, **kwargs):
        latents = encode(*args, **kwargs)
        latents.retain_grad()
        captured["latents"] = latents
        return latents

    model.encode = capture
    out = model.jepa_loss(labels, attention_mask=torch.ones_like(labels), horizon=K)
    out.terms["token"].backward()
    return captured["latents"].grad.abs().sum(dim=(0, 2)), out


def test_real_hint_trains_the_block_end_not_z_t_nor_the_predictor():
    labels = torch.randint(1, 64, (2, 12))
    # Position 0 is only ever a block start; position K is the first block end.
    real = real_hint_model(0.5)
    by_position, out = _token_gradient_by_position(real, labels)
    assert by_position[0] == 0
    assert by_position[K] > 0
    assert all(
        p.grad is None or not p.grad.any() for p in real.predictor.parameters()
    )
    assert real.decoder.lm_head.weight.grad.abs().sum() > 0
    assert "hint_noise" in out.diagnostics

    # The committed hint trains z_t instead (ADR 0002).
    by_position, _ = _token_gradient_by_position(joint_model(0.5), labels)
    assert by_position[0] > 0


def test_predictor_is_trained_by_the_commitment_loss_alone():
    model = real_hint_model(0.5)
    labels = torch.randint(1, 64, (2, 12))

    out = model.jepa_loss(labels, attention_mask=torch.ones_like(labels), horizon=K)
    params = list(model.predictor.parameters())
    total = torch.autograd.grad(out.loss, params, retain_graph=True, allow_unused=True)
    without_token = torch.autograd.grad(
        out.loss - 0.5 * out.terms["token"], params, allow_unused=True
    )
    for grad, expected in zip(total, without_token):
        if expected is None:
            assert grad is None or not grad.any()
        else:
            torch.testing.assert_close(grad, expected)


def _real_hint_reference(model, labels, mask, future=None):
    """Decoder CE with hint (z_t, future); ``future`` defaults to the raw latent."""
    from jepa import build_block_hints

    with torch.no_grad():
        latents = model.encode(labels, mask)
        z_t, real, valid = build_block_hints(latents, K)
        loss_mask = valid.unsqueeze(0) & mask[:, :-1].bool() & mask[:, 1:].bool()
        total, count = model.decoder.token_loss(
            z_t, real if future is None else future, labels,
            attention_mask=mask, loss_mask=loss_mask,
        )
    return (total / count).item()


def test_real_hint_token_term_is_the_noise_free_real_ce_at_zero_noise(monkeypatch):
    import jepa.models.jepa as jepa_module

    model = real_hint_model(0.5)
    labels = torch.randint(1, 64, (2, 12))
    mask = torch.ones_like(labels)
    monkeypatch.setattr(
        jepa_module, "hint_noise_scale", lambda d: torch.zeros_like(d)
    )

    out = model.jepa_loss(labels, attention_mask=mask, horizon=K)

    assert out.terms["token"].item() == pytest.approx(
        _real_hint_reference(model, labels, mask), rel=1e-5
    )
    assert out.diagnostics["hint_noise"].item() == 0.0


def test_hint_noise_follows_the_batch_hindsight_distance():
    from jepa.models.jepa import hint_noise_scale

    model = real_hint_model(0.5)
    labels = torch.randint(1, 64, (2, 12))
    out = model.jepa_loss(labels, attention_mask=torch.ones_like(labels), horizon=K)
    assert out.diagnostics["hint_noise"].item() == pytest.approx(
        hint_noise_scale(out.diagnostics["hindsight_d"]).item()
    )


def test_real_hint_validation_scores_pi_argmax_and_the_noise_free_real_hint():
    from jepa.models.commitment import PiArgmax

    model = real_hint_model(0.5).eval()
    labels = torch.randint(1, 64, (2, 12))
    mask = torch.ones_like(labels)

    with torch.no_grad():
        out = model.jepa_loss(labels, attention_mask=mask, horizon=K, policy=PiArgmax())

    expected, _ = expected_decoder_loss(model, labels, mask, "pi_argmax")
    assert out.terms["token"].item() == pytest.approx(expected.item(), rel=1e-5)
    assert out.diagnostics["real_token"].item() == pytest.approx(
        _real_hint_reference(model, labels, mask), rel=1e-5
    )
    assert "hint_noise" not in out.diagnostics


def test_real_hint_jepa_stage_logs_noise_and_real_perplexity(tmp_path):
    import math
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    model = real_hint_model(0.5)
    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(4)]}
    )
    history = JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(tmp_path), stage="jepa", learning_rate=1e-3,
            batch_size=2, epochs=1, device="cpu", num_workers=0,
            save_every_steps=None,
        ),
        train_dataset=data,
        eval_dataset=data.select(range(2)),
        tokenizer=SimpleNamespace(pad_token_id=0),
    ).train()

    assert "train_jepa_hint_noise" in history
    assert "val_jepa_hint_noise" not in history
    assert history["val_real_perplexity"][0] == pytest.approx(
        math.exp(history["val_jepa_real_token"][0]), rel=1e-4
    )
    assert history["val_perplexity"][0] == pytest.approx(
        math.exp(history["val_jepa_token"][0]), rel=1e-4
    )


# ---------------------------------------------------------------------------
# commitment.move_target_gradient (ADR 0005): the move term trains the target
# ---------------------------------------------------------------------------


def move_target_model(enabled: bool = True, **overrides) -> JepaModel:
    torch.manual_seed(0)
    return JepaModel(apply_overrides(
        tiny_config(), {"commitment": {"move_target_gradient": enabled}, **overrides}
    ))


def _term_gradient_by_position(model, labels, term):
    """Gradient of one JEPA term on each latent position."""
    captured = {}
    encode = model.encode

    def capture(*args, **kwargs):
        latents = encode(*args, **kwargs)
        latents.retain_grad()
        captured["latents"] = latents
        return latents

    model.encode = capture
    out = model.jepa_loss(labels, attention_mask=torch.ones_like(labels), horizon=K)
    if out.terms[term].requires_grad:
        out.terms[term].backward()
    grad = captured["latents"].grad
    if grad is None:
        return torch.zeros(labels.shape[1])
    return grad.abs().sum(dim=(0, 2))


def test_move_target_gradient_defaults_off_and_validates():
    from jepa import JepaConfig

    assert tiny_config().commitment.move_target_gradient is False
    serialized = tiny_config().to_dict()
    serialized["commitment"].pop("move_target_gradient")
    assert JepaConfig(**serialized).commitment.move_target_gradient is False
    with pytest.raises(ValueError, match="move_target_gradient"):
        apply_overrides(tiny_config(), {"commitment": {"move_target_gradient": 1}})


def test_move_term_reaches_the_target_only_when_enabled():
    labels = torch.randint(1, 64, (2, 12))
    # The last position is never a base with a real token K ahead: it is
    # only ever a target, so any gradient on it came through the target.
    last = labels.shape[1] - 1
    assert _term_gradient_by_position(move_target_model(True), labels, "move")[last] > 0
    assert _term_gradient_by_position(move_target_model(False), labels, "move")[last] == 0


@pytest.mark.parametrize("term", ["pi", "mu", "diversity"])
def test_other_commitment_terms_keep_the_target_detached(term):
    labels = torch.randint(1, 64, (2, 12))
    last = labels.shape[1] - 1
    assert _term_gradient_by_position(move_target_model(True), labels, term)[last] == 0


def test_move_target_gradient_changes_no_value():
    labels = torch.randint(1, 64, (2, 12))
    mask = torch.ones_like(labels)
    outs = [
        move_target_model(enabled).jepa_loss(labels, attention_mask=mask, horizon=K)
        for enabled in (False, True)
    ]
    for name in ("move", "pi", "mu", "diversity"):
        torch.testing.assert_close(outs[0].terms[name], outs[1].terms[name])
    for name in ("hindsight_d", "pi_argmax_d", "mu_distance"):
        assert not outs[1].diagnostics[name].requires_grad


def test_move_target_gradient_with_the_real_hint_keeps_the_decoder_paths():
    """The preset combination: the token term still skips z_t and the predictor."""
    model = move_target_model(True, decoder_w=0.5, decoder_hint="real")
    labels = torch.randint(1, 64, (2, 12))
    by_position, _ = _token_gradient_by_position(model, labels)
    assert by_position[0] == 0 and by_position[K] > 0
    assert all(
        p.grad is None or not p.grad.any() for p in model.predictor.parameters()
    )
