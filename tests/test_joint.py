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
