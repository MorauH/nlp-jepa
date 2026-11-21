"""Smoke tests for the composed JEPA stack."""

import pytest
import torch
import torch.nn.functional as F

from jepa import (
    DecoderConfig,
    EncoderConfig,
    JepaConfig,
    JepaDecodeModel,
    JepaModel,
    PredictorConfig,
    VISRegConfig,
)


def tiny_config(
    *,
    with_decoder: bool = True,
    tie_embeddings: bool = False,
    hint_every_layer: bool = False,
) -> JepaConfig:
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
    decoder = (
        DecoderConfig.from_encoder(
            encoder,
            sliding_window=8,
            num_hidden_layers=1,
            tie_embeddings=tie_embeddings,
            hint_every_layer=hint_every_layer,
        )
        if with_decoder
        else None
    )
    return JepaConfig.from_encoder(
        encoder,
        predictor=predictor,
        decoder=decoder,
        pred_horizon=2,
        visreg=VISRegConfig(num_slices=16),
    )


def test_full_stack_forward():
    model = JepaDecodeModel(tiny_config())
    labels = torch.randint(1, 64, (2, 12))
    attention_mask = torch.ones_like(labels)

    jepa_out, decoder_out = model(
        labels=labels,
        attention_mask=attention_mask,
        training=True,
    )

    assert jepa_out["loss"].ndim == 0
    assert decoder_out["loss"].ndim == 0
    assert jepa_out["preds"].shape == (2, 10, 4, 32)


def test_predictor_reads_only_z_t():
    """Each position's output depends on its own latent and nothing else."""
    from jepa import PredictorModel

    torch.manual_seed(0)
    predictor = PredictorModel(
        PredictorConfig.from_encoder(
            tiny_config().encoder, prediction_count=4, num_hidden_layers=1
        )
    ).eval()
    latents = torch.randn(2, 6, 32)

    with torch.no_grad():
        preds, scores = predictor(latents, return_all_timesteps=True)
        assert preds.shape == (2, 6, 4, 32)
        assert scores.shape == (2, 6, 4)
        # The single-step path on z_t alone matches every position.
        for t in range(6):
            p, s = predictor(latents[:, t : t + 1], return_all_timesteps=False)
            torch.testing.assert_close(p, preds[:, t], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(s, scores[:, t], atol=1e-5, rtol=1e-4)

        # Changing one latent changes only that position.
        shifted = latents.clone()
        shifted[:, 3] += 1.0
        shifted_preds, shifted_scores = predictor(shifted, return_all_timesteps=True)
        keep = [0, 1, 2, 4, 5]
        torch.testing.assert_close(shifted_preds[:, keep], preds[:, keep])
        torch.testing.assert_close(shifted_scores[:, keep], scores[:, keep])
        assert not torch.allclose(shifted_preds[:, 3], preds[:, 3])


def test_removed_predictor_options():
    """Old configs with the options off still load; enabled ones fail loudly."""
    encoder = tiny_config().encoder
    config = PredictorConfig.from_encoder(encoder, trunk_layers=0, context_window=0)
    assert not hasattr(config, "trunk_layers")
    assert "context_window" not in config.to_dict()
    for name in ("trunk_layers", "context_window"):
        with pytest.raises(ValueError, match=name):
            PredictorConfig.from_encoder(encoder, **{name: 2})


def test_jepa_inference_without_decoder():
    model = JepaModel(tiny_config(with_decoder=False))
    labels = torch.randint(1, 64, (2, 12))
    attention_mask = torch.ones_like(labels)

    out = model(labels=labels, attention_mask=attention_mask, training=False)

    assert out["z_t"].shape == (2, 32)
    assert out["z_future"].shape == (2, 32)


def test_visreg_dot_product_distance_matches_reference():
    """The dot-product distance and masked diversity equal the direct formulas."""
    from jepa.models.losses import VISRegLoss

    torch.manual_seed(0)
    criterion = VISRegLoss(VISRegConfig(num_slices=8, diversity_w=0.5))
    cand = torch.randn(2, 5, 4, 16, requires_grad=True)
    tgt = torch.randn(2, 5, 16)
    scores = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5)
    mask[1, 3:] = 0

    _, stats = criterion(
        cand, tgt, mask=mask, scores=scores,
        reg_source=tgt, reg_mask=mask, return_stats=True,
    )

    valid = mask.bool()
    d_ref = (
        F.normalize(cand, dim=-1) - F.normalize(tgt, dim=-1).unsqueeze(2)
    ).pow(2).sum(-1)
    best_ref = (d_ref.min(-1).values * mask).sum() / mask.sum()
    torch.testing.assert_close(stats["best_d"], best_ref.detach())

    dirs = F.normalize(cand[valid], dim=-1)
    sim = torch.bmm(dirs, dirs.transpose(1, 2))
    off_diag = 1.0 - torch.eye(4)
    div_ref = (F.relu(sim) * off_diag).sum() / (dirs.shape[0] * 4 * 3)
    torch.testing.assert_close(stats["loss_diversity"], div_ref.detach())

    # Value head: MSE of the predicted distances (-scores) onto d, and the
    # value-selected candidate is the one with the highest score.
    value_ref = ((-scores - d_ref.detach()).pow(2).mean(-1) * mask).sum() / mask.sum()
    torch.testing.assert_close(stats["loss_value"], value_ref)
    picked = d_ref.detach().gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
    torch.testing.assert_close(
        stats["selected_d"], (picked * mask).sum() / mask.sum()
    )


def test_value_loss_trains_scores_toward_distances():
    """The value gradient moves -scores toward the true distances only."""
    from jepa.models.losses import VISRegLoss

    torch.manual_seed(0)
    criterion = VISRegLoss(
        VISRegConfig(num_slices=4, move_w=0.0, diversity_w=0.0, scale_w=0.0,
                     shape_w=0.0, center_w=0.0, score_w=1.0)
    )
    cand = torch.randn(1, 3, 4, 16, requires_grad=True)
    tgt = torch.randn(1, 3, 16)
    scores = torch.zeros(1, 3, 4, requires_grad=True)
    loss = criterion(cand, tgt, scores=scores, reg_source=tgt)
    loss.backward()
    # Predicted distance 0 is below every true distance, so the gradient
    # raises the predicted distance, i.e. lowers the score.
    assert (scores.grad > 0).all()
    # The regression target is detached: candidates get no value gradient.
    assert cand.grad is None or torch.count_nonzero(cand.grad) == 0


# ---------------------------------------------------------------------------
# Predictor parameters
# ---------------------------------------------------------------------------

# Parameters of the z_t-only predictor for the tiny config.
PREDICTOR_KEYS = [
    "hidden_expansion.weights",
    "hidden_expansion.bias",
    "transformer.decoders.0.attn.w_q_norm.0.weight",
    "transformer.decoders.0.attn.w_q_norm.1.rms_norm.weight",
    "transformer.decoders.0.attn.w_k_norm.0.weight",
    "transformer.decoders.0.attn.w_k_norm.1.rms_norm.weight",
    "transformer.decoders.0.attn.w_v.weight",
    "transformer.decoders.0.attn.w_o.weight",
    "transformer.decoders.0.attn_norm.rms_norm.weight",
    "transformer.decoders.0.mlp.branch_silu.0.weight",
    "transformer.decoders.0.mlp.branch_lin.weight",
    "transformer.decoders.0.mlp.linear.weight",
    "transformer.decoders.0.mlp_norm.rms_norm.weight",
    "delta_head.weights",
    "delta_head.bias",
    "value_head.weights",
    "value_head.bias",
]


def test_new_config_fields_default_to_legacy_behaviour():
    encoder = tiny_config().encoder
    decoder = DecoderConfig.from_encoder(encoder)
    assert decoder.tie_embeddings is False
    assert decoder.hint_every_layer is False
    assert decoder.tie_word_embeddings is False


def test_predictor_parameters():
    from jepa import PredictorModel

    predictor = PredictorModel(
        PredictorConfig.from_encoder(
            tiny_config().encoder, prediction_count=4, num_hidden_layers=1
        )
    )
    assert list(predictor.state_dict()) == PREDICTOR_KEYS


# ---------------------------------------------------------------------------
# Decoder: tied embeddings and per-layer hints
# ---------------------------------------------------------------------------


def test_decoder_tied_embeddings_round_trip(tmp_path):
    from safetensors.torch import load_file

    from jepa import DecoderModel

    model = JepaDecodeModel(tiny_config(tie_embeddings=True))
    decoder = model.decoder
    assert decoder.lm_head.weight is decoder.embedding.weight
    # The decoder table stays separate from the encoder's.
    assert decoder.embedding.weight is not model.encoder.embedding.weight

    labels = torch.randint(1, 64, (2, 12))
    with torch.no_grad():
        before = model.decoder._run(
            labels[:, :-1],
            model.decoder._hint(
                torch.zeros(2, 11, 32), torch.zeros(2, 11, 32)
            ),
        ).clone()

    full_dir = tmp_path / "full"
    model.save_pretrained(full_dir)
    # Safetensors cannot store aliases: only the embedding copy is written.
    saved = load_file(full_dir / "model.safetensors")
    assert "decoder.embedding.weight" in saved
    assert "decoder.lm_head.weight" not in saved

    reloaded, info = JepaDecodeModel.from_pretrained(
        full_dir, output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert reloaded.decoder.lm_head.weight is reloaded.decoder.embedding.weight
    assert torch.equal(
        reloaded.decoder.embedding.weight, decoder.embedding.weight
    )
    with torch.no_grad():
        after = reloaded.decoder._run(
            labels[:, :-1],
            reloaded.decoder._hint(
                torch.zeros(2, 11, 32), torch.zeros(2, 11, 32)
            ),
        )
    torch.testing.assert_close(after, before)

    standalone_dir = tmp_path / "decoder"
    decoder.save_pretrained(standalone_dir)
    standalone, info = DecoderModel.from_pretrained(
        standalone_dir, output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert standalone.lm_head.weight is standalone.embedding.weight
    assert torch.equal(standalone.embedding.weight, decoder.embedding.weight)


def test_untied_decoder_keeps_two_tables():
    model = JepaDecodeModel(tiny_config(tie_embeddings=False))
    decoder = model.decoder
    assert decoder.lm_head.weight is not decoder.embedding.weight
    assert decoder.lm_head.weight.shape == decoder.embedding.weight.shape


def test_hint_every_layer_starts_as_input_only_injection():
    """Zero-initialized per-layer projections leave the forward pass unchanged."""
    from jepa import DecoderModel

    torch.manual_seed(0)
    encoder = tiny_config().encoder
    base = DecoderModel(
        DecoderConfig.from_encoder(encoder, sliding_window=8, num_hidden_layers=2)
    ).eval()
    layered = DecoderModel(
        DecoderConfig.from_encoder(
            encoder,
            sliding_window=8,
            num_hidden_layers=2,
            hint_every_layer=True,
        )
    ).eval()
    missing, unexpected = layered.load_state_dict(base.state_dict(), strict=False)
    assert unexpected == []
    assert all(key.startswith("hint_proj.") for key in missing)
    assert all(
        torch.count_nonzero(projection.weight) == 0
        for projection in layered.hint_proj
    )

    input_ids = torch.randint(1, 64, (2, 6))
    hint = torch.randn(2, 6, 32)
    with torch.no_grad():
        torch.testing.assert_close(
            layered._run(input_ids, hint), base._run(input_ids, hint)
        )


def test_hint_every_layer_matches_generate_path():
    from jepa import DecoderModel

    torch.manual_seed(0)
    decoder = DecoderModel(
        DecoderConfig.from_encoder(
            tiny_config().encoder,
            sliding_window=8,
            num_hidden_layers=2,
            hint_every_layer=True,
        )
    ).eval()
    z_t = torch.randn(1, 32)
    z_future = torch.randn(1, 32)
    context_ids = torch.randint(1, 64, (1, 6))
    hint = decoder._hint(z_t[:, None], z_future[:, None])

    with torch.no_grad():
        # Teacher-forced path: exactly what forward() builds (the whole prefix
        # fits in the sliding window here, so generate() sees the same tokens).
        teacher_logits = decoder.lm_head(
            decoder._run(context_ids, hint.expand(-1, context_ids.shape[1], -1))[
                :, -1, :
            ]
        )
        window = context_ids[:, -decoder.sliding_window :]
        generated_logits = decoder.lm_head(
            decoder._run(window, hint.expand(-1, window.shape[1], -1))[:, -1, :]
        )
        torch.testing.assert_close(generated_logits, teacher_logits)

        greedy = decoder.generate(z_t, z_future, context_ids, k=1, temperature=0.0)
        assert int(greedy[0, 0]) == int(teacher_logits.argmax(dim=-1)[0])


# ---------------------------------------------------------------------------
# Full stack with every new option
# ---------------------------------------------------------------------------


def test_full_stack_forward_with_all_new_options():
    model = JepaDecodeModel(
        tiny_config(tie_embeddings=True, hint_every_layer=True)
    )
    labels = torch.randint(1, 64, (2, 12))
    jepa_out, decoder_out = model(
        labels=labels,
        attention_mask=torch.ones_like(labels),
        training=True,
        loss_chunk_size=4,
    )
    assert jepa_out["preds"].shape == (2, 10, 4, 32)
    assert torch.isfinite(jepa_out["loss"]) and torch.isfinite(decoder_out["loss"])

    (2 * jepa_out["loss"] + decoder_out["loss"]).backward()
    grads = dict(model.named_parameters())
    for name in (
        "predictor.hidden_expansion.weights",
        "predictor.transformer.decoders.0.attn.w_v.weight",
        "predictor.delta_head.weights",
        "predictor.value_head.weights",
        "decoder.hint_proj.0.weight",
        "decoder.embedding.weight",
        "encoder.embedding.weight",
    ):
        grad = grads[name].grad
        assert grad is not None, name
        assert torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


def test_decoder_loss_does_not_update_encoder_or_predictor():
    model = JepaDecodeModel(
        tiny_config(tie_embeddings=True, hint_every_layer=True)
    )
    labels = torch.randint(1, 64, (2, 12))
    _, decoder_out = model(
        labels=labels, attention_mask=torch.ones_like(labels), training=True
    )
    decoder_out["loss"].backward()

    assert model.encoder.embedding.weight.grad is None
    assert all(p.grad is None for p in model.predictor.parameters())
    # The decoder's own (tied) table does receive the decoder gradient.
    assert model.decoder.embedding.weight.grad is not None
    assert model.decoder.lm_head.weight.grad is not None



def test_trainer_saves_mid_epoch_checkpoints(tmp_path):
    """save_every_steps writes a loadable checkpoint before the epoch ends."""
    import json
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    torch.manual_seed(0)
    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(10)]}
    )
    trainer = JepaTrainer(
        model=JepaDecodeModel(tiny_config()),
        config=TrainConfig(
            output_dir=str(tmp_path),
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            device="cpu",
        ),
        train_dataset=data,
        eval_dataset=data.select(range(4)),
        tokenizer=SimpleNamespace(pad_token_id=0),
        use_amp=False,
        save_every_steps=2,
    )

    saves = []
    original_save = trainer._save

    def recording_save(output_dir):
        saves.append(dict(trainer.history.get("last_checkpoint", {})))
        original_save(output_dir)

    trainer._save = recording_save
    trainer.train()

    # 10 examples / batch 2 = 5 steps: mid-epoch saves at 2 and 4, then epoch end.
    assert [s["global_step"] for s in saves] == [2, 4, 5]
    assert not saves[0].get("end_of_epoch") and saves[-1]["end_of_epoch"]
    history = json.loads((tmp_path / "history.json").read_text())
    assert history["last_checkpoint"]["global_step"] == 5
    JepaDecodeModel.from_pretrained(str(tmp_path))


def test_decoder_hindsight_waypoints_in_train_selected_in_eval():
    """Train mode feeds the hindsight-best candidate; eval the score-selected."""
    from jepa.models.stack import (
        build_block_hints,
        hindsight_waypoints,
        select_waypoints,
    )

    torch.manual_seed(0)
    model = JepaDecodeModel(tiny_config())
    model.decoder_hindsight_prob = 1.0
    labels = torch.randint(1, 64, (2, 12))
    mask = torch.ones_like(labels)
    k = model.pred_horizon

    def decoder_loss_with(future, jepa_out):
        z_t, z_future = build_block_hints(
            jepa_out["latents"], k, predicted_future=future.detach()
        )
        return model.decoder(
            z_t=z_t.detach(), z_future=z_future, labels=labels,
            training=True, attention_mask=mask,
        )["loss"]

    with torch.no_grad():
        model.train()
        jepa_out, decoder_out = model(labels=labels, attention_mask=mask, training=True)
        hindsight = hindsight_waypoints(jepa_out["preds"], jepa_out["latents"][:, k:])
        # Same choice as the multi-hypothesis loss's winner.
        torch.testing.assert_close(hindsight, jepa_out["z_future"])
        torch.testing.assert_close(
            decoder_out["loss"], decoder_loss_with(hindsight, jepa_out)
        )

        model.eval()
        jepa_out, decoder_out = model(labels=labels, attention_mask=mask, training=True)
        selected = select_waypoints(jepa_out["preds"], jepa_out["scores"])
        torch.testing.assert_close(
            decoder_out["loss"], decoder_loss_with(selected, jepa_out)
        )


def test_mix_waypoints_probabilities():
    from jepa.models.stack import mix_waypoints

    selected = torch.zeros(4, 50, 3)
    hindsight = torch.ones(4, 50, 3)
    assert torch.equal(mix_waypoints(selected, hindsight, 1.0), hindsight)
    assert torch.equal(mix_waypoints(selected, hindsight, 0.0), selected)
    torch.manual_seed(0)
    mixed = mix_waypoints(selected, hindsight, 0.3)
    # Whole positions are switched, never individual dimensions.
    assert torch.all(mixed.amin(-1) == mixed.amax(-1))
    assert 0.2 < mixed[..., 0].mean().item() < 0.4


def test_normalize_candidates_fixes_scale():
    from jepa import PredictorModel

    torch.manual_seed(0)
    encoder = tiny_config().encoder
    legacy = PredictorModel(PredictorConfig.from_encoder(encoder, prediction_count=4))
    normed = PredictorModel(
        PredictorConfig.from_encoder(encoder, prediction_count=4, normalize_candidates=True)
    )
    # No new parameters: old checkpoints keep loading.
    assert legacy.state_dict().keys() == normed.state_dict().keys()
    preds, _ = normed(torch.randn(2, 5, 32) * 7.0)
    rms = preds.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-3, rtol=1e-3)


def test_annealed_temperature_schedule_and_loading(tmp_path):
    from jepa.models.losses import VISRegLoss

    loss = VISRegLoss(
        VISRegConfig(mh_temperature=0.1, mh_temperature_start=1.0, mh_anneal_fraction=0.5)
    )
    # Starts (and evaluates) at the final value until the trainer drives it.
    assert loss.mh_temperature_now.item() == pytest.approx(0.1)
    for progress, expected in ((0.0, 1.0), (0.25, 0.1 ** 0.5), (0.5, 0.1), (0.9, 0.1)):
        loss.set_anneal_progress(progress)
        assert loss.mh_temperature_now.item() == pytest.approx(expected, rel=1e-5)
    assert "mh_temperature_now" not in loss.state_dict()

    constant = VISRegLoss(VISRegConfig(mh_temperature=0.1))
    constant.set_anneal_progress(0.0)
    assert constant.mh_temperature_now.item() == pytest.approx(0.1)

    # The non-persistent buffer must survive from_pretrained (meta-device init).
    config = tiny_config()
    config.visreg.mh_temperature_start = 1.0
    model = JepaDecodeModel(config)
    model.save_pretrained(tmp_path)
    loaded = JepaDecodeModel.from_pretrained(str(tmp_path))
    assert loaded.criterion.mh_temperature_now.item() == pytest.approx(0.1)
    assert loaded.criterion.mh_temperature_start == 1.0


def test_diversity_margin_ignores_dissimilar_pairs():
    from jepa.models.losses import VISRegLoss

    torch.manual_seed(0)
    # Two candidates at cosine 0.3, two at ~0.99.
    a = F.normalize(torch.randn(16), dim=0)
    b = F.normalize(torch.randn(16), dim=0)
    b = F.normalize(b - (b @ a) * a, dim=0)
    far = torch.stack([a, 0.3 * a + (1 - 0.09) ** 0.5 * b])
    near = torch.stack([a, F.normalize(a + 0.05 * b, dim=0)])

    def div(cands, margin):
        crit = VISRegLoss(VISRegConfig(num_slices=4, diversity_w=1.0, diversity_margin=margin))
        _, stats = crit(
            cands[None, None], torch.randn(1, 1, 16), scores=torch.zeros(1, 1, 2),
            reg_source=torch.randn(1, 4, 16), return_stats=True,
        )
        return stats["loss_diversity"].item()

    assert div(far, 0.0) == pytest.approx(0.3, abs=1e-4)
    assert div(far, 0.5) == 0.0
    assert div(near, 0.5) > 0.4


def test_trainer_drives_temperature_annealing(tmp_path):
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    torch.manual_seed(0)
    config = tiny_config()
    config.visreg.mh_temperature_start = 1.0
    config.visreg.mh_anneal_fraction = 0.5
    model = JepaDecodeModel(config)
    seen = []
    original = model.criterion.set_anneal_progress

    def record(progress):
        original(progress)
        seen.append(model.criterion.mh_temperature_now.item())

    model.criterion.set_anneal_progress = record
    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(8)]}
    )
    JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(tmp_path),
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            device="cpu",
        ),
        train_dataset=data,
        eval_dataset=data.select(range(2)),
        tokenizer=SimpleNamespace(pad_token_id=0),
        use_amp=False,
    ).train()

    assert len(seen) == 4
    assert seen[0] == pytest.approx(1.0)
    assert seen == sorted(seen, reverse=True)
    assert seen[-1] == pytest.approx(0.1)  # 3/4 progress > anneal fraction


# ---------------------------------------------------------------------------
# Decoder-only (frozen JEPA) training
# ---------------------------------------------------------------------------


def test_frozen_jepa_trains_only_the_decoder(tmp_path):
    """freeze_jepa: no grads or updates for the backbone, eval mode, no JEPA loss."""
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    torch.manual_seed(0)
    model = JepaDecodeModel(tiny_config(hint_every_layer=True))
    model.freeze_jepa()
    model.decoder_hindsight_prob = 1.0

    backbone = list(model.encoder.parameters()) + list(model.predictor.parameters())
    before_backbone = [p.detach().clone() for p in backbone]
    before_decoder = [p.detach().clone() for p in model.decoder.parameters()]

    training_modes = []
    model.encoder.register_forward_hook(
        lambda module, *args: training_modes.append(module.training)
    )
    model.predictor.register_forward_hook(
        lambda module, *args: training_modes.append(module.training)
    )

    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(4)]}
    )
    history = JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(tmp_path),
            learning_rate=1e-3,
            batch_size=2,
            epochs=1,
            device="cpu",
        ),
        train_dataset=data,
        eval_dataset=data.select(range(2)),
        tokenizer=SimpleNamespace(pad_token_id=0),
        use_amp=False,
        save_every_steps=1,
    ).train()

    # The trainer calls model.train(); the frozen backbone stays in eval mode.
    assert model.training and model.decoder.training
    assert not model.encoder.training and not model.predictor.training
    assert training_modes and not any(training_modes)

    assert all(p.grad is None for p in backbone)
    assert all(torch.equal(p, q) for p, q in zip(backbone, before_backbone))
    assert any(
        not torch.equal(p, q) for p, q in zip(model.decoder.parameters(), before_decoder)
    )

    # Frozen runs log no JEPA losses (there is none to compute).
    assert set(history) >= {"train_loss", "train_decoder_loss", "val_decoder_loss"}
    assert not any("jepa" in key for key in history)
    assert all("jepa" not in key for key in history["last_checkpoint"])


@pytest.mark.parametrize("amp_dtype", [None, torch.bfloat16])
def test_frozen_decoder_loss_matches_joint_mode(amp_dtype):
    """Both entry points feed the decoder identically for the same weights."""
    import copy
    from contextlib import nullcontext

    torch.manual_seed(0)
    joint = JepaDecodeModel(tiny_config(hint_every_layer=True))
    joint.decoder_hindsight_prob = 1.0
    frozen = copy.deepcopy(joint)
    frozen.freeze_jepa()
    frozen.decoder_hindsight_prob = 1.0

    labels = torch.randint(1, 64, (2, 12))
    mask = torch.ones_like(labels)
    autocast = (
        torch.autocast("cpu", dtype=amp_dtype) if amp_dtype else nullcontext()
    )

    with torch.no_grad(), autocast:
        # Training: hindsight waypoint (prob 1.0).  The frozen path computes it
        # from hindsight_waypoints(); joint mode uses the loss's winner.
        joint.train()
        frozen.train()
        joint_jepa, joint_decoder = joint(
            labels=labels, attention_mask=mask, training=True, right_padded=True
        )
        frozen_jepa, frozen_decoder = frozen(
            labels=labels, attention_mask=mask, training=True, right_padded=True
        )
        assert frozen_jepa["loss"] is None
        torch.testing.assert_close(frozen_jepa["z_future"], joint_jepa["z_future"])
        torch.testing.assert_close(frozen_decoder["loss"], joint_decoder["loss"])

        # Evaluation: score-selected waypoint (the inference condition).
        joint.eval()
        frozen.eval()
        _, joint_decoder = joint(labels=labels, attention_mask=mask, training=True)
        _, frozen_decoder = frozen(labels=labels, attention_mask=mask, training=True)
        torch.testing.assert_close(frozen_decoder["loss"], joint_decoder["loss"])


def test_train_decoder_entry_point_writes_loadable_checkpoint(tmp_path, monkeypatch):
    """`train-decoder` main(): frozen backbone, fresh decoder, mid-epoch saves."""
    import json
    from types import SimpleNamespace

    from datasets import Dataset, DatasetDict
    from jepa.training import train_decoder
    from jepa.training.trainer import JepaTrainer

    class FakeTokenizer(SimpleNamespace):
        model_max_length = 32

        def __len__(self):
            return 64

    torch.manual_seed(0)
    base = tmp_path / "base"
    base_model = JepaDecodeModel(tiny_config(hint_every_layer=True))
    base_model.save_pretrained(base)
    for name in ("encoder", "predictor", "decoder"):
        getattr(base_model, name).save_pretrained(base / name)

    rows = [torch.randint(1, 64, (12,)).tolist() for _ in range(4)]
    data_dir = tmp_path / "data"
    DatasetDict(
        {
            "train": Dataset.from_dict({"input_ids": rows}),
            "validation": Dataset.from_dict({"input_ids": rows[:2]}),
        }
    ).save_to_disk(str(data_dir))

    monkeypatch.setattr(
        train_decoder, "load_training_tokenizer", lambda path: FakeTokenizer(pad_token_id=0)
    )
    monkeypatch.setattr(train_decoder, "COMPILE_MODEL", False)
    monkeypatch.setattr(train_decoder, "SAVE_EVERY_STEPS", 1)

    saves = []
    original_save = JepaTrainer._save

    def recording_save(self, output_dir):
        saves.append(dict(self.history.get("last_checkpoint", {})))
        original_save(self, output_dir)

    monkeypatch.setattr(JepaTrainer, "_save", recording_save)

    # Same directory in and out (the default runs/jepa setup): the joint run's
    # history.json must survive, the decoder logs to its own file.
    joint_history = {"train_jepa_loss": [1.0], "last_checkpoint": {"global_step": 7}}
    (base / "history.json").write_text(json.dumps(joint_history))
    out = base
    train_decoder.main(
        [
            "--model-dir", str(base),
            "--output-dir", str(out),
            "--fineweb-dir", str(data_dir),
            "--tokenizer", str(tmp_path / "unused-tokenizer"),
            "--batch-size", "2",
            "--epochs", "1",
            "--device", "cpu",
        ]
    )

    # 4 examples / batch 2 = 2 steps: mid-epoch save after each, then epoch end.
    assert [s["global_step"] for s in saves] == [1, 2, 2]
    assert not saves[0].get("end_of_epoch") and saves[-1]["end_of_epoch"]

    assert json.loads((out / "history.json").read_text()) == joint_history
    history = json.loads((out / train_decoder.HISTORY_FILE).read_text())
    assert "train_decoder_loss" in history and "val_decoder_loss" in history
    assert not any("jepa" in key for key in history)

    reloaded = JepaDecodeModel.from_pretrained(str(out))
    assert not any("_orig_mod." in key for key in reloaded.state_dict())
    for name in ("encoder", "predictor", "decoder"):
        assert (out / name / "config.json").exists()

    # The encoder was loaded from the checkpoint and never updated; the decoder
    # is fresh (the base model's decoder is not reused).
    assert torch.equal(
        reloaded.encoder.embedding.weight, base_model.encoder.embedding.weight
    )
    assert not torch.equal(
        reloaded.decoder.embedding.weight, base_model.decoder.embedding.weight
    )

    # --resume keeps training *that* decoder (not a fresh one), stays frozen
    # elsewhere, and appends to the decoder history.
    loaded = {}
    original_load = train_decoder.load_model_for_resume

    def spy_load(model_dir):
        model = original_load(model_dir)
        loaded["decoder"] = model.decoder.embedding.weight.detach().clone()
        loaded["frozen"] = model.frozen_jepa
        return model

    monkeypatch.setattr(train_decoder, "load_model_for_resume", spy_load)
    train_decoder.main(
        [
            "--resume",
            "--model-dir", str(out),
            "--output-dir", str(out),
            "--fineweb-dir", str(data_dir),
            "--tokenizer", str(tmp_path / "unused-tokenizer"),
            "--batch-size", "2",
            "--epochs", "1",
            "--device", "cpu",
        ]
    )
    assert loaded["frozen"]
    assert torch.equal(loaded["decoder"], reloaded.decoder.embedding.weight)
    resumed = JepaDecodeModel.from_pretrained(str(out))
    assert not torch.equal(
        resumed.decoder.embedding.weight, reloaded.decoder.embedding.weight
    )
    assert torch.equal(
        resumed.encoder.embedding.weight, base_model.encoder.embedding.weight
    )
    history = json.loads((out / train_decoder.HISTORY_FILE).read_text())
    assert len(history["train_decoder_loss"]) == 2
    assert json.loads((out / "history.json").read_text()) == joint_history


def test_jepa_only_never_runs_the_decoder(tmp_path):
    """skip_decoder: decoder untouched and never called; JEPA trains normally."""
    from types import SimpleNamespace

    from datasets import Dataset

    from jepa.training.trainer import JepaTrainer, TrainConfig

    torch.manual_seed(0)
    model = JepaDecodeModel(tiny_config(hint_every_layer=True))
    model.skip_decoder()

    before_encoder = [p.detach().clone() for p in model.encoder.parameters()]
    before_decoder = {n: p.detach().clone() for n, p in model.decoder.named_parameters()}
    decoder_calls = []
    model.decoder.register_forward_hook(lambda *args: decoder_calls.append(1))

    labels = torch.randint(1, 64, (2, 12))
    jepa_out, decoder_out = model(
        labels=labels, attention_mask=torch.ones_like(labels), training=True
    )
    assert decoder_out is None and jepa_out["loss"].ndim == 0

    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(4)]}
    )
    history = JepaTrainer(
        model=model,
        config=TrainConfig(
            output_dir=str(tmp_path), learning_rate=1e-3, batch_size=2,
            epochs=1, device="cpu",
        ),
        train_dataset=data,
        eval_dataset=data.select(range(2)),
        tokenizer=SimpleNamespace(pad_token_id=0),
        use_amp=False,
        save_every_steps=1,
    ).train()

    assert not decoder_calls
    assert all(p.grad is None for p in model.decoder.parameters())
    assert all(torch.equal(p, before_decoder[n]) for n, p in model.decoder.named_parameters())
    assert any(
        not torch.equal(p, q) for p, q in zip(model.encoder.parameters(), before_encoder)
    )
    # Only JEPA losses are logged; the checkpoint still has a (loadable) decoder.
    assert set(history) >= {"train_loss", "train_jepa_loss", "val_jepa_loss"}
    assert not any("decoder" in key or "perplexity" in key for key in history)
    reloaded = JepaDecodeModel.from_pretrained(str(tmp_path))
    assert all(
        torch.equal(p, before_decoder[n]) for n, p in reloaded.decoder.named_parameters()
    )


def test_freeze_jepa_and_skip_decoder_are_exclusive():
    model = JepaDecodeModel(tiny_config())
    model.skip_decoder()
    with pytest.raises(ValueError):
        model.freeze_jepa()
    model = JepaDecodeModel(tiny_config())
    model.freeze_jepa()
    with pytest.raises(ValueError):
        model.skip_decoder()
