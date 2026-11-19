"""Smoke tests for the composed JEPA stack."""

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
    trunk_layers: int = 0,
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
        trunk_layers=trunk_layers,
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


def test_predictor_context_window_matches_single_step():
    """All-timestep windows equal the single-step path and never look ahead."""
    from jepa import PredictorModel

    torch.manual_seed(0)
    encoder = tiny_config().encoder
    predictor = PredictorModel(
        PredictorConfig.from_encoder(
            encoder, prediction_count=4, num_hidden_layers=1, context_window=3
        )
    ).eval()
    latents = torch.randn(2, 6, 32)

    with torch.no_grad():
        preds, scores = predictor(latents, return_all_timesteps=True)
        assert preds.shape == (2, 6, 4, 32)
        assert scores.shape == (2, 6, 4)
        # t = 0, 1 have a partial (masked) window; t >= 2 a full one.
        for t in range(6):
            p, s = predictor(latents[:, : t + 1], return_all_timesteps=False)
            torch.testing.assert_close(p, preds[:, t], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(s, scores[:, t], atol=1e-5, rtol=1e-4)

        # The context changes the output (the window is actually used).
        shifted = latents.clone()
        shifted[:, 3] += 1.0
        _, s2 = predictor(shifted, return_all_timesteps=True)
        assert not torch.allclose(s2[:, 4], scores[:, 4])
        torch.testing.assert_close(s2[:, :3], scores[:, :3])  # no look-ahead


def test_full_stack_forward_with_context_window():
    config = tiny_config()
    config.predictor.context_window = 3
    model = JepaDecodeModel(config)
    labels = torch.randint(1, 64, (2, 12))
    jepa_out, decoder_out = model(
        labels=labels, attention_mask=torch.ones_like(labels), training=True
    )
    assert jepa_out["preds"].shape == (2, 10, 4, 32)
    assert torch.isfinite(jepa_out["loss"]) and torch.isfinite(decoder_out["loss"])


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


# ---------------------------------------------------------------------------
# Predictor trunk
# ---------------------------------------------------------------------------

# Parameters of the legacy (trunk_layers=0) predictor for the tiny config.
# Frozen on purpose: old checkpoints must keep loading into the same keys.
LEGACY_PREDICTOR_KEYS = [
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
    "score_head.weights",
    "score_head.bias",
]


def test_new_config_fields_default_to_legacy_behaviour():
    encoder = tiny_config().encoder
    predictor = PredictorConfig.from_encoder(encoder)
    assert predictor.trunk_layers == 0

    decoder = DecoderConfig.from_encoder(encoder)
    assert decoder.tie_embeddings is False
    assert decoder.hint_every_layer is False
    assert decoder.tie_word_embeddings is False


def test_predictor_trunk_off_keeps_legacy_parameters():
    from jepa import PredictorModel

    predictor = PredictorModel(
        PredictorConfig.from_encoder(
            tiny_config().encoder, prediction_count=4, num_hidden_layers=1
        )
    )
    assert list(predictor.state_dict()) == LEGACY_PREDICTOR_KEYS
    assert not any("trunk" in key for key in predictor.state_dict())


def test_predictor_trunk_causality():
    from jepa import PredictorModel

    torch.manual_seed(0)
    predictor = PredictorModel(
        PredictorConfig.from_encoder(
            tiny_config().encoder,
            prediction_count=4,
            num_hidden_layers=1,
            trunk_layers=2,
        )
    ).eval()
    latents = torch.randn(2, 6, 32)

    with torch.no_grad():
        preds, scores = predictor(latents, return_all_timesteps=True)
        assert preds.shape == (2, 6, 4, 32)
        assert scores.shape == (2, 6, 4)

        # The trunk summarises the prefix, so a prefix-only single-step run
        # matches the all-timesteps output at that position.
        for t in range(6):
            p, s = predictor(latents[:, : t + 1], return_all_timesteps=False)
            torch.testing.assert_close(p, preds[:, t], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(s, scores[:, t], atol=1e-5, rtol=1e-4)

        # Perturbing a latent only affects that position and later ones.
        shifted = latents.clone()
        shifted[:, 3] += 1.0
        shifted_preds, shifted_scores = predictor(
            shifted, return_all_timesteps=True
        )
        torch.testing.assert_close(shifted_preds[:, :3], preds[:, :3])
        torch.testing.assert_close(shifted_scores[:, :3], scores[:, :3])
        assert not torch.allclose(shifted_preds[:, 3], preds[:, 3])
        assert not torch.allclose(shifted_scores[:, 4], scores[:, 4])


def test_predictor_trunk_with_context_window():
    """The trunk can be combined with the legacy context window."""
    from jepa import PredictorModel

    torch.manual_seed(0)
    predictor = PredictorModel(
        PredictorConfig.from_encoder(
            tiny_config().encoder,
            prediction_count=4,
            num_hidden_layers=1,
            trunk_layers=1,
            context_window=2,
        )
    ).eval()
    latents = torch.randn(2, 5, 32)

    with torch.no_grad():
        preds, _ = predictor(latents, return_all_timesteps=True)
        p, _ = predictor(latents[:, :3], return_all_timesteps=False)
        torch.testing.assert_close(p, preds[:, 2], atol=1e-5, rtol=1e-4)


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
        tiny_config(trunk_layers=2, tie_embeddings=True, hint_every_layer=True)
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
        "predictor.trunk_in.weight",
        "predictor.trunk.0.attn.w_v.weight",
        "predictor.trunk_norm.rms_norm.weight",
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
        tiny_config(trunk_layers=1, tie_embeddings=True, hint_every_layer=True)
    )
    labels = torch.randint(1, 64, (2, 12))
    _, decoder_out = model(
        labels=labels, attention_mask=torch.ones_like(labels), training=True
    )
    decoder_out["loss"].backward()

    assert model.encoder.embedding.weight.grad is None
    assert model.predictor.trunk_in.weight.grad is None
    # The decoder's own (tied) table does receive the decoder gradient.
    assert model.decoder.embedding.weight.grad is not None
    assert model.decoder.lm_head.weight.grad is not None



def test_trainer_saves_mid_epoch_checkpoints(tmp_path):
    """save_every_steps writes a loadable checkpoint before the epoch ends."""
    import json
    from types import SimpleNamespace

    from datasets import Dataset
    from transformers import TrainingArguments

    from jepa.training.trainer import JepaTrainer

    torch.manual_seed(0)
    data = Dataset.from_dict(
        {"input_ids": [torch.randint(1, 64, (12,)).tolist() for _ in range(10)]}
    )
    args = TrainingArguments(
        output_dir=str(tmp_path),
        optim="adamw_torch",
        eval_strategy="epoch",
        learning_rate=1e-3,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        report_to=[],
        use_cpu=True,
    )
    trainer = JepaTrainer(
        model=JepaDecodeModel(tiny_config()),
        args=args,
        train_dataset=data,
        eval_dataset=data.select(range(4)),
        processing_class=SimpleNamespace(pad_token_id=0),
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
