"""Smoke tests for the composed JEPA stack."""

import torch

from jepa import (
    DecoderConfig,
    EncoderConfig,
    JepaConfig,
    JepaDecodeModel,
    JepaModel,
    PredictorConfig,
    VISRegConfig,
)


def tiny_config(*, with_decoder: bool = True) -> JepaConfig:
    encoder = EncoderConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    predictor = PredictorConfig.from_encoder(
        encoder, prediction_count=4, num_hidden_layers=1
    )
    decoder = (
        DecoderConfig.from_encoder(
            encoder, sliding_window=8, num_hidden_layers=1
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


def test_jepa_inference_without_decoder():
    model = JepaModel(tiny_config(with_decoder=False))
    labels = torch.randint(1, 64, (2, 12))
    attention_mask = torch.ones_like(labels)

    out = model(labels=labels, attention_mask=attention_mask, training=False)

    assert out["z_t"].shape == (2, 32)
    assert out["z_future"].shape == (2, 32)
