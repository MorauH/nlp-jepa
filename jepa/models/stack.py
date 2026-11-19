# Full JEPA + decoder stack, ready to be trained auto-regressively

from __future__ import annotations

import torch

from ..config import JepaConfig
from .decoder import DecoderModel
from .jepa import JepaModel


def build_block_hints(latents, horizon: int, predicted_future=None):
    """Return ``(z_t, z_future)`` for the next-token decoder positions.

    For input position ``i`` (which predicts token ``i + 1``), the hint is the
    latent at the start and end of the ``horizon``-token block containing that
    target token.

    When ``predicted_future`` is given, it must have shape
    ``(batch, seq - horizon, hidden)``: one predictor waypoint per base
    position.  The decoder then sees the predictor's waypoint instead of the
    raw block-end latent, which avoids leaking the last target token.
    """
    seq_len = latents.shape[1]
    positions = torch.arange(seq_len - 1, device=latents.device)
    block_start = (positions // horizon) * horizon

    if predicted_future is None:
        block_end = torch.clamp(block_start + horizon, max=seq_len - 1)
        z_future = latents[:, block_end, :]
    else:
        index = block_start.clamp(max=seq_len - horizon - 1)
        z_future = predicted_future[:, index, :]

    return latents[:, block_start, :], z_future


def select_waypoints(preds, scores):
    """Select the highest-scoring candidate at each base position.

    ``preds`` is ``(batch, positions, prediction_count, hidden)`` and
    ``scores`` is ``(batch, positions, prediction_count)``.
    """
    best = scores.argmax(dim=-1)
    return preds.gather(
        2, best[..., None, None].expand(-1, -1, 1, preds.shape[-1])
    ).squeeze(2)


class JepaDecodeModel(JepaModel):
    """JEPA encoder/predictor plus the autoregressive decoder.

    Configured by a single :class:`jepa.config.JepaConfig` whose ``decoder``
    field is set (use :meth:`JepaConfig.from_encoder` with
    ``with_decoder=True``)::

        config = JepaConfig.from_encoder(enc_config, pred_horizon=5)
        model = JepaDecodeModel(config)

    Existing components can be injected, which lets a decoder be trained
    against a frozen encoder/predictor::

        model = JepaDecodeModel(config, encoder=encoder, predictor=predictor)
    """

    config_class = JepaConfig

    def __init__(self, config: JepaConfig, encoder=None, predictor=None, decoder=None):
        super().__init__(config, encoder=encoder, predictor=predictor)

        if config.decoder is None:
            raise ValueError(
                "JepaDecodeModel requires JepaConfig.decoder. Build the config "
                "with JepaConfig.from_encoder(..., with_decoder=True) or pass a "
                "DecoderConfig explicitly."
            )

        self.decoder = decoder if decoder is not None else DecoderModel(config.decoder)

        self.post_init()

    def forward(
        self,
        labels,
        attention_mask=None,
        training: bool = True,
        right_padded: bool = False,
        loss_chunk_size: int | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
    ):
        """
        Args:
            labels: (batch, seq) token ids.
            attention_mask: optional (batch, seq), 1 = real token.
            training: Training returns per-component losses and predictions;
                      inference generates the next ``k`` tokens.
            right_padded: Padding is right-aligned, so the causal encoder can
                      skip the padding mask in attention.
            loss_chunk_size: Optional number of positions per decoder-loss
                      chunk.  The decoder's ``lm_head`` is evaluated in chunks,
                      so the full (B, S, vocab) logits tensor is never held.
            temperature: Token sampling temperature for inference; 0 = greedy.
            top_k: Optional top-k cutoff for inference.

        Returns:
            Training: ``(jepa_out, decoder_out)`` where ``jepa_out`` contains the
                      VISReg loss and ``decoder_out`` contains the token loss.
            Inference: generated token ids of shape (batch, k).
        """
        jepa_out = super().forward(
            labels,
            attention_mask=attention_mask,
            training=training,
            right_padded=right_padded,
        )

        if not training:
            return self.decoder.generate(
                z_t=jepa_out["z_t"],
                z_future=jepa_out["z_future"],
                context_ids=labels,
                k=self.pred_horizon,
                temperature=temperature,
                top_k=top_k,
            )

        # The decoder is trained on frozen predictor waypoints.  For each token
        # position, the hint is the real latent at its block start and the
        # predictor's selected waypoint at the block end.  Using the predictor
        # here matches inference and avoids leaking the final target token.
        latents = jepa_out["latents"]
        predicted_future = select_waypoints(
            jepa_out["preds"], jepa_out["scores"]
        ).detach()
        z_t, z_future = build_block_hints(
            latents,
            self.pred_horizon,
            predicted_future=predicted_future,
        )

        decoder_out = self.decoder(
            z_t=z_t.detach(),
            z_future=z_future,
            labels=labels,
            training=True,
            loss_chunk_size=loss_chunk_size,
            attention_mask=attention_mask,
        )

        return jepa_out, decoder_out
