# Full JEPA + decoder stack, ready to be trained auto-regressively

from __future__ import annotations

import torch
import torch.nn.functional as F

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
    ``scores`` is ``(batch, positions, prediction_count)``: win-probability
    logits, so this picks the most likely future.
    """
    best = scores.argmax(dim=-1)
    return preds.gather(
        2, best[..., None, None].expand(-1, -1, 1, preds.shape[-1])
    ).squeeze(2)


def hindsight_waypoints(preds, target):
    """Select the candidate closest to the real future latent ("hindsight").

    ``preds`` is ``(batch, positions, prediction_count, hidden)`` and
    ``target`` the real ``(batch, positions, hidden)`` latents ``k`` steps
    ahead.  Uses the same normalized (cosine-style) distance as the
    multi-hypothesis loss.  The result is still a predictor output, so feeding
    it to the decoder carries information about the actual future without
    leaking the raw block-end latent (which contains the last target token).
    """
    preds_cmp = F.normalize(preds.float(), dim=-1)
    target_cmp = F.normalize(target.float(), dim=-1)
    best = torch.einsum("btpd,btd->btp", preds_cmp, target_cmp).argmax(-1)
    return preds.gather(
        2, best[..., None, None].expand(-1, -1, 1, preds.shape[-1])
    ).squeeze(2)


class JepaDecodeModel(JepaModel):
    """JEPA encoder/predictor plus the autoregressive decoder.

    Configured by a single :class:`jepa.config.JepaConfig` whose ``decoder``
    field is set (use :meth:`JepaConfig.from_encoder` with
    ``with_decoder=True``)::

        config = JepaConfig.from_encoder(enc_config)
        model = JepaDecodeModel(config)

    Existing components can be injected, which lets a decoder be trained
    against a frozen encoder/predictor::

        model = JepaDecodeModel(config, encoder=encoder, predictor=predictor)
        model.freeze_jepa()
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

        # Decoder-only training (``freeze_jepa``): forward then runs the
        # encoder/predictor under ``no_grad`` and computes no JEPA loss.
        self.frozen_jepa = False
        # JEPA-only training (``skip_decoder``): forward returns no decoder
        # output, so the decoder is never run.
        self.decoder_skipped = False

        self.post_init()

    def freeze_jepa(self) -> None:
        """Freeze the encoder/predictor for decoder-only training.

        Their parameters stop requiring gradients, so the optimizer only sees
        the decoder, and they stay in eval mode even when the trainer calls
        ``model.train()``.  :meth:`forward` then skips the JEPA loss: the
        hindsight waypoint comes from :func:`hindsight_waypoints` instead of
        the loss's winner.
        """
        if self.decoder_skipped:
            raise ValueError("freeze_jepa() and skip_decoder() leave nothing to train")
        self.frozen_jepa = True
        for module in (self.encoder, self.predictor):
            module.requires_grad_(False)
            module.eval()

    def skip_decoder(self) -> None:
        """Train only the encoder/predictor (JEPA loss); never run the decoder.

        The decoder stays in the model (and in saved checkpoints) untouched:
        its parameters stop requiring gradients, so the optimizer and gradient
        clipping ignore it, and :meth:`forward` returns ``decoder_out=None`` in
        training mode.  A decoder can be trained afterwards with
        ``python -m jepa train-decoder``.  Generation is unaffected.
        """
        if self.frozen_jepa:
            raise ValueError("freeze_jepa() and skip_decoder() leave nothing to train")
        self.decoder_skipped = True
        self.decoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_jepa:
            # The trainer switches the whole model to train mode each epoch;
            # the frozen backbone must stay in eval mode.
            self.encoder.eval()
            self.predictor.eval()
        return self

    @torch.no_grad()
    def _frozen_jepa_forward(self, labels, attention_mask, right_padded):
        """Encoder/predictor outputs without the VISReg loss (frozen mode).

        Mirrors the training branch of :meth:`JepaModel.forward`; the loss is
        pointless when neither encoder nor predictor can change.
        """
        latents = self._encode(labels, attention_mask, right_padded)
        horizon = self.pred_horizon
        if latents.shape[1] <= horizon:
            raise ValueError(
                f"sequence length ({latents.shape[1]}) must exceed "
                f"pred_horizon ({horizon})"
            )
        preds, scores, mean = self.predictor(
            latents[:, :-horizon, :], return_all_timesteps=True
        )
        # The VISReg loss computes the same candidate-target distance in fp32
        # (it disables autocast internally); matching that keeps the frozen and
        # joint paths' decoder hints identical under bf16 autocast.
        device_type = "cuda" if latents.is_cuda else "cpu"
        with torch.autocast(device_type, enabled=False):
            z_future = hindsight_waypoints(
                preds,
                latents[:, horizon:, :],
            )
        return {
            "loss": None,
            "latents": latents,
            "preds": preds,
            "scores": scores,
            "mean": mean,
            "z_future": z_future,
        }

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
                      VISReg loss (``None`` when the JEPA part is frozen) and
                      ``decoder_out`` contains the token loss (``decoder_out``
                      is ``None`` after :meth:`skip_decoder`).
            Inference: generated token ids of shape (batch, k).
        """
        if self.frozen_jepa and training:
            jepa_out = self._frozen_jepa_forward(
                labels, attention_mask, right_padded
            )
        else:
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

        if self.decoder_skipped:
            return jepa_out, None

        # The decoder is trained on frozen predictor waypoints.  For each token
        # position, the hint is the real latent at its block start and a
        # predictor candidate at the block end.  Using a predictor output (not
        # the raw block-end latent) avoids leaking the final target token.
        #
        # In training, ``DecoderConfig.train_policy`` picks the candidate:
        # "hindsight" is the one closest to the real future
        # (``jepa_out["z_future"]``), "pi_argmax" the score-selected one.  The
        # hindsight choice correlates with the actual continuation, so the
        # decoder learns that the hint is informative instead of ignoring it.
        # Evaluation always uses the score-selected candidate, so validation
        # measures the inference condition.
        latents = jepa_out["latents"]
        if self.training and self.config.decoder.train_policy == "hindsight":
            predicted_future = jepa_out["z_future"]
        else:
            predicted_future = select_waypoints(jepa_out["preds"], jepa_out["scores"])
        predicted_future = predicted_future.detach()
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
