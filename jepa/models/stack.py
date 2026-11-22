# Full JEPA + decoder stack, ready to be trained auto-regressively

from __future__ import annotations

import torch

from ..config import JepaConfig
from .commitment import Hindsight, PiArgmax, policy
from .decoder import DecoderModel
from .jepa import JepaModel


def build_block_hints(latents, horizon: int, committed=None):
    """Return ``(z_t, committed)`` for the next-token decoder positions.

    For input position ``i`` (which predicts token ``i + 1``), the hint is the
    latent at the start of the ``horizon``-token block containing that target
    token and the committed future at the block end.

    When ``committed`` is given, it must have shape
    ``(batch, seq - horizon, hidden)``: one committed future per base
    position.  The decoder then sees the predictor's commit instead of the raw
    block-end latent, which avoids leaking the last target token.
    """
    seq_len = latents.shape[1]
    positions = torch.arange(seq_len - 1, device=latents.device)
    block_start = (positions // horizon) * horizon

    if committed is None:
        block_end = torch.clamp(block_start + horizon, max=seq_len - 1)
        committed = latents[:, block_end, :]
    else:
        index = block_start.clamp(max=seq_len - horizon - 1)
        committed = committed[:, index, :]

    return latents[:, block_start, :], committed


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
        committed future is chosen by the decoder's train policy directly from
        :class:`Futures`.
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
        futures = self.predictor(
            latents[:, :-horizon, :], return_all_timesteps=True
        )
        committed = futures.commit(Hindsight(), latents[:, horizon:, :]).future
        return {
            "loss": None,
            "latents": latents,
            "futures": futures,
            "committed": committed,
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
            training: Training returns per-component losses and the
                      predictor's futures; inference generates the next ``k``
                      tokens.
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
                committed=jepa_out["committed"],
                context_ids=labels,
                k=self.pred_horizon,
                temperature=temperature,
                top_k=top_k,
            )

        if self.decoder_skipped:
            return jepa_out, None

        # The decoder is trained on committed futures.  For each token
        # position, the hint is the real latent at its block start and the
        # committed future at the block end.  Using a predictor commit (not the
        # raw block-end latent) avoids leaking the final target token.
        #
        # In training, ``DecoderConfig.train_policy`` picks the policy:
        # "hindsight" commits to the candidate closest to the real future and
        # "pi_argmax" to the most likely one under pi.  The hindsight choice
        # correlates with the actual continuation, so the decoder learns that
        # the hint is informative instead of ignoring it.  Evaluation always
        # commits by pi-argmax, so validation measures the inference condition.
        latents = jepa_out["latents"]
        futures = jepa_out["futures"]
        if self.training:
            committed = futures.commit(
                policy(self.config.decoder.train_policy),
                latents[:, self.pred_horizon:, :],
            ).future
        else:
            committed = futures.commit(PiArgmax()).future
        z_t, committed = build_block_hints(
            latents,
            self.pred_horizon,
            committed=committed.detach(),
        )

        decoder_out = self.decoder(
            z_t=z_t.detach(),
            committed=committed,
            labels=labels,
            training=True,
            loss_chunk_size=loss_chunk_size,
            attention_mask=attention_mask,
        )

        return jepa_out, decoder_out
