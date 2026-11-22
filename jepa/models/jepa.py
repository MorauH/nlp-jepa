"""JEPA encoder + predictor + decoder and the two training objectives."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel

from ..config import JepaConfig
from .commitment import PiArgmax, policy
from .decoder import DecoderModel
from .encoder import EncodeTransformer
from .losses import CommitmentLoss, LossOutput, VISReg
from .predictor import PredictorModel


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


class JepaModel(PreTrainedModel):
    """Full JEPA stack: encoder, predictor and autoregressive decoder.

    Training is split into two stages (see
    ``docs/adr/0001-two-stage-training.md``):

    * :meth:`jepa_loss` trains the encoder and predictor with VISReg plus the
      commitment loss.
    * :meth:`decoder_loss` trains the decoder against the frozen encoder and
      predictor.

    The model does not switch modes itself; the trainer decides which stage's
    modules require gradients and which stay in eval mode.

    Components can be injected, which lets a decoder be trained against an
    existing JEPA checkpoint::

        model = JepaModel(config, encoder=encoder, predictor=predictor)
    """

    config_class = JepaConfig

    def __init__(
        self,
        config: JepaConfig,
        encoder: EncodeTransformer | None = None,
        predictor: PredictorModel | None = None,
        decoder: DecoderModel | None = None,
    ):
        super().__init__(config)

        if config.decoder is None:
            raise ValueError(
                "JepaModel requires JepaConfig.decoder; every checkpoint has a "
                "decoder."
            )

        self.encoder = (
            encoder if encoder is not None else EncodeTransformer(config.encoder)
        )
        self.predictor = (
            predictor if predictor is not None else PredictorModel(config.predictor)
        )
        self.decoder = (
            decoder if decoder is not None else DecoderModel(config.decoder)
        )

        self.visreg = VISReg(config.visreg)
        self.commitment_loss = CommitmentLoss(config.commitment)

        self.post_init()

    @property
    def pred_horizon(self) -> int:
        """Number of tokens between a latent and the future it predicts."""
        return self.config.pred_horizon

    def _encode(self, labels, attention_mask):
        """Encode tokens, dropping the padding mask when that is safe.

        Batches are right-padded and the encoder is causal, so no real query
        can attend a pad key: dropping the mask enables the fused causal SDPA
        kernel.  The mask still validates loss positions.
        """
        skip_mask = self.encoder.config.is_causal
        return self.encoder(
            labels, attention_mask=None if skip_mask else attention_mask
        )

    def jepa_loss(
        self,
        labels,
        attention_mask=None,
        progress=None,
    ) -> LossOutput:
        """Run the JEPA stage objective.

        Args:
            labels: ``(batch, seq)`` token ids.
            attention_mask: optional ``(batch, seq)``, 1 = real token.
            progress: 0-dim tensor with the fraction of training completed, or
                ``None`` for the final commitment temperature (evaluation).

        Returns:
            :class:`~jepa.models.losses.LossOutput` whose total is
            VISReg + commitment loss and whose terms/diagnostics merge both.
        """
        enc_hidden = self._encode(labels, attention_mask)

        k = self.pred_horizon
        seq_len = enc_hidden.shape[1]
        if seq_len <= k:
            raise ValueError(
                f"sequence length ({seq_len}) must exceed pred_horizon ({k})"
            )

        X = enc_hidden[:, :-k, :]
        Y = enc_hidden[:, k:, :].detach()

        if attention_mask is not None:
            valid = attention_mask[:, :-k].bool() & attention_mask[:, k:].bool()
        else:
            valid = None

        # From each hidden[i, t, :] predict hidden[i, t+k, :]
        futures = self.predictor(X)

        # VISReg acts on the online encoder latents, not on the predictor
        # output.  Applying it to the committed future would let the predictor
        # cancel the encoder's mean in its own output without fixing the
        # encoder.
        visreg_out = self.visreg(enc_hidden, mask=attention_mask)
        commitment_out = self.commitment_loss(
            futures, Y, mask=valid, progress=progress
        )

        return LossOutput(
            loss=visreg_out.loss + commitment_out.loss,
            terms={**visreg_out.terms, **commitment_out.terms},
            diagnostics={
                **visreg_out.diagnostics,
                **commitment_out.diagnostics,
            },
        )

    def decoder_loss(
        self,
        labels,
        attention_mask=None,
    ) -> LossOutput:
        """Run the decoder stage objective against the frozen JEPA.

        The encoder and predictor run under ``torch.no_grad()``.  The decoder
        is trained on committed futures: in training the commit uses
        ``config.decoder.train_policy`` (target = real latent ``k`` ahead), in
        evaluation (and therefore validation) the inference-time pi-argmax
        commit.  Detached hints mean the decoder loss cannot update the JEPA.

        Returns:
            :class:`~jepa.models.losses.LossOutput` with the unweighted
            ``token`` term and a ``perplexity`` diagnostic.
        """
        k = self.pred_horizon
        seq_len = labels.shape[1]
        if seq_len <= k:
            raise ValueError(
                f"sequence length ({seq_len}) must exceed pred_horizon ({k})"
            )

        with torch.no_grad():
            latents = self._encode(labels, attention_mask)
            futures = self.predictor(latents[:, :-k, :])
            if self.training:
                committed = futures.commit(
                    policy(self.config.decoder.train_policy),
                    latents[:, k:, :],
                ).future
            else:
                committed = futures.commit(PiArgmax()).future

            z_t, committed = build_block_hints(
                latents, k, committed=committed.detach()
            )

        token_sum, token_count = self.decoder.token_loss(
            z_t.detach(),
            committed.detach(),
            labels,
            attention_mask=attention_mask,
            chunk_size=self.config.decoder.loss_chunk_size,
        )
        token_loss = token_sum / token_count.clamp_min(1)
        return LossOutput(
            loss=token_loss,
            terms={"token": token_loss},
            diagnostics={"perplexity": torch.exp(token_loss.detach().clamp_max(20.0))},
        )
