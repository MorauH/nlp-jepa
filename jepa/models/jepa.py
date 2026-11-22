"""JEPA encoder + predictor + decoder and the two training objectives."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel

from ..config import JepaConfig
from .commitment import Policy, policy as policy_from_name
from .decoder import DecoderModel
from .encoder import EncodeTransformer
from .losses import CommitmentLoss, LossOutput, VISReg
from .predictor import PredictorModel


def build_block_hints(latents, horizon: int, committed=None):
    """Return ``(z_t, committed, valid)`` for the next-token decoder positions.

    For input position ``i`` (which predicts token ``i + 1``), the hint is the
    latent at the start of the ``horizon``-token block containing that target
    token and the future at the block end, both taken from the same block
    start.

    ``valid`` (shape ``(seq - 1,)``) is False where the block has no real
    latent ``horizon`` tokens ahead, i.e. where
    ``block_start > seq - horizon - 1``.  Those positions have no honest future
    to condition on, so the decoder loss must mask them out.

    When ``committed`` is given, it must have shape
    ``(batch, seq - horizon, hidden)``: one committed future per base
    position.  ``committed[:, block_start]`` is then the predictor's commit
    for ``z_{block_start + horizon}``.

    Information about the target token: the *raw* block-end latent
    (``committed=None``) contains the block's last token, which is a decoder
    target, so it leaks that token.  A hindsight commit is still a predictor
    output, but it was chosen among ``P`` candidates using the real future, so
    it carries up to ``log2(P)`` bits about that future (intended teacher
    forcing).  A π-argmax commit carries none.
    """
    seq_len = latents.shape[1]
    positions = torch.arange(seq_len - 1, device=latents.device)
    block_start = (positions // horizon) * horizon
    valid = block_start <= seq_len - horizon - 1
    # Invalid rows reuse index 0: they are data, never used, and keeping the
    # gather in range means no position is silently paired with another block.
    index = block_start.masked_fill(~valid, 0)

    if committed is None:
        committed = latents[:, (block_start + horizon).masked_fill(~valid, 0), :]
    else:
        committed = committed[:, index, :]

    return latents[:, block_start, :], committed, valid


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

    def encode(self, labels, attention_mask=None):
        """Encode tokens to latents, dropping the padding mask when that is safe.

        Batches are right-padded and the encoder is causal, so no real query
        can attend a pad key: dropping the mask enables the fused causal SDPA
        kernel.  The mask still validates loss positions, so callers keep it
        for the losses.
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
        latents = self.encode(labels, attention_mask)

        k = self.pred_horizon
        seq_len = latents.shape[1]
        if seq_len <= k:
            raise ValueError(
                f"sequence length ({seq_len}) must exceed pred_horizon ({k})"
            )

        X = latents[:, :-k, :]
        Y = latents[:, k:, :].detach()

        if attention_mask is not None:
            valid = attention_mask[:, :-k].bool() & attention_mask[:, k:].bool()
        else:
            valid = None

        # From each latent[i, t, :] predict latent[i, t+k, :]
        futures = self.predictor(X)

        # VISReg acts on the online encoder latents, not on the predictor
        # output.  Applying it to the committed future would let the predictor
        # cancel the encoder's mean in its own output without fixing the
        # encoder.
        visreg_out = self.visreg(latents, mask=attention_mask)
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
        policy: Policy | None = None,
    ) -> LossOutput:
        """Run the decoder stage objective against the frozen JEPA.

        The encoder and predictor run under ``torch.no_grad()``.  The decoder
        is trained on committed futures; detached hints mean the decoder loss
        cannot update the JEPA.

        Args:
            labels: ``(batch, seq)`` token ids.
            attention_mask: optional ``(batch, seq)``, 1 = real token.
            policy: commit rule for the hint.  ``None`` uses
                ``config.decoder.train_policy`` (the training condition);
                validation and measurement pass ``PiArgmax()`` (the inference
                condition).  A hindsight policy commits against the real
                future ``k`` tokens ahead, which is why its hint carries up to
                ``log2(P)`` bits about that future; π-argmax carries none.

        Returns:
            :class:`~jepa.models.losses.LossOutput` with the unweighted
            ``token`` term and a ``tokens`` diagnostic (the counted target
            tokens, for token-weighted averaging).  The last block of a
            sequence runs past the end, so its target positions are masked out
            of the loss.
        """
        k = self.pred_horizon
        seq_len = labels.shape[1]
        if seq_len <= k:
            raise ValueError(
                f"sequence length ({seq_len}) must exceed pred_horizon ({k})"
            )

        with torch.no_grad():
            latents = self.encode(labels, attention_mask)
            futures = self.predictor(latents[:, :-k, :])
            rule = (
                policy_from_name(self.config.decoder.train_policy)
                if policy is None
                else policy
            )
            target = latents[:, k:, :] if rule.needs_target else None
            committed = futures.commit(rule, target).future
            z_t, committed, valid = build_block_hints(
                latents, k, committed=committed
            )

        # A block start past ``seq - k - 1`` has no committed future, so those
        # targets are excluded instead of being paired with another block.
        loss_mask = valid.unsqueeze(0)
        if attention_mask is not None:
            loss_mask = (
                loss_mask
                & attention_mask[:, :-1].bool()
                & attention_mask[:, 1:].bool()
            )

        token_sum, token_count = self.decoder.token_loss(
            z_t,
            committed,
            labels,
            attention_mask=attention_mask,
            chunk_size=self.config.decoder.loss_chunk_size,
            loss_mask=loss_mask,
        )
        token_loss = token_sum / token_count.clamp_min(1)
        return LossOutput(
            loss=token_loss,
            terms={"token": token_loss},
            diagnostics={"tokens": token_count.detach()},
        )
