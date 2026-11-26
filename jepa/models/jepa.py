"""JEPA encoder + predictor + decoder and the two training objectives."""

from __future__ import annotations

import math

import torch
from transformers import PreTrainedModel

from ..config import JepaConfig
from .commitment import Policy, policy as policy_from_name
from .decoder import DecoderModel
from .encoder import EncodeTransformer
from .losses import CommitmentLoss, LossOutput, VISReg
from .predictor import PredictorModel


def horizon_targets(latents, attention_mask, horizon, *, detach=True):
    """Return ``(target, valid)``: the latent ``horizon`` tokens ahead.

    ``target[:, t]`` is ``latents[:, t + horizon]``, detached unless
    ``detach=False`` (the move term's target under ADR 0005).  ``valid`` is
    the ``(batch, seq)`` mask of positions where token ``t`` and token
    ``t + horizon`` are both real; elsewhere ``target`` is data that must not be
    used.  The shapes do not depend on ``horizon``, so a 0-dim tensor never
    triggers a recompile per value.
    """
    batch, seq_len = latents.shape[:2]
    ahead = torch.arange(seq_len, device=latents.device) + horizon
    in_range = ahead < seq_len
    ahead = ahead.clamp(max=seq_len - 1)
    valid = in_range.unsqueeze(0).expand(batch, -1)
    if attention_mask is not None:
        real = attention_mask.bool()
        valid = valid & real & real[:, ahead]
    target = latents[:, ahead, :]
    return (target.detach() if detach else target), valid


def build_block_hints(latents, horizon, committed=None):
    """Return ``(z_t, committed, valid)`` for the next-token decoder positions.

    For input position ``i`` (which predicts token ``i + 1``), the hint is the
    latent at the start of the ``horizon``-token block containing that target
    token and the future at the block end, both taken from the same block
    start.

    ``valid`` (shape ``(seq - 1,)``) is False where the block has no real
    latent ``horizon`` tokens ahead, i.e. where
    ``block_start + horizon > seq - 1``.  Those positions have no honest future
    to condition on, so the decoder loss must mask them out.

    ``horizon`` is an int or a 0-dim tensor (no recompile per value).  When
    ``committed`` is given, it holds one committed future per base position,
    for at least the first ``seq - horizon`` positions.
    ``committed[:, block_start]`` is then the predictor's commit for
    ``z_{block_start + horizon}``.

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
    valid = block_start + horizon <= seq_len - 1
    # Invalid rows reuse index 0: they are data, never used, and keeping the
    # gather in range means no position is silently paired with another block.
    index = block_start.masked_fill(~valid, 0)

    if committed is None:
        committed = latents[:, (block_start + horizon).masked_fill(~valid, 0), :]
    else:
        committed = committed[:, index, :]

    return latents[:, block_start, :], committed, valid


def hint_noise_scale(d):
    """The noise scale ``s`` that puts a blurred latent at distance ``d``.

    ``s² = 1 / (1 - d/2)² - 1``, with ``d`` detached and clamped to [0, 1]:
    adding isotropic noise of RMS ``s · ‖z‖ / √D`` gives
    ``cos(ẑ, z) ≈ 1 / √(1 + s²) = 1 - d/2``, so ``E[distance(ẑ, z)] ≈ d``.
    A 0-dim tensor in, a 0-dim tensor out: no recompile per value.
    """
    d = d.detach().float().clamp(0.0, 1.0)
    return ((1.0 - d / 2).pow(-2) - 1.0).clamp_min(0.0).sqrt()


def blur(latents, scale, generator=None):
    """``latents + scale · (‖z‖ / √D) · ε``, ε ~ N(0, I) per position.

    The per-position scale is detached: the noise level is not a gradient
    path, the latent itself is.  ``generator=None`` draws from the global RNG
    (restored on resume like the rest of training).
    """
    if generator is None:
        # ``randn_like``: a factory call with symbolic sizes breaks the graph
        # under ``torch.compile(dynamic=True)``.
        noise = torch.randn_like(latents)
    else:
        noise = torch.randn(
            latents.shape, generator=generator, device=latents.device,
            dtype=latents.dtype,
        )
    rms = latents.detach().norm(dim=-1, keepdim=True) / math.sqrt(latents.shape[-1])
    return latents + scale.to(latents.dtype) * rms * noise


class JepaModel(PreTrainedModel):
    """Full JEPA stack: encoder, predictor and autoregressive decoder.

    Training has two stages (see ``docs/adr/0002-joint-decoder-gradient.md``):

    * :meth:`jepa_loss` trains the encoder and predictor with VISReg plus the
      commitment loss, and with ``config.decoder_w > 0`` also the decoder
      through a weighted token term whose gradient reaches the encoder via
      the hint ``z_t`` (``decoder_hint = "committed"``) or the blurred block
      end (``"real"``, ADR 0004).
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

    @property
    def joint_jepa_stage(self) -> bool:
        """Whether the JEPA stage also trains the decoder (``decoder_w > 0``)."""
        return self.config.decoder_w > 0

    @property
    def loss_weights(self) -> dict[str, float]:
        """Term name -> weight for every term :meth:`jepa_loss` can return."""
        weights = {**self.visreg.weights, **self.commitment_loss.weights}
        if self.joint_jepa_stage:
            weights["token"] = float(self.config.decoder_w)
        return weights

    def _commit_rule(self, policy: Policy | None) -> Policy:
        """``policy``, or the decoder's training policy when it is ``None``."""
        if policy is not None:
            return policy
        return policy_from_name(self.config.decoder.train_policy)

    def _check_length(self, labels) -> None:
        """Reject sequences some horizon in the range could not be used on."""
        longest = self.config.horizon_range[1]
        if labels.shape[1] <= longest:
            raise ValueError(
                f"sequence length ({labels.shape[1]}) must exceed the longest "
                f"horizon ({longest})"
            )

    def _block_token_loss(self, labels, attention_mask, hints):
        """Mean decoder CE on block hints and the number of counted targets.

        ``hints`` is :func:`build_block_hints`' ``(z_t, future, valid)``.
        Gradients follow whatever the hints carry: the decoder stage passes
        them without a graph; the joint JEPA stage passes an online ``z_t``
        and a detached committed future (``"committed"``), or a detached
        ``z_t`` and an online blurred block end (``"real"``).
        """
        z_t, future, valid = hints
        # A block start past ``seq - horizon - 1`` has no future, so those
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
            future,
            labels,
            attention_mask=attention_mask,
            chunk_size=self.config.decoder.loss_chunk_size,
            loss_mask=loss_mask,
        )
        return token_sum / token_count.clamp_min(1), token_count.detach()

    def jepa_loss(
        self,
        labels,
        attention_mask=None,
        *,
        horizon,
        progress=None,
        policy: Policy | None = None,
    ) -> LossOutput:
        """Run the JEPA stage objective.

        With ``config.decoder_w > 0`` the decoder token loss is added as the
        ``token`` term.  The predictor is trained by the commitment loss alone
        in both hint modes (``config.decoder_hint``):

        * ``"committed"`` (ADR 0002): the hint is (``z_t``, the detached
          committed future); the gradient reaches the encoder through ``z_t``.
        * ``"real"`` (ADR 0004): the hint is (detached ``z_t``, the real
          block-end latent blurred by :func:`blur` to the batch's hindsight
          distance); the gradient reaches the encoder through the block end.
          The noise scale is logged as the ``hint_noise`` diagnostic.

        With ``decoder_w == 0`` the decoder never runs.

        Args:
            labels: ``(batch, seq)`` token ids.
            attention_mask: optional ``(batch, seq)``, 1 = real token.
            horizon: the batch's k (ADR 0003), a 0-dim tensor or an int in
                ``config.horizon_range``.  It places the target and the decoder
                blocks; the encoder, predictor and decoder never receive it.
            progress: 0-dim tensor with the fraction of training completed, or
                ``None`` for the final commitment temperature (evaluation).
            policy: ``None`` is the training condition: the hint of
                ``config.decoder_hint`` (committed by
                ``config.decoder.train_policy``, or the blurred real future).
                A policy is an evaluation condition in either hint mode: the
                token term decodes that commit, and the ``"real"`` mode also
                reports the noise-free real hint's CE as ``real_token``.
                Ignored when ``decoder_w == 0``.

        Returns:
            :class:`~jepa.models.losses.LossOutput` whose total is
            ``sum(loss_weights[name] * terms[name])`` and whose
            terms/diagnostics merge VISReg, the commitment loss and, when
            joint, the ``token`` term and ``tokens`` diagnostic.
        """
        self._check_length(labels)
        latents = self.encode(labels, attention_mask)
        # The commitment loss detaches the target for every term but ``move``
        # (and for that one too unless ``move_target_gradient``, ADR 0005).
        target, valid = horizon_targets(
            latents, attention_mask, horizon,
            detach=not self.config.commitment.move_target_gradient,
        )

        # From each latent[i, t, :] predict latent[i, t+k, :].  Positions with
        # no real token k ahead are predicted too and masked, so no shape
        # depends on k.
        futures = self.predictor(latents)

        # VISReg acts on the online encoder latents, not on the predictor
        # output.  Applying it to the committed future would let the predictor
        # cancel the encoder's mean in its own output without fixing the
        # encoder.
        visreg_out = self.visreg(latents, mask=attention_mask)
        commitment_out = self.commitment_loss(
            futures, target, mask=valid, progress=progress
        )
        loss = visreg_out.loss + commitment_out.loss
        terms = {**visreg_out.terms, **commitment_out.terms}
        diagnostics = {**visreg_out.diagnostics, **commitment_out.diagnostics}

        if self.joint_jepa_stage:
            real_hint = self.config.decoder_hint == "real"
            if real_hint and policy is None:
                # ADR 0004: (sg(z_t), blurred z_{t+k}).  The token loss reaches
                # the encoder through the block end only, and the committed
                # future is not part of the objective.
                scale = hint_noise_scale(commitment_out.diagnostics["hindsight_d"])
                z_t, _, valid_blocks = build_block_hints(latents, horizon)
                _, future, _ = build_block_hints(blur(latents, scale), horizon)
                hints = (z_t.detach(), future, valid_blocks)
                diagnostics["hint_noise"] = scale
            else:
                rule = self._commit_rule(policy)
                # Detached: a hindsight choice used the real future, so a
                # gradient through it would train the JEPA on information that
                # does not exist at inference.
                committed = futures.commit(
                    rule, target.detach() if rule.needs_target else None
                )
                hints = build_block_hints(
                    latents, horizon, committed=committed.future.detach()
                )
            token, count = self._block_token_loss(labels, attention_mask, hints)
            loss = loss + self.config.decoder_w * token
            terms["token"] = token
            diagnostics["tokens"] = count
            if real_hint and policy is not None:
                # Evaluation: what rendering the real future costs, next to
                # the committed one in the token term.
                real_token, _ = self._block_token_loss(
                    labels, attention_mask,
                    build_block_hints(latents.detach(), horizon),
                )
                diagnostics["real_token"] = real_token.detach()

        return LossOutput(loss=loss, terms=terms, diagnostics=diagnostics)

    def decoder_loss(
        self,
        labels,
        attention_mask=None,
        *,
        horizon,
        policy: Policy | None = None,
    ) -> LossOutput:
        """Run the decoder stage objective against the frozen JEPA.

        The encoder and predictor run under ``torch.no_grad()``.  The decoder
        is trained on committed futures; detached hints mean the decoder loss
        cannot update the JEPA.

        Args:
            labels: ``(batch, seq)`` token ids.
            attention_mask: optional ``(batch, seq)``, 1 = real token.
            horizon: the batch's k, as in :meth:`jepa_loss`: the block length
                and the offset of the hindsight target.
            policy: commit rule for the hint.  ``None`` uses
                ``config.decoder.train_policy`` (the training condition);
                validation and measurement pass ``PiArgmax()`` (the inference
                condition).  A hindsight policy commits against the real
                future ``k`` tokens ahead, which is why its hint carries up to
                ``log2(P)`` bits about that future, and may reveal k on top;
                π-argmax carries neither.

        Returns:
            :class:`~jepa.models.losses.LossOutput` with the unweighted
            ``token`` term and a ``tokens`` diagnostic (the counted target
            tokens, for token-weighted averaging).  The last block of a
            sequence runs past the end, so its target positions are masked out
            of the loss.
        """
        self._check_length(labels)
        with torch.no_grad():
            latents = self.encode(labels, attention_mask)
            futures = self.predictor(latents)
            rule = self._commit_rule(policy)
            target = None
            if rule.needs_target:
                target, _ = horizon_targets(latents, attention_mask, horizon)
            committed = futures.commit(rule, target).future

        token_loss, token_count = self._block_token_loss(
            labels, attention_mask,
            build_block_hints(latents, horizon, committed=committed),
        )
        return LossOutput(
            loss=token_loss,
            terms={"token": token_loss},
            diagnostics={"tokens": token_count},
        )
