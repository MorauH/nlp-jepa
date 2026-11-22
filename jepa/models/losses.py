"""The JEPA objective: VISReg regularization and the commitment loss.

The two modules are independent:

* :class:`VISReg` is the anti-collapse regularizer on the online encoder
  latents (scale / shape / center).
* :class:`CommitmentLoss` is the paper's core objective on the predictor's
  :class:`jepa.models.commitment.Futures`: a relaxed winner-takes-all move
  loss, a cross-entropy that trains pi, a regression that trains mu, and an
  optional candidate-diversity penalty.

Both report a :class:`LossOutput`, so the total is always
``sum(weights[name] * terms[name])``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import CommitmentLossConfig, VISRegConfig


class LossOutput(NamedTuple):
    """A loss plus the pieces needed to check it and to log it.

    loss:        weighted total.
    terms:       unweighted scalar loss terms.  ``loss`` is exactly
                 ``sum(weights[name] * terms[name])``.
    diagnostics: detached scalars that are not part of the loss.
    """

    loss: Tensor
    terms: dict[str, Tensor]
    diagnostics: dict[str, Tensor]


class VISReg(nn.Module):
    """Anti-collapse regularizer on the online encoder latents.

    Three distribution terms push the latent cloud toward a standard normal:

    * scale: per-dimension standard deviation toward 1,
    * shape: sliced Wasserstein distance to Gaussian quantiles,
    * center: mean vector toward 0.
    """

    def __init__(self, config: VISRegConfig | None = None, **overrides):
        super().__init__()

        if config is None:
            config = VISRegConfig(**overrides)
        elif overrides:
            raise ValueError("Pass either `config` or keyword overrides, not both.")

        config.validate()

        self.config = config
        self.num_slices = config.num_slices
        self.eps = config.eps
        self.weights = {
            "scale": config.scale_w,
            "shape": config.shape_w,
            "center": config.center_w,
        }

    def _terms(self, z: Tensor) -> dict[str, Tensor]:
        """Scale/shape/center for a single ``(N, D)`` fp32 tensor."""
        N, D = z.shape
        device = z.device
        zero = z.new_zeros(())

        if N < 2:  # edge case: almost no valid tokens
            return {"scale": zero, "shape": zero, "center": zero}

        # Center
        mu = z.mean(dim=0)
        l_center = (mu**2).mean()

        # Scale
        z_cent = z - mu
        std = z_cent.std(dim=0, unbiased=False)
        l_scale = ((1.0 - std) ** 2).mean()

        # Shape (SWD to Gaussian quantiles)
        z_norm = z_cent / (std.detach() + self.eps)

        directions = torch.randn(D, self.num_slices, device=device)
        directions = F.normalize(directions, dim=0)

        projections = z_norm @ directions  # (N, K)
        sorted_proj, _ = torch.sort(projections, dim=0)

        # Quantile positions: i / (N+1)
        u = torch.arange(1, N + 1, device=device, dtype=z.dtype) / (N + 1)
        # Standard normal inverse CDF, without constructing a Normal object
        # (which is not safe under transformers' meta-device initialization).
        target_q = (torch.erfinv(2 * u - 1) * math.sqrt(2.0)).unsqueeze(1)

        l_shape = ((sorted_proj - target_q) ** 2).mean()

        return {"scale": l_scale, "shape": l_shape, "center": l_center}

    def forward(self, latents: Tensor, mask: Tensor | None = None) -> LossOutput:
        """Regularize the online encoder latents.

        Args:
            latents: ``(..., D)`` encoder latents.
            mask: optional ``(...)`` boolean/0-1 mask; only masked positions
                contribute.

        Returns:
            :class:`LossOutput` with terms ``scale``, ``shape``, ``center``
            and no diagnostics.
        """
        device_type = "cuda" if latents.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            z = latents.float()
            if mask is not None:
                z = z[mask.bool().to(z.device)]
            else:
                z = z.reshape(-1, z.shape[-1])
            terms = self._terms(z)
            loss = sum(self.weights[name] * terms[name] for name in terms)
        return LossOutput(loss=loss, terms=terms, diagnostics={})


class CommitmentLoss(nn.Module):
    """The commitment objective on the predictor's proposal.

    :class:`~jepa.models.commitment.Futures` holds ``(B, T, P, D)``
    candidates, ``(B, T, P)`` win-probability logits pi, and a ``(B, T, D)``
    mean future mu.  The single-prediction invariance term is replaced by:

    * a relaxed winner-takes-all move loss that pulls each target toward its
      nearest candidate (with a small uniform component so no candidate dies),
    * a cross-entropy that trains pi to predict the same soft assignment from
      ``z_t`` alone (which candidate will be closest; used for sampling),
    * a regression of mu onto the normalized target, so mu learns
      ``E[ŷ | z_t]`` (auxiliary; its input is detached in the predictor),
    * an optional diversity penalty that stops candidates collapsing onto the
      same direction.

    Weights and annealing live in
    :class:`jepa.config.CommitmentLossConfig`.
    """

    def __init__(self, config: CommitmentLossConfig | None = None, **overrides):
        super().__init__()

        if config is None:
            config = CommitmentLossConfig(**overrides)
        elif overrides:
            raise ValueError("Pass either `config` or keyword overrides, not both.")

        config.validate()

        self.config = config
        self.diversity_margin = config.diversity_margin
        self.temperature = config.temperature
        self.temperature_start = config.temperature_start
        self.anneal_fraction = config.anneal_fraction
        self.smoothing = config.smoothing
        self.weights = {
            "move": config.move_w,
            "pi": config.pi_w,
            "mu": config.mu_w,
            "diversity": config.diversity_w,
        }

    def temperature_multiplier(self, progress: Tensor | None, like: Tensor) -> Tensor:
        """Annealed multiplier on the soft-assignment temperature.

        ``progress`` is the fraction of training completed, a 0-dim tensor in
        ``[0, 1]``.  ``None`` means 1.0, i.e. the final temperature, which is
        what evaluation should use.  Everything is tensor arithmetic (no
        Python branch on the value), so ``torch.compile`` does not recompile
        as training progresses.
        """
        if progress is None:
            progress = like.new_ones(())
        else:
            progress = torch.as_tensor(progress, dtype=like.dtype, device=like.device)
        fraction = (progress / self.anneal_fraction).clamp(0.0, 1.0)
        return self.temperature_start * (
            self.temperature / self.temperature_start
        ) ** fraction

    def _loss(
        self,
        futures,
        target: Tensor,
        mask: Tensor | None,
        progress: Tensor | None,
    ) -> LossOutput:
        """Loss for ``(B, T, P, D)`` candidates, ``(B, T, P)`` pi logits and a
        ``(B, T, D)`` mu from the same :class:`Futures`.

        ``r`` is the detached soft assignment derived from the true
        candidate-target distances.  The move loss pulls candidates through
        ``d``; pi is trained to predict ``r`` (which candidate is closest).
        pi is a sampling distribution over futures.  It is *not* an
        expected-distance selector: under the normalized metric the expected
        distance is ``2 - 2 ĉ_i · mu``, minimized by the candidate nearest the
        mean future, which no multi-candidate head can beat a single mean
        predictor at.
        """
        candidates = futures.candidates
        batch, seq, count, _ = candidates.shape
        device = candidates.device

        if mask is None:
            valid = torch.ones((batch, seq), dtype=torch.bool, device=device)
        else:
            valid = mask.bool().to(device)

        cand = candidates.float()
        tgt = target.float()
        logits = futures.pi.float()
        mu = futures.mu.float()
        valid_f = valid.to(cand.dtype)
        denom = valid_f.sum().clamp_min(1.0)

        # The one candidate-target distance in the codebase: fp32, normalized,
        # computed from the dot product (no (B, T, P, D) difference).  The
        # clamp hides fp32 rounding when a candidate equals the target.
        d = futures.distances(target)

        # Adaptive temperature: relative differences stay usable even as the
        # overall distance shrinks during training.
        multiplier = self.temperature_multiplier(progress, cand)
        d_mean = (d.detach() * valid_f.unsqueeze(-1)).sum() / (denom * count)
        temperature = (multiplier.to(d_mean.dtype) * d_mean).clamp_min(1e-6)

        # Relaxed winner-takes-all assignment.  The uniform smoothing keeps
        # every candidate alive instead of only the current winner.
        with torch.no_grad():
            r = F.softmax(-d.detach() / temperature, dim=-1)
            r = (1.0 - self.smoothing) * r + self.smoothing / count
            r = r * valid_f.unsqueeze(-1)

        l_move = (r * d).sum(-1)  # (B, T)
        l_move = (l_move * valid_f).sum() / denom

        # pi: the head sees only z_t, so this teaches it how likely each
        # candidate is to be the closest, without access to the target.
        log_probs = F.log_softmax(logits, dim=-1)
        l_pi = -(r * log_probs).sum(-1)  # (B, T)
        l_pi = (l_pi * valid_f).sum() / denom

        def masked_mean(values):
            return (values * valid_f).sum() / denom

        d_detached = d.detach()
        # Distance of the pi-argmax candidate (greedy generation's choice).
        picked = logits.detach().argmax(dim=-1, keepdim=True)
        pi_argmax_d = masked_mean(d_detached.gather(-1, picked).squeeze(-1))

        # mu: regress onto the normalized target, so mu -> E[ŷ | z_t].  Its
        # input is detached in the predictor, so this only trains mean_head.
        tgt_unit = F.normalize(tgt, dim=-1)
        l_mu = masked_mean((mu - tgt_unit).pow(2).sum(-1))
        with torch.no_grad():
            mu_norm = masked_mean(mu.norm(dim=-1))
            # The direction of mu is the optimal single-head prediction: what
            # multiple candidates must beat.
            mu_dir_d = masked_mean(
                2.0 - 2.0 * (F.normalize(mu, dim=-1) * tgt_unit).sum(-1)
            )

        # Optional repulsion between candidates.  Off-diagonal cosine
        # similarity is penalized only above ``diversity_margin``, i.e. only
        # for near-duplicates.  With margin 0 any positive cosine costs,
        # and pointing orthogonal to everything becomes a free hiding place
        # for unused slots.
        if self.weights["diversity"] > 0 and count > 1:
            dirs = F.normalize(cand, dim=-1)
            n_valid = valid_f.sum()
            # Pairwise similarities for every position, then sum only over the
            # valid ones.  Gathering the valid rows first (the old path) copied
            # and re-normalized an (N, P, D) fp32 tensor; the bmm output is
            # only (N, P, P) and is multiplied by the mask.
            dirs = dirs.reshape(-1, count, dirs.shape[-1])
            sim = torch.bmm(dirs, dirs.transpose(1, 2))  # (N, P, P)
            off_diag = 1.0 - torch.eye(count, device=device, dtype=sim.dtype)
            mask_div = valid_f.reshape(-1, 1, 1)
            l_div = (
                F.relu(sim - self.diversity_margin) * off_diag * mask_div
            ).sum() / ((n_valid * count * (count - 1)).clamp_min(1))
        else:
            l_div = cand.new_zeros(())

        terms = {
            "move": l_move,
            "pi": l_pi,
            "mu": l_mu,
            "diversity": l_div,
        }
        loss = sum(self.weights[name] * terms[name] for name in terms)
        diagnostics = {
            "best_d": (d_detached.min(-1).values * valid_f).sum() / denom,
            "pi_argmax_d": pi_argmax_d,
            "mu_norm": mu_norm,
            "mu_dir_d": mu_dir_d,
            "temperature": multiplier.detach(),
        }
        return LossOutput(loss=loss, terms=terms, diagnostics=diagnostics)

    def forward(
        self,
        futures,
        target: Tensor,
        mask: Tensor | None = None,
        progress: Tensor | None = None,
    ) -> LossOutput:
        """
        Args:
            futures: :class:`jepa.models.commitment.Futures` with
                ``(B, T, P, D)`` candidates, ``(B, T, P)`` pi logits and
                ``(B, T, D)`` mu.
            target: ``(B, T, D)``.
            mask: ``(B, T)``, 1 => valid prediction position.
            progress: 0-dim tensor with the fraction of training completed, or
                ``None`` for the final temperature (evaluation).

        Returns:
            :class:`LossOutput` with terms ``move``, ``pi``, ``mu``,
            ``diversity`` and detached diagnostics.
        """
        candidates = futures.candidates
        if candidates.ndim != 4:
            raise ValueError(
                "Expected candidate predictions (B, T, P, D); got "
                f"{tuple(candidates.shape)}"
            )
        if target.shape != futures.mu.shape:
            raise ValueError(
                f"target shape {tuple(target.shape)} must match mu shape "
                f"{tuple(futures.mu.shape)}"
            )

        device_type = "cuda" if candidates.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            return self._loss(futures, target, mask, progress)
