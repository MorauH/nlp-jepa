"""VISReg regularization and the multi-hypothesis JEPA objective."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import VISRegConfig


class VISRegLoss(nn.Module):
    """
    VISReg regularization plus a multi-hypothesis prediction objective.

    The predictor returns a set of candidate latents with per-candidate
    scores (negated predicted distances).  The single-prediction invariance
    term is replaced by:

    * a relaxed winner-takes-all move loss that pulls each target toward its
      nearest candidate (with a small uniform component so no candidate dies),
    * a value regression that trains the predictor to predict each
      candidate's distance to the target from ``z_t`` alone, so selection
      minimizes the expected distance,
    * an optional diversity penalty that stops candidates collapsing onto the
      same direction.

    The scale/shape/center regularizers are unchanged.  They are evaluated on
    the online encoder latents (``reg_source``), so they shape the
    representation the predictor consumes rather than the predictor output.

    Weights live in :class:`jepa.config.VISRegConfig`.
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
        self.scale_w = config.scale_w
        self.shape_w = config.shape_w
        self.center_w = config.center_w
        self.move_w = config.move_w
        self.score_w = config.score_w
        self.diversity_w = config.diversity_w
        self.diversity_margin = config.diversity_margin
        self.mh_temperature = config.mh_temperature
        self.mh_temperature_start = config.mh_temperature_start
        self.mh_anneal_fraction = config.mh_anneal_fraction
        self.mh_eps = config.mh_eps
        self.normalize_distance = config.normalize_distance
        self.eps = config.eps

        # Current temperature multiplier.  A tensor buffer, not a Python
        # float: torch.compile guards on float attributes and would recompile
        # every step while it anneals, whereas buffer values are graph inputs.
        # Non-persistent, so checkpoints are unchanged; it starts at the final
        # value, which is what evaluation of a trained model should use.
        self.register_buffer(
            "mh_temperature_now",
            torch.tensor(float(config.mh_temperature)),
            persistent=False,
        )

    @torch.no_grad()
    def reset_annealing(self) -> None:
        """Set the temperature multiplier back to its final value.

        Called after loading (see ``JepaModel._init_weights``): the buffer is
        not saved, so a loaded model would otherwise hold uninitialized memory.
        """
        self.mh_temperature_now.fill_(float(self.mh_temperature))

    @torch.no_grad()
    def set_anneal_progress(self, progress: float) -> None:
        """Set the annealed temperature for training progress in ``[0, 1]``.

        Decays geometrically from ``mh_temperature_start`` to
        ``mh_temperature`` over the first ``mh_anneal_fraction`` of training,
        then stays at ``mh_temperature``.  No-op without annealing.
        """
        if self.mh_temperature_start is None:
            return
        t = min(1.0, max(0.0, progress / self.mh_anneal_fraction))
        value = self.mh_temperature_start * (
            self.mh_temperature / self.mh_temperature_start
        ) ** t
        self.mh_temperature_now.fill_(value)

    def _reg_terms(self, z):
        """Returns (scale, shape, center) losses for a single tensor."""
        N, D = z.shape
        device = z.device

        if N < 2:  # edge case: almost no valid tokens
            return z.new_zeros(()), z.new_zeros(()), z.new_zeros(())

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

        return l_scale, l_shape, l_center

    def _multi_hypothesis_loss(
        self, candidates, target, scores, mask, reg_source, reg_mask
    ):
        """Loss for ``(B, T, P, D)`` candidate latents and ``(B, T, P)`` scores.

        ``r`` is the detached soft assignment derived from the true
        candidate-target distances; the move loss pulls candidates through
        ``d``.  ``scores`` are negated predicted distances, regressed onto the
        detached ``d``: the value head learns ``E[d_i | z_t]``, not how often
        candidate ``i`` wins (the argmax of win frequency can have a worse
        average distance than a constant choice).

        ``reg_source`` holds the online encoder latents that VISReg acts on.
        Regularizing those directly prevents the predictor from satisfying
        the distribution terms by transforming its own output while leaving
        the encoder collapsed.
        """
        batch, seq, count, _ = candidates.shape
        device = candidates.device

        if mask is None:
            valid = torch.ones((batch, seq), dtype=torch.bool, device=device)
        else:
            valid = mask.bool().to(device)

        cand = candidates.float()
        tgt = target.float()
        predicted_d = -scores.float()
        valid_f = valid.to(cand.dtype)
        denom = valid_f.sum().clamp_min(1.0)

        if self.normalize_distance:
            cand_cmp = F.normalize(cand, dim=-1)
            tgt_cmp = F.normalize(tgt, dim=-1)
            # Both sides are unit vectors, so the squared distance is
            # 2 - 2 * (a . b).  Computing it from the dot product avoids
            # materializing the (B, T, P, D) fp32 difference and its square
            # (about 1.4 GB each at batch 32 x 1024).  Clamping at zero hides
            # the fp32 rounding when a == b.
            dot = torch.einsum("btpd,btd->btp", cand_cmp, tgt_cmp)
            d = (2.0 - 2.0 * dot).clamp_min(0.0)
        else:
            cand_cmp = cand
            d = (cand - tgt.unsqueeze(2)).pow(2).sum(-1)

        # Adaptive temperature: relative differences stay usable even as the
        # overall distance shrinks during training.
        d_mean = (d.detach() * valid_f.unsqueeze(-1)).sum() / (denom * count)
        temperature = (
            self.mh_temperature_now.to(d_mean.dtype) * d_mean
        ).clamp_min(1e-6)

        # Relaxed winner-takes-all assignment.  The uniform smoothing keeps
        # every candidate alive instead of only the current winner.
        with torch.no_grad():
            r = F.softmax(-d.detach() / temperature, dim=-1)
            r = (1.0 - self.mh_eps) * r + self.mh_eps / count
            r = r * valid_f.unsqueeze(-1)

        l_move = (r * d).sum(-1)  # (B, T)
        l_move = (l_move * valid_f).sum() / denom

        # Value regression: the head sees only z_t, so it learns the expected
        # distance of every candidate, and argmin d̂ minimizes the expected
        # distance of the selected waypoint.  All candidates are supervised
        # (not only the winner), so dead slots still get calibrated values.
        d_target = d.detach()
        l_value = (predicted_d - d_target).pow(2).mean(-1)  # (B, T)
        l_value = (l_value * valid_f).sum() / denom

        # Distance of the value-selected candidate (the inference choice).
        picked = predicted_d.detach().argmin(dim=-1, keepdim=True)
        selected_d = d_target.gather(-1, picked).squeeze(-1)
        selected_d = (selected_d * valid_f).sum() / denom

        # Optional repulsion between candidates.  Off-diagonal cosine
        # similarity is penalized only above ``diversity_margin``, i.e. only
        # for near-duplicates.  With margin 0 any positive cosine costs,
        # and pointing orthogonal to everything becomes a free hiding place
        # for unused slots.
        if self.diversity_w > 0 and count > 1:
            dirs = (
                cand_cmp if self.normalize_distance
                else F.normalize(cand, dim=-1)
            )
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
            ).sum() / (
                (n_valid * count * (count - 1)).clamp_min(1)
            )
        else:
            l_div = cand.new_zeros(())

        # The candidate closest to the target is the committed future under
        # teacher forcing.  It is also returned for the decoder.
        win = d.detach().argmin(dim=-1)  # (B, T)

        # VISReg acts on the online encoder latents, not on the predictor
        # output.  Applying it to the winner would let the predictor cancel
        # the encoder's mean in its own output without fixing the encoder.
        reg_z = reg_source.float()
        if reg_mask is not None:
            reg_valid = reg_mask.bool().to(reg_z.device)
            reg_z = reg_z[reg_valid]
        else:
            reg_z = reg_z.reshape(-1, reg_z.shape[-1])
        l_scale, l_shape, l_center = self._reg_terms(reg_z)

        loss = (
            self.move_w * l_move
            + self.score_w * l_value
            + self.diversity_w * l_div
            + self.scale_w * l_scale
            + self.shape_w * l_shape
            + self.center_w * l_center
        )

        stats = {
            "loss_move": l_move.detach(),
            "loss_value": l_value.detach(),
            "loss_diversity": l_div.detach(),
            "loss_scale": l_scale.detach(),
            "loss_shape": l_shape.detach(),
            "loss_center": l_center.detach(),
            "best_d": (d.detach().min(-1).values * valid_f).sum() / denom,
            "selected_d": selected_d,
            "temperature": temperature.detach(),
            "win": win,
        }
        return loss, stats

    def forward(
        self,
        pred,
        target,
        mask=None,
        scores=None,
        reg_source=None,
        reg_mask=None,
        return_stats=False,
    ):
        """
        Args:
            pred: (B, T, P, D) candidate predictions.
            target: (B, T, D)
            mask: (B, T), 1 => valid prediction position
            scores: (B, T, P) negated predicted candidate-target distances.
            reg_source: (B, S, D) online encoder latents for VISReg.
            reg_mask: (B, S), 1 => valid token in ``reg_source``.
            return_stats: also return detached diagnostics for logging.

        Returns:
            The loss, or ``(loss, stats)`` when ``return_stats`` is True.
        """
        if pred.ndim != 4:
            raise ValueError(
                f"Expected candidate predictions (B, T, P, D); got {tuple(pred.shape)}"
            )
        if scores is None:
            raise ValueError("scores is required for candidate predictions")
        if reg_source is None:
            raise ValueError("reg_source (online encoder latents) is required")

        device_type = "cuda" if pred.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            loss, stats = self._multi_hypothesis_loss(
                pred, target, scores, mask, reg_source, reg_mask
            )
        return (loss, stats) if return_stats else loss
