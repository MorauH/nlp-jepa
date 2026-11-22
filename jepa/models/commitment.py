"""Commitment: choose one candidate and continue from it.

The predictor proposes ``P`` candidates for the future latent ``z_{t+k}``,
win-probability logits π over them, and the mean future μ.  A **commit** turns
that proposal into one future: a policy scores the candidates and the highest
score wins.

Invariant: a committed future is always exactly one of the candidates, never a
mix of them.  A policy only ranks the candidates; the value that leaves this
module is a copy of one row of :attr:`Futures.candidates`, namely the row at
``Commit.index``.  Whether the committed future stays a clear single future is
the research question; averaging candidates is never the answer the code gives.

This module is the single home of the candidate-target distance
``d = 2 - 2 cos(candidate, target)``.  It is always computed in fp32 with
autocast disabled, from normalized vectors and their dot product.  The VISReg
loss, training-time hindsight, inference, sampling and measurement all call
:meth:`Futures.distances`, so they agree on which candidate is closest.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor


def _no_autocast(tensor: Tensor):
    """Autocast-off context for fp32 distance arithmetic on ``tensor``."""
    device_type = "cuda" if tensor.is_cuda else "cpu"
    return torch.autocast(device_type=device_type, enabled=False)


class Futures(NamedTuple):
    """The predictor's proposal for ``z_{t+k}``.

    candidates: ``(..., P, D)`` candidate future latents.
    pi:         ``(..., P)`` win-probability logits: how likely each candidate
                is the closest one to the real future.  Sample from these to
                commit without seeing the future.
    mu:         ``(..., D)`` mean normalized future.  It is the blurry
                single-prediction baseline commitment is compared against, and
                its norm measures how spread the real futures are (~1 =
                predictable, short = divergent).
    """

    candidates: Tensor
    pi: Tensor
    mu: Tensor

    def distances(self, target: Tensor) -> Tensor:
        """Candidate-target distances, ``(..., P)`` fp32.

        ``d = 2 - 2 cos(candidate, target)``, clamped at 0 to hide the fp32
        rounding when the vectors coincide.  Both sides are normalized and the
        distance is computed from the dot product, so no ``(..., P, D)``
        difference is materialized.  This is the only candidate-target
        distance in the codebase.
        """
        with _no_autocast(self.candidates):
            cand = F.normalize(self.candidates.float(), dim=-1)
            tgt = F.normalize(target.float(), dim=-1)
            dot = torch.einsum("...pd,...d->...p", cand, tgt)
            return (2.0 - 2.0 * dot).clamp_min(0.0)

    def expected_distances(self) -> Tensor:
        """Expected candidate-target distances under μ, ``(..., P)`` fp32.

        For unit vectors ``d = 2 - 2 ĉ_i·ŷ``, and the candidate is fixed given
        ``z_t``, so ``E[d_i | z_t] = 2 - 2 ĉ_i·E[ŷ | z_t] = 2 - 2 ĉ_i·μ``.
        μ is not renormalized: it is a mean of unit futures, and its norm is
        the predictability signal.
        """
        with _no_autocast(self.candidates):
            cand = F.normalize(self.candidates.float(), dim=-1)
            dot = torch.einsum("...pd,...d->...p", cand, self.mu.float())
            return 2.0 - 2.0 * dot

    def commit(self, policy: "Policy", target: Tensor | None = None) -> "Commit":
        """Score the candidates with ``policy`` and take the highest score.

        ``target`` is the real future latent, ``(..., D)``, needed only by
        policies that commit by hindsight.  Policies that do not need it ignore
        it.  Returns the committed future (one candidate, in the candidates'
        dtype, not detached) and its index.
        """
        if policy.needs_target and target is None:
            raise ValueError(
                f"{type(policy).__name__} commits by hindsight and needs the "
                "real future; pass target=..."
            )
        if target is not None and target.shape != self.mu.shape:
            raise ValueError(
                f"target shape {tuple(target.shape)} must match mu shape "
                f"{tuple(self.mu.shape)}"
            )

        scores = policy.scores(self, target)
        index = scores.argmax(dim=-1)
        width = self.candidates.shape[-1]
        future = self.candidates.gather(
            -2, index.unsqueeze(-1).unsqueeze(-1).expand(*index.shape, 1, width)
        ).squeeze(-2)
        return Commit(future=future, index=index)


class Commit(NamedTuple):
    """The outcome of a commit.

    future: ``(..., D)`` the committed future, in the candidates' dtype.  Not
            detached; callers detach when the commit is only a target.
    index:  ``(...,)`` long index of the chosen candidate.
    """

    future: Tensor
    index: Tensor


def rollout(
    predictor,
    z: Tensor,
    policy: "Policy",
    steps: int,
    targets: Tensor | None = None,
) -> Commit:
    """Chain ``steps`` commits from ``z`` without re-encoding any tokens.

    ``z_1 = commit(predictor(z))``, ``z_2 = commit(predictor(z_1))``, ... Each
    committed future is exactly one of the predictor's candidates, so it lives
    at the encoder-latent scale and can be fed straight back into the
    predictor.  No tokens (and no encoder) are involved.

    Args:
        predictor: any callable ``z -> Futures``, e.g. a
            :class:`~jepa.models.predictor.PredictorModel`.
        z: ``(..., D)`` starting latent.
        policy: the commit rule used at every depth.  A hindsight policy needs
            ``targets``.
        steps: number of commits, at least 1.
        targets: ``(..., steps, D)`` real latents at depths 1..steps; required
            iff ``policy.needs_target``.  Depth ``i`` (1-based) commits against
            ``targets[..., i - 1, :]``.

    Returns:
        :class:`Commit` with ``future`` ``(..., steps, D)`` and ``index``
        ``(..., steps)``.

    Gradients are not detached; the caller decides.  This is evaluation-only
    for now: the predictor is trained on real latents only.  A multi-step
    training loss could later be built on top of this function.
    """
    if steps < 1:
        raise ValueError(f"steps must be at least 1; got {steps!r}")
    if policy.needs_target and targets is None:
        raise ValueError(
            f"{type(policy).__name__} commits by hindsight and needs the real "
            "path; pass targets=..."
        )
    if targets is not None:
        expected = (*z.shape[:-1], steps, z.shape[-1])
        if tuple(targets.shape) != expected:
            raise ValueError(
                f"targets shape {tuple(targets.shape)} must be {expected}"
            )

    futures: list[Tensor] = []
    indices: list[Tensor] = []
    for depth in range(steps):
        target = None if targets is None else targets[..., depth, :]
        commit = predictor(z).commit(policy, target)
        z = commit.future
        futures.append(commit.future)
        indices.append(commit.index)

    return Commit(torch.stack(futures, dim=-2), torch.stack(indices, dim=-1))


class Policy(Protocol):
    """A rule that scores candidates so the highest score can be committed."""

    needs_target: bool

    def scores(self, futures: Futures, target: Tensor | None) -> Tensor:
        """``(..., P)`` scores, higher wins."""
        ...


class Hindsight:
    """Commit to the candidate closest to the real future.

    Available only where the real future is known (training and measurement),
    never at inference.
    """

    needs_target = True

    def scores(self, futures: Futures, target: Tensor | None) -> Tensor:
        return -futures.distances(target)


class PiArgmax:
    """Commit to the most likely candidate under π."""

    needs_target = False

    def scores(self, futures: Futures, target: Tensor | None = None) -> Tensor:
        return futures.pi


class PiSample:
    """Sample a candidate from π.

    Gumbel-max sampling: optionally mask π to the ``top_k`` logits, then draw
    ``argmax(pi / temperature + Gumbel noise)``.  A non-positive temperature
    commits by plain argmax (the most likely candidate).
    """

    needs_target = False

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator=None,
    ):
        self.temperature = float(temperature)
        self.top_k = top_k
        self.generator = generator

    def scores(self, futures: Futures, target: Tensor | None = None) -> Tensor:
        logits = futures.pi
        if self.temperature <= 0:
            return logits

        if self.top_k is not None and self.top_k > 0:
            k = min(int(self.top_k), logits.shape[-1])
            kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        logits = logits / self.temperature
        # Draw the noise in fp32 so the two logarithms have full exponent
        # range even when pi is bf16/fp16 under autocast.
        noise = torch.rand(
            logits.shape,
            generator=self.generator,
            device=logits.device,
            dtype=torch.float32,
        )
        # Gumbel noise: -log(-log(u)), with u kept strictly positive so the
        # double logarithm stays finite.  torch.rand already returns u < 1.
        u = noise.clamp_min(torch.finfo(torch.float32).tiny)
        return logits.float() - torch.log(-torch.log(u))


class MuSafe:
    """Commit to the candidate with the lowest expected distance under μ."""

    needs_target = False

    def scores(self, futures: Futures, target: Tensor | None = None) -> Tensor:
        return -futures.expected_distances()


POLICIES: dict[str, type] = {
    "hindsight": Hindsight,
    "pi_argmax": PiArgmax,
    "pi_sample": PiSample,
    "mu_safe": MuSafe,
}


def policy(name: str, **kwargs) -> Policy:
    """Build the policy registered under ``name``."""
    try:
        cls = POLICIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown policy {name!r}; expected one of {sorted(POLICIES)}"
        ) from None
    return cls(**kwargs)
