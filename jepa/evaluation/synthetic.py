"""The synthetic world: a two-successor Markov chain with known futures.

``N`` states live at the encoder-latent scale (unit-normalised vectors scaled
to ``sqrt(D)``, so their per-dim RMS is 1).  Each state has exactly two
distinct successors, chosen once at random.  A transition picks one of the two
with probability 1/2.  There is no encoder: the state latents are the real
latents.

Only a tiny :class:`~jepa.models.predictor.PredictorModel` is trained, with
:class:`~jepa.models.losses.CommitmentLoss` (no VISReg), on (state, sampled
successor) pairs.

Shared by ``tests/test_research_claim.py`` (the claim as assertions) and
``docs/report/synthetic.py`` (the report's numbers over seeds), so both use
one world and one training loop.  The μ path is
:func:`jepa.evaluation.measure.mu_path`, the same one the text metrics use.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import PredictorConfig
from ..models import CommitmentLoss
from ..models.commitment import pairwise_distance
from ..models.predictor import PredictorModel

N_STATES = 8
DIM = 32
BATCH_SIZE = 128
TRAIN_STEPS = 400
LEARNING_RATE = 3e-3
# A rollout latent "lies on a state" below this distance to it.
ON_STATE_DISTANCE = 0.1


class World(NamedTuple):
    """states: ``(N, D)`` state latents.  successors: ``(N, 2)`` long."""

    states: Tensor
    successors: Tensor


def build_world(seed: int) -> World:
    """Fixed state latents and two distinct successors per state."""
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(N_STATES, DIM, generator=generator)
    # Norm sqrt(D) <=> per-dim RMS 1, the predictor's output scale.
    states = F.normalize(states, dim=-1) * math.sqrt(DIM)

    successors = torch.empty(N_STATES, 2, dtype=torch.long)
    for state in range(N_STATES):
        order = torch.randperm(N_STATES, generator=generator).tolist()
        choices = [other for other in order if other != state][:2]
        successors[state] = torch.tensor(choices)
    return World(states, successors)


def train_predictor(
    world: World, prediction_count: int, seed: int
) -> PredictorModel:
    """Fit a tiny predictor on (state, sampled successor) pairs."""
    states, successors = world
    torch.manual_seed(seed)
    predictor = PredictorModel(
        PredictorConfig(
            target_dim=DIM,
            hidden_size=DIM,
            num_attention_heads=4,
            num_hidden_layers=1,
            prediction_count=prediction_count,
        )
    )
    criterion = CommitmentLoss()
    optimiser = torch.optim.Adam(predictor.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed + 1)

    for step in range(TRAIN_STEPS):
        state = torch.randint(0, N_STATES, (BATCH_SIZE,), generator=generator)
        choice = torch.randint(0, 2, (BATCH_SIZE,), generator=generator)
        target = states[successors[state, choice]]

        # CommitmentLoss consumes one extra sequence axis.
        futures = predictor(states[state].unsqueeze(1))
        progress = torch.tensor(step / TRAIN_STEPS)
        output = criterion(futures, target.unsqueeze(1), progress=progress)

        optimiser.zero_grad()
        output.loss.backward()
        optimiser.step()

    return predictor.eval()


def sample_path_indices(
    successors: Tensor, batch: int, steps: int, generator: torch.Generator
) -> Tensor:
    """``(batch, steps + 1)`` state indices of real paths through the chain."""
    indices = [torch.randint(0, successors.shape[0], (batch,), generator=generator)]
    for _ in range(steps):
        choice = torch.randint(0, 2, (batch,), generator=generator)
        indices.append(successors[indices[-1], choice])
    return torch.stack(indices, dim=1)


def nearest_state(latents: Tensor, states: Tensor) -> tuple[Tensor, Tensor]:
    """``(distance, index)`` of each latent's nearest state, shape ``(...)``."""
    flat = latents.reshape(-1, latents.shape[-1])
    nearest = pairwise_distance(flat, states).min(dim=-1)
    shape = latents.shape[:-1]
    return nearest.values.reshape(shape), nearest.indices.reshape(shape)


def manifold_distance(latents: Tensor, states: Tensor) -> Tensor:
    """**Manifold distance**: nearest-state distance, ``(...)``."""
    return nearest_state(latents, states)[0]


def path_validity(latents: Tensor, start: Tensor, world: World) -> Tensor:
    """**Path validity** per depth: fraction of rollouts real up to that depth.

    ``latents`` is ``(B, S, D)`` rollout latents from the start states
    ``start`` ``(B,)``.  Step ``i`` is real when it lies on a state (distance
    below :data:`ON_STATE_DISTANCE`) and that state is a successor of step
    ``i - 1``'s state.  Depth ``i`` counts a rollout when steps ``1..i`` are
    all real.  Returns ``(S,)``.
    """
    distance, index = nearest_state(latents, world.states)
    previous = torch.cat([start.unsqueeze(1), index[:, :-1]], dim=1)
    is_successor = (world.successors[previous] == index.unsqueeze(-1)).any(-1)
    real = is_successor & (distance < ON_STATE_DISTANCE)
    return real.long().cumprod(dim=1).float().mean(dim=0)


def branch_entropy(latents: Tensor, world: World) -> float:
    """**Branch entropy** in bits: how evenly commits split between successors.

    ``latents`` is ``(N, K, D)``: ``K`` depth-1 commits from each state, in
    state order.  A commit counts for a successor when it lies on it
    (distance below :data:`ON_STATE_DISTANCE`); commits that land elsewhere
    are left out (depth-1 :func:`path_validity` reports how many).  The
    entropy of the split between the two successors is averaged over states:
    an even split is 1 bit, always taking one successor is 0.
    """
    distance, index = nearest_state(latents, world.states)
    on_state = distance < ON_STATE_DISTANCE
    hits = (index.unsqueeze(-1) == world.successors.unsqueeze(1)) & on_state.unsqueeze(-1)
    counts = hits.sum(dim=1).float()  # (N, 2)
    p = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    bits = -(p * p.clamp_min(1e-12).log2()).sum(dim=-1)
    return bits.mean().item()


def mode_coverage(candidates: Tensor, world: World) -> float:
    """**Mode coverage**: the worst successor's distance to its nearest candidate.

    ``candidates`` is ``(N, P, D)``, the predictor's candidates from each
    state.  Small means every successor mode has a candidate next to it.
    """
    successors = world.states[world.successors]  # (N, 2, D)
    distances = pairwise_distance(successors, candidates)  # (N, 2, P)
    return distances.min(dim=-1).values.max().item()
