"""The research claim, on a synthetic Markov chain.

The paper's idea is that committing to one **candidate** keeps each latent, and
every chained **rollout** latent, a clear single future, while the **μ (mean
future)** baseline blurs the possible futures into their average.

World
-----
``N`` states live at the encoder-latent scale (unit-normalised vectors
scaled to ``sqrt(D)``, so their per-dim RMS is 1).  Each state has exactly two
distinct successors, chosen once at random.  A transition picks one of the two
with probability 1/2.  There is no encoder: the state latents are the real
latents.

Only a tiny :class:`~jepa.models.predictor.PredictorModel` is trained, with
:class:`~jepa.models.losses.CommitmentLoss` (no VISReg), on (state, sampled
successor) pairs.  Four claims are then measured:

a) every successor mode has a **candidate** next to it,
b) **μ** is the blurry average of the two unit-normalised successors,
c) a committed **rollout** stays on the **manifold** of real latents while a μ
   rollout blurs off it,
d) a **hindsight** rollout has near-zero **rollout error**, unlike the μ
   rollout.

The training setup is fixed and deterministic (seed 0).  The thresholds were
chosen after checking seeds 0-3; see the task notes in the commit message /
report.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from jepa import PredictorConfig
from jepa.models import (
    CommitmentLoss,
    Hindsight,
    PiSample,
    rollout,
)
from jepa.models.predictor import PredictorModel

N_STATES = 8
DIM = 32
PREDICTION_COUNT = 4
HORIZON = 3
BATCH_SIZE = 128
TRAIN_STEPS = 400
LEARNING_RATE = 3e-3
SEED = 0


def build_world(seed: int):
    """Return fixed state latents ``(N, D)`` and two successors per state."""
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(N_STATES, DIM, generator=generator)
    # Norm sqrt(D) <=> per-dim RMS 1, the predictor's output scale.
    states = F.normalize(states, dim=-1) * math.sqrt(DIM)

    successors = torch.empty(N_STATES, 2, dtype=torch.long)
    for state in range(N_STATES):
        order = torch.randperm(N_STATES, generator=generator).tolist()
        choices = [other for other in order if other != state][:2]
        successors[state] = torch.tensor(choices)
    return states, successors


def train_predictor(states, successors, seed: int = SEED) -> PredictorModel:
    """Fit a tiny predictor on (state, sampled successor) pairs."""
    torch.manual_seed(seed)
    predictor = PredictorModel(
        PredictorConfig(
            target_dim=DIM,
            hidden_size=DIM,
            num_attention_heads=4,
            num_hidden_layers=1,
            prediction_count=PREDICTION_COUNT,
        )
    )
    criterion = CommitmentLoss()
    optimiser = torch.optim.Adam(predictor.parameters(), lr=LEARNING_RATE)
    generator = torch.Generator().manual_seed(seed + 1)

    for step in range(TRAIN_STEPS):
        state = torch.randint(0, N_STATES, (BATCH_SIZE,), generator=generator)
        pick = torch.randint(0, 2, (BATCH_SIZE,), generator=generator)
        target = states[successors[state, pick]]

        # CommitmentLoss consumes one extra sequence axis.
        futures = predictor(states[state].unsqueeze(1))
        progress = torch.tensor(step / TRAIN_STEPS)
        output = criterion(futures, target.unsqueeze(1), progress=progress)

        optimiser.zero_grad()
        output.loss.backward()
        optimiser.step()

    return predictor.eval()


@pytest.fixture(scope="module")
def trained():
    states, successors = build_world(SEED)
    return train_predictor(states, successors), states, successors


def cosine_distance(a, b):
    """``2 - 2 cos``, the distance used by :meth:`Futures.distances`."""
    dot = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)
    return (2.0 - 2.0 * dot).clamp_min(0.0)


def manifold_distance(latents, bank):
    """**Manifold distance**: nearest-neighbour distance in the latent bank."""
    flat = latents.reshape(-1, latents.shape[-1])
    distances = cosine_distance(flat[:, None, :], bank[None, :, :])
    nearest = distances.min(dim=-1).values
    return nearest.reshape(*latents.shape[:-1])


def mu_rollout(predictor, z, steps: int):
    """The μ baseline: feed back ``normalize(μ)`` at the latent scale."""
    latents = []
    for _ in range(steps):
        mu = predictor(z).mu
        z = F.normalize(mu, dim=-1) * math.sqrt(z.shape[-1])
        latents.append(z)
    return torch.stack(latents, dim=-2)


def sample_path_indices(successors, batch: int, steps: int, generator):
    """Sample ``batch`` real ``steps``-step paths through the chain."""
    indices = [torch.randint(0, N_STATES, (batch,), generator=generator)]
    for _ in range(steps):
        pick = torch.randint(0, 2, (batch,), generator=generator)
        indices.append(successors[indices[-1], pick])
    return torch.stack(indices, dim=1)


def test_every_successor_has_a_candidate_next_to_it(trained):
    """Candidates land on the modes, not between them."""
    predictor, states, successors = trained
    worst = 0.0
    with torch.no_grad():
        for state in range(N_STATES):
            futures = predictor(states[state])
            for successor in successors[state]:
                distance = futures.distances(states[successor]).min().item()
                worst = max(worst, distance)
    # Measured max over states/modes: ~0.003.
    assert worst < 0.05, f"a successor mode has no candidate nearby (d={worst})"


def test_mu_is_the_blurry_average_of_the_successors(trained):
    """μ points at the average of the two unit successor vectors, with norm < 1."""
    predictor, states, successors = trained
    with torch.no_grad():
        mu = predictor(states).mu

    successor_dirs = F.normalize(states[successors], dim=-1)  # (N, 2, D)
    average_dir = F.normalize(successor_dirs.sum(dim=1), dim=-1)
    cosine = (F.normalize(mu, dim=-1) * average_dir).sum(-1)
    norm = mu.norm(dim=-1)

    # Measured: cosine >= 0.978, norm in [0.58, 0.81], mean ~0.69
    # (two near-orthogonal successors would give 1/sqrt(2) ~= 0.707).
    assert cosine.min() > 0.9, f"μ is not the mean vector: min cos {cosine.min()}"
    assert norm.max() < 0.9, f"μ did not blur: max norm {norm.max()}"
    assert norm.min() > 0.5, f"μ collapsed toward zero: min norm {norm.min()}"


def test_committed_rollout_stays_on_manifold_while_mu_rollout_blurs(trained):
    """A **Rollout** of committed futures stays a real latent; μ blurs off."""
    predictor, states, _ = trained
    generator = torch.Generator().manual_seed(SEED + 123)

    with torch.no_grad():
        committed = rollout(
            predictor, states, PiSample(generator=generator), HORIZON
        )
        assert committed.future.shape == (N_STATES, HORIZON, DIM)

        committed_distance = manifold_distance(committed.future, states).mean(0)
        mu_distance = manifold_distance(
            mu_rollout(predictor, states, HORIZON), states
        ).mean(0)

    # Measured committed: ~[0.002, 0.003, 0.003]; μ: ~[0.52, 0.62, 0.63].
    assert (committed_distance < 0.1).all(), committed_distance
    for depth in range(HORIZON):
        assert mu_distance[depth] > 0.2, (depth, mu_distance)
        assert mu_distance[depth] > 3.0 * committed_distance[depth], (
            depth,
            committed_distance,
            mu_distance,
        )


def test_hindsight_rollout_has_low_error_while_mu_rollout_drifts(trained):
    """**Hindsight** keeps **rollout error** near zero; the μ rollout drifts."""
    predictor, states, successors = trained
    generator = torch.Generator().manual_seed(SEED + 321)
    path_indices = sample_path_indices(
        successors, BATCH_SIZE, HORIZON, generator
    )
    path = states[path_indices]  # (B, HORIZON + 1, D)

    with torch.no_grad():
        hindsight = rollout(
            predictor,
            path[:, 0],
            Hindsight(),
            HORIZON,
            targets=path[:, 1:],
        )
        error = cosine_distance(hindsight.future, path[:, 1:]).mean(0)
        mu_error = cosine_distance(
            mu_rollout(predictor, path[:, 0], HORIZON), path[:, 1:]
        ).mean(0)

    # Measured hindsight error: ~[0.0015, 0.0020, 0.0023]; μ error:
    # ~[0.61, 0.99, 1.27].
    assert (error < 0.05).all(), error
    for depth in range(HORIZON):
        assert mu_error[depth] > 0.2, (depth, mu_error)
        assert mu_error[depth] > 3.0 * error[depth], (depth, error, mu_error)
