"""The research claim, on a synthetic Markov chain.

The paper's idea is that committing to one **candidate** keeps each latent, and
every chained **rollout** latent, a clear single future, while the **μ (mean
future)** path blurs the possible futures into their average.

The world and the training loop live in :mod:`jepa.evaluation.synthetic`,
shared with ``docs/report/synthetic.py``
(which reports the same world's numbers over seeds).  Four claims are
measured:

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

import pytest
import torch
import torch.nn.functional as F

from jepa.evaluation.synthetic import (
    BATCH_SIZE,
    DIM,
    N_STATES,
    build_world,
    manifold_distance,
    mode_coverage,
    sample_path_indices,
    train_predictor,
)
from jepa.evaluation.measure import mu_path
from jepa.models import Hindsight, PiSample, distance, rollout

PREDICTION_COUNT = 4
HORIZON = 3
SEED = 0


@pytest.fixture(scope="module")
def trained():
    world = build_world(SEED)
    return train_predictor(world, PREDICTION_COUNT, SEED), world


def test_every_successor_has_a_candidate_next_to_it(trained):
    """Candidates land on the modes, not between them."""
    predictor, world = trained
    with torch.no_grad():
        worst = mode_coverage(predictor(world.states).candidates, world)
    # Measured max over states/modes: ~0.003.
    assert worst < 0.05, f"a successor mode has no candidate nearby (d={worst})"


def test_mu_is_the_blurry_average_of_the_successors(trained):
    """μ points at the average of the two unit successor vectors, with norm < 1."""
    predictor, (states, successors) = trained
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


def test_committed_rollout_stays_on_manifold_while_mu_path_blurs(trained):
    """A **Rollout** of committed futures stays a real latent; μ blurs off."""
    predictor, (states, _) = trained
    generator = torch.Generator().manual_seed(SEED + 123)

    with torch.no_grad():
        committed = rollout(
            predictor, states, PiSample(generator=generator), HORIZON
        )
        assert committed.future.shape == (N_STATES, HORIZON, DIM)

        committed_distance = manifold_distance(committed.future, states).mean(0)
        mu_distance = manifold_distance(
            mu_path(predictor, states, HORIZON), states
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


def test_hindsight_rollout_has_low_error_while_mu_path_drifts(trained):
    """**Hindsight** keeps **rollout error** near zero; the μ rollout drifts."""
    predictor, (states, successors) = trained
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
        error = distance(hindsight.future, path[:, 1:]).mean(0)
        mu_error = distance(
            mu_path(predictor, path[:, 0], HORIZON), path[:, 1:]
        ).mean(0)

    # Measured hindsight error: ~[0.0015, 0.0020, 0.0023]; μ error:
    # ~[0.61, 0.99, 1.27].
    assert (error < 0.05).all(), error
    for depth in range(HORIZON):
        assert mu_error[depth] > 0.2, (depth, mu_error)
        assert mu_error[depth] > 3.0 * error[depth], (depth, error, mu_error)
