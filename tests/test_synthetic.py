"""The synthetic-world metrics, on a hand-built chain with known answers.

Four orthogonal states; ``0 -> {1, 2}``, ``1 -> {2, 3}``, ``2 -> {3, 0}``,
``3 -> {0, 1}``.
"""

import math

import pytest
import torch

from jepa.evaluation.synthetic import (
    World,
    branch_entropy,
    mode_coverage,
    path_validity,
)

STATES = torch.eye(4) * 2.0
WORLD = World(STATES, torch.tensor([[1, 2], [2, 3], [3, 0], [0, 1]]))


def latents(*rows):
    """Rollout latents from state indices; a tuple blends states equally."""
    out = []
    for row in rows:
        indices = row if isinstance(row, tuple) else (row,)
        out.append(STATES[list(indices)].mean(dim=0))
    return torch.stack(out)


def test_path_validity_is_the_fraction_of_rollouts_real_up_to_each_depth():
    rollouts = torch.stack(
        [
            latents(1, 3, 0),  # 0 -> 1 -> 3 -> 0: valid throughout
            latents(3, 0, 1),  # 0 -> 3 is not a transition
            latents(2, (3, 0), 3),  # step 2 is a blur off every state
            latents(1, 2, 1),  # 2 -> 1 is not a transition
        ]
    )
    start = torch.tensor([0, 0, 0, 0])

    validity = path_validity(rollouts, start, WORLD)

    torch.testing.assert_close(validity, torch.tensor([0.75, 0.5, 0.25]))


def test_branch_entropy_is_the_mean_bits_of_each_states_split_between_successors():
    commits = torch.stack(
        [
            latents(1, 2, 1, 2, 3),  # even split; the 3 is no successor of 0
            latents(2, 2, 2, 2, (2, 3)),  # one successor (and a blur): 0 bits
            latents(3, 0, 3, 0, 3),  # 3:2 split
            latents(0, 1, 0, 1, 0),  # 3:2 split
        ]
    )

    three_two = -(0.6 * math.log2(0.6) + 0.4 * math.log2(0.4))
    expected = (1.0 + 0.0 + 2 * three_two) / 4
    assert branch_entropy(commits, WORLD) == pytest.approx(expected)


def test_mode_coverage_is_the_worst_successors_distance_to_a_candidate():
    candidates = torch.stack(
        [
            latents(1, 2),
            latents(2, 3),
            latents(3, 0),
            latents(0, (1, 2)),  # successor 1 is 45 degrees off a candidate
        ]
    )

    coverage = mode_coverage(candidates, WORLD)

    assert coverage == pytest.approx(2.0 - math.sqrt(2.0), abs=1e-6)
