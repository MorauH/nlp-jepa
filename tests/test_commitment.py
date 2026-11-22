"""Tests for :mod:`jepa.models.commitment` (Futures, policies and commit).

Everything goes through the public interface: build a ``Futures`` and commit.
"""

import pytest
import torch
import torch.nn.functional as F

from jepa.models import (
    Futures,
    Hindsight,
    MuSafe,
    PiArgmax,
    PiSample,
    policy,
    rollout,
)
from jepa.models.predictor import PredictorModel
from jepa import PredictorConfig


def pick(candidates, index):
    """``candidates[index]`` for any number of leading dims, written out."""
    flat = candidates.reshape(-1, *candidates.shape[-2:])
    rows = torch.arange(flat.shape[0])
    return flat[rows, index.reshape(-1)].reshape(*index.shape, candidates.shape[-1])


def make_futures(*leading, count=4, dim=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return Futures(
        candidates=torch.randn(*leading, count, dim, generator=generator),
        pi=torch.randn(*leading, count, generator=generator),
        mu=torch.randn(*leading, dim, generator=generator),
    )


# ---------------------------------------------------------------------------
# The invariant: a committed future is one of the candidates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("leading", [(), (3,), (2, 5)])
def test_commit_returns_the_indexed_candidate(leading):
    futures = make_futures(*leading)
    commit = futures.commit(PiArgmax())

    assert commit.index.shape == leading
    assert torch.equal(commit.index, futures.pi.argmax(dim=-1))
    assert torch.equal(commit.future, pick(futures.candidates, commit.index))
    assert commit.future.dtype == futures.candidates.dtype


def test_commit_works_for_a_single_future():
    """No leading dims at all: candidates (P, D) -> future (D,)."""
    futures = make_futures()
    commit = futures.commit(Hindsight(), futures.candidates[..., 0, :])

    assert commit.index.shape == ()
    assert commit.future.shape == futures.candidates.shape[-1:]
    assert torch.equal(commit.future, futures.candidates[commit.index])


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def test_hindsight_picks_the_closest_candidate():
    futures = make_futures(2, 5)
    target = torch.randn(2, 5, 6)
    commit = futures.commit(Hindsight(), target)

    expected = futures.distances(target).argmin(dim=-1)
    assert torch.equal(commit.index, expected)
    assert torch.equal(commit.future, pick(futures.candidates, commit.index))


def test_pi_argmax_picks_the_highest_pi():
    futures = make_futures(3)
    commit = futures.commit(PiArgmax())

    assert torch.equal(commit.index, futures.pi.argmax(dim=-1))


def test_mu_safe_picks_the_lowest_expected_distance():
    futures = make_futures(2, 5)
    candidates = F.normalize(futures.candidates.float(), dim=-1)
    mu = futures.mu.float()
    # Brute force over candidates: 2 - 2 ĉ_i·μ, one candidate at a time.
    reference = torch.stack(
        [2.0 - 2.0 * (candidates[..., i, :] * mu).sum(-1) for i in range(4)],
        dim=-1,
    )

    torch.testing.assert_close(futures.expected_distances(), reference)
    assert torch.equal(
        futures.commit(MuSafe()).index, reference.argmin(dim=-1)
    )


def test_pi_sample_nonpositive_temperature_is_argmax():
    futures = make_futures(3)
    for temperature in (0.0, -1.0):
        commit = futures.commit(PiSample(temperature=temperature))
        assert torch.equal(commit.index, futures.pi.argmax(dim=-1))


def test_pi_sample_top_k_one_is_argmax():
    futures = make_futures(3)
    commit = futures.commit(PiSample(temperature=1.0, top_k=1))
    assert torch.equal(commit.index, futures.pi.argmax(dim=-1))


def test_pi_sample_top_k_never_leaves_the_top_k():
    futures = make_futures(*((50,)), seed=1)
    top2 = futures.pi.topk(2, dim=-1).indices
    commit = futures.commit(PiSample(temperature=2.0, top_k=2))
    assert torch.equal(
        (commit.index[:, None] == top2).any(dim=-1),
        torch.ones(50, dtype=torch.bool),
    )


def test_pi_sample_is_reproducible_with_a_generator():
    futures = make_futures(16, seed=2)
    first = futures.commit(
        PiSample(temperature=1.0, generator=torch.Generator().manual_seed(1234))
    ).index
    second = futures.commit(
        PiSample(temperature=1.0, generator=torch.Generator().manual_seed(1234))
    ).index
    assert torch.equal(first, second)


def test_pi_sample_frequencies_follow_pi():
    """Gumbel-max samples from softmax(pi) over many independent draws."""
    logits = torch.tensor([1.0, 0.5, 0.0, -1.0])
    draws = 8000
    futures = Futures(
        candidates=torch.randn(draws, 4, 4),
        pi=logits.expand(draws, -1),
        mu=torch.randn(draws, 4),
    )
    commit = futures.commit(
        PiSample(generator=torch.Generator().manual_seed(7))
    )
    frequencies = torch.bincount(commit.index, minlength=4).float() / draws
    torch.testing.assert_close(
        frequencies, torch.softmax(logits, dim=-1), atol=0.03, rtol=0
    )


# ---------------------------------------------------------------------------
# Rollout: chaining commits without re-encoding
# ---------------------------------------------------------------------------


def tiny_predictor(dim=8, count=4):
    torch.manual_seed(0)
    return PredictorModel(
        PredictorConfig(
            target_dim=dim,
            hidden_size=dim,
            num_attention_heads=2,
            num_hidden_layers=1,
            prediction_count=count,
        )
    ).eval()


@pytest.mark.parametrize("leading", [(), (3,)])
@pytest.mark.parametrize("steps", [1, 3])
def test_rollout_has_one_committed_future_per_depth(leading, steps):
    predictor = tiny_predictor()
    z = torch.randn(*leading, 8)

    with torch.no_grad():
        result = rollout(predictor, z, PiArgmax(), steps)

    assert result.future.shape == (*leading, steps, 8)
    assert result.index.shape == (*leading, steps)

    # Depth i commits from the previous depth's committed future.
    previous = z
    for depth in range(steps):
        commit = predictor(previous).commit(PiArgmax())
        assert torch.equal(result.index[..., depth], commit.index)
        assert torch.equal(result.future[..., depth, :], commit.future)
        previous = result.future[..., depth, :]


def test_rollout_hindsight_matches_committing_by_hand():
    predictor = tiny_predictor()
    z = torch.randn(2, 8)
    targets = torch.randn(2, 3, 8)

    with torch.no_grad():
        result = rollout(predictor, z, Hindsight(), 3, targets=targets)

        previous = z
        for depth in range(3):
            commit = predictor(previous).commit(
                Hindsight(), targets[..., depth, :]
            )
            assert torch.equal(result.index[..., depth], commit.index)
            assert torch.equal(
                result.future[..., depth, :], commit.future
            )
            previous = result.future[..., depth, :]


def test_rollout_rejects_bad_steps_and_missing_hindsight_targets():
    predictor = tiny_predictor()
    z = torch.randn(2, 8)

    with pytest.raises(ValueError):
        rollout(predictor, z, PiArgmax(), 0)
    with pytest.raises(ValueError):
        rollout(predictor, z, Hindsight(), 3)
    with pytest.raises(ValueError):
        rollout(predictor, z, Hindsight(), 3, targets=torch.randn(2, 2, 8))


def test_rollout_leaves_gradients_attached():
    predictor = tiny_predictor()
    z = torch.randn(2, 8, requires_grad=True)

    result = rollout(predictor, z, PiArgmax(), 2)
    result.future.sum().backward()

    assert z.grad is not None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_hindsight_requires_a_target():
    futures = make_futures()
    with pytest.raises(ValueError):
        futures.commit(Hindsight())


def test_wrong_target_shape_raises():
    futures = make_futures(2)
    with pytest.raises(ValueError):
        futures.commit(Hindsight(), torch.randn(6))  # mu is (2, 6)
    with pytest.raises(ValueError):
        futures.commit(PiArgmax(), torch.randn(6))


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        policy("nope")


def test_decoder_train_policies_are_registered():
    from jepa import DecoderConfig
    from jepa.models.commitment import POLICIES

    assert set(DecoderConfig.TRAIN_POLICIES) <= set(POLICIES)
    for name in DecoderConfig.TRAIN_POLICIES:
        assert policy(name).needs_target == (name == "hindsight")


# ---------------------------------------------------------------------------
# Distances: fp32 under autocast, reproducible under bf16
# ---------------------------------------------------------------------------


def test_distances_are_fp32_under_autocast():
    torch.manual_seed(0)
    candidates = torch.randn(2, 3, 4, 8, dtype=torch.bfloat16)
    target = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    futures = Futures(
        candidates,
        torch.randn(2, 3, 4, dtype=torch.bfloat16),
        torch.randn(2, 3, 8, dtype=torch.bfloat16),
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        distances = futures.distances(target)
        commit = futures.commit(Hindsight(), target)

    assert distances.dtype == torch.float32
    reference = (
        2.0
        - 2.0
        * (
            F.normalize(candidates.float(), dim=-1)
            @ F.normalize(target.float(), dim=-1).unsqueeze(-1)
        ).squeeze(-1)
    ).clamp_min(0.0)
    torch.testing.assert_close(distances, reference)
    # The same fp32 distance decides the commit, so bf16 autocast cannot make
    # hindsight disagree with itself.
    assert torch.equal(commit.index, reference.argmin(dim=-1))


def test_distances_are_non_negative_for_identical_vectors():
    futures = make_futures(2)
    target = futures.candidates[..., 2, :]
    distances = futures.distances(target)
    assert (distances >= 0).all()
    assert torch.equal(distances.argmin(dim=-1), torch.full((2,), 2))


def test_distance_pairs_rows_and_pairwise_distance_crosses_them():
    from jepa.models import distance, pairwise_distance

    def reference(a, b):
        cos = (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)
        return (2.0 - 2.0 * cos).clamp_min(0.0)

    a, b = torch.randn(4, 3), torch.randn(4, 3)
    torch.testing.assert_close(distance(a, b), reference(a, b))
    assert distance(torch.randn(3), torch.randn(3)).shape == ()

    # Same number of rows on both sides must still give every pair, not the
    # paired distances: this is the manifold-distance case.
    points, bank = torch.randn(4, 3), torch.randn(4, 3)
    crossed = pairwise_distance(points, bank)
    assert crossed.shape == (4, 4)
    torch.testing.assert_close(
        crossed, reference(points[:, None, :], bank[None, :, :]), atol=1e-5, rtol=1e-5
    )

    # Candidates against a target: (..., P) from a matmul.
    futures = make_futures(2, 5)
    target = torch.randn(2, 5, futures.candidates.shape[-1])
    torch.testing.assert_close(
        futures.distances(target),
        reference(futures.candidates, target[..., None, :]),
        atol=1e-5,
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_gradient_flows_from_commit_to_candidates():
    candidates = torch.randn(2, 4, 8, requires_grad=True)
    futures = Futures(
        candidates=candidates,
        pi=torch.randn(2, 4),
        mu=torch.randn(2, 8),
    )
    commit = futures.commit(PiArgmax())
    commit.future.sum().backward()

    assert candidates.grad is not None
    expected = torch.zeros_like(candidates)
    expected.scatter_(
        -2,
        commit.index[..., None, None].expand(-1, 1, candidates.shape[-1]),
        1.0,
    )
    assert torch.equal(candidates.grad, expected)
