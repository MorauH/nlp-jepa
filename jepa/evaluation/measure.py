"""The paper's numbers: representation, commitment, rollouts and decoding.

Run from the repository root::

    python -m jepa measure --model runs/jepa
    python -m jepa measure --model runs/jepa --json runs/jepa/metrics.json

``measure`` replaces the old ``verify`` command.  :func:`collect_metrics`
returns one JSON-able dict (nested by section), :func:`print_report` prints it,
and ``--json`` writes the same dict, so two runs can be diffed without copying
numbers by hand.

Data
----
Queries are validation-split chunks, evaluated in mini-batches and averaged
with token weights.  ``prepare-data`` holds out whole documents for validation,
so the train-split latents used for the **manifold bank** come from different
documents than the queries.  That bank is a sample of real encoder latents:
manifold distance is the distance from a latent to its nearest real latent,
and the ``real`` rollout row calibrates what "on the manifold" means.

Old checkpoints (from before the predictor became a pure function of ``z_t``)
do not load with this code; run the old commit's code in a worktree to compare
against them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data.tokenizer import load_tokenizer
from ..models.commitment import Hindsight, MuSafe, PiArgmax, PiSample, rollout
from ..models.jepa import JepaModel, build_block_hints
from ..paths import DEFAULT_RUN_DIR, FINEWEB_DIR, TOKENIZER_DIR
from ..runtime import select_device
from ..training.data import tokenize_batch
from ..training.train import load_datasets

DEFAULT_QUERY_EXAMPLES = 64
DEFAULT_MAX_LENGTH = 1024
DEFAULT_BATCH_SIZE = 4
DEFAULT_BANK_SIZE = 50_000
DEFAULT_ROLLOUT_POSITIONS = 2_048
DEFAULT_DEPTHS = 4
DEFAULT_LOSS_CHUNK_SIZE = 256

# Effective rank is sample-size sensitive; more latents is more rank.
REPRESENTATION_LATENTS = 20_000

# The four measured rollouts (the μ path is a baseline, not a policy).
ROLLOUT_PATHS = ("hindsight", "pi_argmax", "pi_sample", "mu")
# Decoder hint conditions, in report order.
DECODER_CONDITIONS = (
    "pi_argmax",
    "pi_sample",
    "mu_safe",
    "hindsight",
    "real",
    "shuffled",
    "null",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _value(value) -> float:
    """A tensor or number as a plain float (JSON-able)."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def _generator(device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _cosine_distance(a, b):
    """``2 - 2 cos`` for non-candidate vectors (paths, μ, copies).

    Candidate-to-target distances always go through
    :meth:`jepa.models.commitment.Futures.distances`; this is the same metric
    for the vectors that are not candidates.
    """
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return (2.0 - 2.0 * (a * b).sum(-1)).clamp_min(0.0)


def _prediction_mask(attention_mask, horizon: int):
    """Valid positions: a real token with a real token ``horizon`` ahead."""
    return attention_mask[:, :-horizon].bool() & attention_mask[:, horizon:].bool()


def _picked_distance(distances, futures, policy, valid):
    """True candidate-target distance of the candidate ``policy`` commits to."""
    index = futures.commit(policy).index[valid]
    return distances.gather(-1, index[:, None]).squeeze(-1)


def _walk(latents, masks, horizon: int):
    """Yield ``(z_t, z_{t+k}, valid)`` per batch, all on the model device."""
    for z, mask in zip(latents, masks):
        valid = _prediction_mask(mask, horizon)
        yield z[:, :-horizon, :], z[:, horizon:, :], valid


def _mu_path(predictor, z, steps: int):
    """The μ baseline: feed back ``normalize(μ)`` at the latent scale.

    This is the measurement baseline from the research-claim test: it never
    commits to a candidate, so it shows what a single mean prediction does
    over a rollout.
    """
    path = []
    for _ in range(steps):
        mu = predictor(z).mu
        # Back to the predictor's dtype so a bf16/fp16 checkpoint can consume
        # the path; the normalisation itself is fp32.
        z = F.normalize(mu.float(), dim=-1).to(mu.dtype) * math.sqrt(
            z.shape[-1]
        )
        path.append(z)
    return torch.stack(path, dim=-2)


def _manifold_scores(points, bank, chunk_size: int | None):
    """Nearest-neighbour distance of each row to the real-latent bank.

    ``min_b 2 - 2 cos(x, b)``.  Points are processed in chunks so the
    ``(chunk, bank)`` similarity matrix stays bounded (the default budget is
    about 16 MB).
    """
    points = F.normalize(points.float(), dim=-1)
    if chunk_size is None:
        chunk_size = max(1, 4_000_000 // bank.shape[0])
    scores = []
    for start in range(0, points.shape[0], chunk_size):
        similarity = points[start : start + chunk_size] @ bank.T
        scores.append(
            (2.0 - 2.0 * similarity.max(dim=-1).values).clamp_min(0.0)
        )
    return torch.cat(scores)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_query_batches(
    dataset,
    tokenizer,
    *,
    num_examples: int = DEFAULT_QUERY_EXAMPLES,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
):
    """Sample validation chunks as padded ``(labels, attention_mask)`` batches."""
    if num_examples < 1:
        raise ValueError("num_examples must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if len(dataset) < 1:
        raise ValueError("the validation split is empty")

    generator = torch.Generator().manual_seed(seed)
    count = min(num_examples, len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    examples = [
        {"input_ids": list(dataset[index]["input_ids"])[:max_length]}
        for index in indices
    ]
    return [
        tokenize_batch(examples[start : start + batch_size], tokenizer)
        for start in range(0, count, batch_size)
    ]


@torch.no_grad()
def build_bank(
    model,
    dataset,
    tokenizer,
    *,
    size: int = DEFAULT_BANK_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 0,
) -> torch.Tensor:
    """Sample real encoder latents from the train split for the manifold bank.

    Chunks are visited in a seeded random order and every valid position in
    them is a candidate latent; the final sample is trimmed to ``size``.  The
    result is fp32 on the CPU (raw latents; ``collect_metrics`` normalises).
    """
    if size < 1:
        raise ValueError("bank size must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if len(dataset) < 1:
        raise ValueError("the train split is empty")

    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()

    chunks: list[torch.Tensor] = []
    collected = 0
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        examples = [
            {"input_ids": list(dataset[index]["input_ids"])[:max_length]}
            for index in indices
        ]
        labels, attention_mask = tokenize_batch(examples, tokenizer)
        latents = model.encoder(
            labels.to(device), attention_mask=attention_mask.to(device)
        )
        valid = latents[attention_mask.bool().to(device)]
        chunks.append(valid.float().cpu())
        collected += valid.shape[0]
        if collected >= size:
            break

    bank = torch.cat(chunks, dim=0)
    if bank.shape[0] > size:
        picked = torch.randperm(bank.shape[0], generator=generator)[:size]
        bank = bank[picked]
    return bank


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _representation_section(model, batches, latents) -> dict:
    """Section 1: no-collapse checks on the encoder latents."""
    horizon = model.pred_horizon
    pooled: list[torch.Tensor] = []
    pooled_count = 0
    temporal_sum = 0.0
    temporal_count = 0
    token_count = 0
    visreg_sums = {name: 0.0 for name in model.visreg.weights}

    for (_, attention_mask), z in zip(batches, latents):
        valid = attention_mask.bool()
        tokens = int(valid.sum())
        token_count += tokens

        # Effective rank needs many latents, but not all of them.
        if pooled_count < REPRESENTATION_LATENTS:
            take = z[valid].float()
            take = take[: REPRESENTATION_LATENTS - pooled_count]
            pooled.append(take)
            pooled_count += take.shape[0]

        visreg_out = model.visreg(z, mask=attention_mask)
        for name, value in visreg_out.terms.items():
            visreg_sums[name] += _value(value) * tokens

        z_t = z[:, :-horizon, :]
        future = z[:, horizon:, :]
        valid_pairs = _prediction_mask(attention_mask, horizon)
        if valid_pairs.any():
            cos = F.cosine_similarity(
                z_t[valid_pairs].float(), future[valid_pairs].float(), dim=-1
            )
            temporal_sum += cos.sum().item()
            temporal_count += cos.numel()

    z_all = torch.cat(pooled, dim=0)
    mean = z_all.mean(dim=0)
    centered = z_all - mean
    std = centered.std(dim=0, unbiased=False)
    covariance = centered.T @ centered / max(1, z_all.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum().clamp_min(1e-12)
    shares = eigenvalues / total
    effective_rank = torch.exp(
        -(shares * shares.clamp_min(1e-12).log()).sum()
    )

    return {
        "per_dim_std": _value(std.mean()),
        "mean_vector_norm": _value(mean.norm()),
        "effective_rank": _value(effective_rank),
        "latents": int(z_all.shape[0]),
        "temporal_cos": temporal_sum / max(1, temporal_count),
        "visreg": {
            name: visreg_sums[name] / max(1, token_count)
            for name in visreg_sums
        },
    }


def _loss_section(model, batches) -> dict:
    """Section 2: every JEPA term, its weight, and its contribution."""
    horizon = model.pred_horizon
    weights = {**model.visreg.weights, **model.commitment_loss.weights}
    term_sums: dict[str, float] = {}
    diagnostic_sums: dict[str, float] = {}
    count = 0

    for labels, attention_mask in batches:
        valid = int(_prediction_mask(attention_mask, horizon).sum())
        if valid == 0:
            continue
        out = model.jepa_loss(
            labels, attention_mask=attention_mask, progress=None
        )
        for name, value in out.terms.items():
            term_sums[name] = term_sums.get(name, 0.0) + _value(value) * valid
        for name, value in out.diagnostics.items():
            diagnostic_sums[name] = (
                diagnostic_sums.get(name, 0.0) + _value(value) * valid
            )
        count += valid

    if count == 0:
        raise ValueError("no valid prediction positions in the query batches")

    terms = {}
    for name, total in term_sums.items():
        value = total / count
        weight = float(weights[name])
        terms[name] = {
            "weight": weight,
            "value": value,
            "weighted": weight * value,
        }
    total = sum(term["weighted"] for term in terms.values())

    entropy = diagnostic_sums.get("assignment_entropy", 0.0) / count
    return {
        "terms": terms,
        "total": total,
        "floor": {
            "assignment_entropy": entropy,
            "kl_to_pi": terms["pi"]["value"] - entropy,
        },
        "diagnostics": {
            name: value / count for name, value in diagnostic_sums.items()
        },
    }


def _heads_section(model, batches, latents) -> dict:
    """Section 3: candidates, π, μ and their calibration."""
    horizon = model.pred_horizon
    masks = [mask for _, mask in batches]
    prediction_count = model.config.predictor.prediction_count

    count = 0
    best_sum = 0.0
    pair_sim_sum = 0.0
    pair_count = 0
    delta_sum = 0.0
    z_norm_sum = 0.0
    win_counts = torch.zeros(prediction_count, dtype=torch.long)
    pi_nll_sum = 0.0
    correct = 0
    mu_norm_sum = 0.0
    mu_dir_sum = 0.0

    for z_t, future, valid in _walk(latents, masks, horizon):
        if not valid.any():
            continue
        futures = model.predictor(z_t)
        distances = futures.distances(future)[valid]  # (N, P) fp32
        future_valid = future[valid].float()
        z_valid = z_t[valid].float()
        n = distances.shape[0]
        count += n

        best_sum += distances.min(dim=-1).values.sum().item()

        if prediction_count > 1:
            dirs = F.normalize(futures.candidates[valid].float(), dim=-1)
            similarity = torch.bmm(dirs, dirs.transpose(1, 2))
            off_diagonal = 1.0 - torch.eye(
                prediction_count,
                device=similarity.device,
                dtype=similarity.dtype,
            )
            pair_sim_sum += (similarity * off_diagonal).sum().item()
            pair_count += n * prediction_count * (prediction_count - 1)

        delta_sum += (
            futures.candidates[valid].float() - z_valid.unsqueeze(-2)
        ).norm(dim=-1).sum().item()
        z_norm_sum += z_valid.norm(dim=-1).sum().item()

        hard_win = distances.argmin(dim=-1)
        win_counts += torch.bincount(
            hard_win, minlength=prediction_count
        ).cpu()
        log_pi = F.log_softmax(futures.pi[valid].float(), dim=-1)
        pi_nll_sum += -log_pi.gather(-1, hard_win[:, None]).sum().item()
        correct += (log_pi.argmax(dim=-1) == hard_win).sum().item()

        mu = futures.mu[valid].float()
        mu_norm_sum += mu.norm(dim=-1).sum().item()
        mu_dir_sum += _cosine_distance(mu, future_valid).sum().item()

    if count == 0:
        raise ValueError("no valid prediction positions in the query batches")

    shares = win_counts.float() / win_counts.sum().clamp_min(1)
    entropy = float(-(shares * shares.clamp_min(1e-12).log()).sum())
    pi_nll = pi_nll_sum / count
    mu_norm = mu_norm_sum / count
    mu_dir = mu_dir_sum / count

    return {
        "best_d": best_sum / count,
        "candidate_pair_cos": (
            pair_sim_sum / pair_count if pair_count else 1.0
        ),
        "delta_ratio": (delta_sum / (count * prediction_count))
        / max(z_norm_sum / count, 1e-12),
        "win_shares": [float(value) for value in shares.tolist()],
        "live_slots": int((shares > 1.0 / (4 * prediction_count)).sum()),
        "effective_slots": math.exp(entropy),
        "pi_nll": pi_nll,
        "constant_nll": entropy,
        "pi_gain": entropy - pi_nll,
        "pi_accuracy": correct / count,
        "majority_accuracy": float(shares.max()),
        "mu_norm": mu_norm,
        "mu_dir_distance_measured": mu_dir,
        "mu_dir_distance_predicted": 2.0 - 2.0 * mu_norm,
        "mu_calibration_error": mu_dir - (2.0 - 2.0 * mu_norm),
    }


def _policies_section(model, batches, latents, *, seed: int) -> dict:
    """Section 4: distance-to-real at depth 1 for every commit rule."""
    horizon = model.pred_horizon
    masks = [mask for _, mask in batches]
    prediction_count = model.config.predictor.prediction_count
    generator = _generator(latents[0].device, seed + 1)

    count = 0
    copy_sum = 0.0
    hindsight_sum = 0.0
    pi_argmax_sum = 0.0
    pi_sample_sum = 0.0
    mu_safe_sum = 0.0
    mu_dir_sum = 0.0
    slot_sums = torch.zeros(prediction_count, dtype=torch.float64)

    for z_t, future, valid in _walk(latents, masks, horizon):
        if not valid.any():
            continue
        futures = model.predictor(z_t)
        distances = futures.distances(future)[valid]  # (N, P) fp32
        future_valid = future[valid].float()
        z_valid = z_t[valid].float()
        n = distances.shape[0]
        count += n

        copy_sum += _cosine_distance(z_valid, future_valid).sum().item()
        hindsight_sum += distances.min(dim=-1).values.sum().item()

        pi_argmax_sum += _picked_distance(
            distances, futures, PiArgmax(), valid
        ).sum().item()
        pi_sample_sum += _picked_distance(
            distances, futures, PiSample(generator=generator), valid
        ).sum().item()
        mu_safe_sum += _picked_distance(
            distances, futures, MuSafe(), valid
        ).sum().item()
        mu_dir_sum += _cosine_distance(
            futures.mu[valid].float(), future_valid
        ).sum().item()

        slot_sums += distances.sum(dim=0).double().cpu()

    if count == 0:
        raise ValueError("no valid prediction positions in the query batches")

    fixed_index = int(slot_sums.argmin())
    fixed_distance = slot_sums[fixed_index].item() / count
    hindsight = hindsight_sum / count
    pi_argmax = pi_argmax_sum / count
    mu_direction = mu_dir_sum / count

    return {
        "copy": copy_sum / count,
        "hindsight": hindsight,
        "pi_argmax": pi_argmax,
        "pi_sample": pi_sample_sum / count,
        "mu_safe": mu_safe_sum / count,
        "mu_direction": mu_direction,
        "fixed_slot": {"index": fixed_index, "distance": fixed_distance},
        "oracle_gain_over_mu_direction": mu_direction - hindsight,
        "selection_headroom": pi_argmax - hindsight,
    }


def _rollout_starts(batches, stride: int, steps: int, limit: int, generator):
    """Sample ``(batch, row, t)`` starts where ``t + steps * k`` is valid."""
    candidates: list[tuple[int, int, int]] = []
    for index, (_, attention_mask) in enumerate(batches):
        last = attention_mask.sum(dim=-1) - 1 - steps * stride
        for row, last_t in enumerate(last.tolist()):
            candidates.extend(
                (index, row, t) for t in range(int(last_t) + 1)
            )
    if not candidates:
        raise ValueError(
            "no rollout start has t + depths * pred_horizon valid; "
            "the query sequences are too short"
        )
    if len(candidates) > limit:
        picked = torch.randperm(len(candidates), generator=generator)[:limit]
        candidates = [candidates[i] for i in picked.tolist()]

    groups: dict[int, list[tuple[int, int]]] = {}
    for index, row, t in candidates:
        groups.setdefault(index, []).append((row, t))
    return len(candidates), groups


def _rollouts_section(
    model,
    batches,
    latents,
    bank,
    *,
    depths: int,
    positions: int,
    seed: int,
    bank_chunk_size: int | None,
) -> dict:
    """Section 5: rollout error and manifold distance per depth."""
    stride = model.pred_horizon
    device = latents[0].device
    sampled, groups = _rollout_starts(
        batches,
        stride,
        depths,
        positions,
        _generator(torch.device("cpu"), seed),
    )
    sample_generator = _generator(device, seed + 2)

    error_sums = {name: [0.0] * depths for name in ROLLOUT_PATHS}
    manifold_sums = {
        name: [0.0] * depths for name in (*ROLLOUT_PATHS, "real")
    }
    counts = [0] * depths

    for index, pairs in groups.items():
        z = latents[index]
        rows = torch.tensor(
            [row for row, _ in pairs], dtype=torch.long, device=device
        )
        starts = torch.tensor(
            [t for _, t in pairs], dtype=torch.long, device=device
        )
        z_start = z[rows, starts]  # (N, D)
        targets = torch.stack(
            [
                z[rows, starts + (depth + 1) * stride]
                for depth in range(depths)
            ],
            dim=1,
        )  # (N, depths, D)
        n = z_start.shape[0]
        for depth in range(depths):
            counts[depth] += n

        paths = {
            "hindsight": rollout(
                model.predictor,
                z_start,
                Hindsight(),
                depths,
                targets=targets,
            ).future,
            "pi_argmax": rollout(
                model.predictor, z_start, PiArgmax(), depths
            ).future,
            "pi_sample": rollout(
                model.predictor,
                z_start,
                PiSample(generator=sample_generator),
                depths,
            ).future,
            "mu": _mu_path(model.predictor, z_start, depths),
        }

        for name, path in paths.items():
            for depth in range(depths):
                error_sums[name][depth] += (
                    _cosine_distance(path[:, depth, :], targets[:, depth, :])
                    .sum()
                    .item()
                )
                manifold_sums[name][depth] += (
                    _manifold_scores(
                        path[:, depth, :], bank, bank_chunk_size
                    )
                    .sum()
                    .item()
                )
        for depth in range(depths):
            manifold_sums["real"][depth] += (
                _manifold_scores(
                    targets[:, depth, :], bank, bank_chunk_size
                )
                .sum()
                .item()
            )

    def means(sums: list[float]) -> list[float]:
        return [
            sums[depth] / max(1, counts[depth]) for depth in range(depths)
        ]

    return {
        "positions": sampled,
        "depths": depths,
        "stride": stride,
        "error": {name: means(error_sums[name]) for name in ROLLOUT_PATHS},
        "manifold": {
            name: means(manifold_sums[name])
            for name in (*ROLLOUT_PATHS, "real")
        },
    }


def _decoder_section(
    model, batches, latents, *, seed: int, loss_chunk_size: int | None
) -> dict:
    """Section 6: token-weighted CE per hint condition."""
    horizon = model.pred_horizon
    device = latents[0].device
    sample_generator = _generator(device, seed + 3)
    shuffle_generator = _generator(device, seed + 4)

    totals = {name: 0.0 for name in DECODER_CONDITIONS}
    token_count = 0

    for (labels, attention_mask), z in zip(batches, latents):
        futures = model.predictor(z[:, :-horizon, :])
        real_future = z[:, horizon:, :]

        z_hint, _ = build_block_hints(z, horizon)
        committed = {
            "pi_argmax": futures.commit(PiArgmax()).future,
            "pi_sample": futures.commit(
                PiSample(generator=sample_generator)
            ).future,
            "mu_safe": futures.commit(MuSafe()).future,
            "hindsight": futures.commit(Hindsight(), real_future).future,
        }
        order = torch.randperm(
            z.shape[0], generator=shuffle_generator, device=device
        )
        hints = {
            name: build_block_hints(z, horizon, committed=value)[1]
            for name, value in committed.items()
        }
        hints["real"] = build_block_hints(z, horizon)[1]
        hints["shuffled"] = build_block_hints(
            z, horizon, committed=committed["pi_argmax"][order]
        )[1]
        hints["null"] = torch.zeros_like(z_hint)

        for name in DECODER_CONDITIONS:
            total, count = model.decoder.token_loss(
                z_hint.detach(),
                hints[name].detach(),
                labels,
                attention_mask=attention_mask,
                chunk_size=loss_chunk_size,
            )
            totals[name] += _value(total)
        token_count += int(count)

    if token_count == 0:
        raise ValueError("no decoder tokens in the query batches")

    ce = {name: totals[name] / token_count for name in DECODER_CONDITIONS}
    return {
        "ce": ce,
        "tokens": token_count,
        "hint_use": ce["shuffled"] - ce["pi_argmax"],
        "selection_headroom": ce["pi_argmax"] - ce["hindsight"],
        "specificity": {
            "shuffled_minus_hindsight": ce["shuffled"] - ce["hindsight"],
            "pi_argmax_minus_hindsight": ce["pi_argmax"] - ce["hindsight"],
        },
    }


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def _verdict(name, metric, value, threshold, comparison) -> dict:
    value = float(value)
    threshold = float(threshold)
    ok = value > threshold if comparison == ">" else value < threshold
    return {
        "name": name,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "ok": bool(ok),
    }


def _verdicts(metrics) -> list[dict]:
    """Section 7: pass/fail checks, each naming its metric and threshold."""
    heads = metrics["heads"]
    policies = metrics["policies"]
    decoder = metrics["decoder"]
    representation = metrics["representation"]
    rollouts = metrics["rollouts"]

    verdicts = [
        _verdict(
            "pi_uses_z_t", "heads.pi_gain", heads["pi_gain"], 0.0, ">"
        ),
        _verdict(
            "candidates_beat_mean",
            "policies.oracle_gain_over_mu_direction",
            policies["oracle_gain_over_mu_direction"],
            0.01,
            ">",
        ),
        _verdict(
            "decoder_uses_commitment",
            "decoder.hint_use",
            decoder["hint_use"],
            0.05,
            ">",
        ),
        _verdict(
            "mu_calibrated",
            "abs(heads.mu_calibration_error)",
            abs(heads["mu_calibration_error"]),
            0.05,
            "<",
        ),
        _verdict(
            "encoder_center",
            "representation.visreg.center",
            representation["visreg"]["center"],
            0.05,
            "<",
        ),
        _verdict(
            "encoder_effective_rank",
            "representation.effective_rank",
            representation["effective_rank"],
            15.0,
            ">",
        ),
    ]

    for depth in range(2, rollouts["depths"] + 1):
        pi_sample = rollouts["manifold"]["pi_sample"][depth - 1]
        mu = rollouts["manifold"]["mu"][depth - 1]
        verdicts.append(
            _verdict(
                f"committed_on_manifold_depth_{depth}",
                f"rollouts.manifold.pi_sample[depth={depth}] - "
                f"rollouts.manifold.mu[depth={depth}]",
                pi_sample - mu,
                0.0,
                "<",
            )
        )
    return verdicts


# ---------------------------------------------------------------------------
# Collect
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_metrics(
    model,
    batches,
    bank,
    *,
    depths: int = DEFAULT_DEPTHS,
    rollout_positions: int = DEFAULT_ROLLOUT_POSITIONS,
    seed: int = 0,
    loss_chunk_size: int | None = DEFAULT_LOSS_CHUNK_SIZE,
    bank_chunk_size: int | None = None,
) -> dict:
    """Measure ``model`` on ``batches`` against a bank of real latents.

    Args:
        model: a :class:`~jepa.models.jepa.JepaModel`; put in eval mode here.
        batches: iterable of ``(labels, attention_mask)`` pairs, as produced by
            :func:`load_query_batches`.
        bank: ``(N, D)`` real encoder latents (raw scale; normalised here).
        depths: rollout depths, 1..``depths``.
        rollout_positions: number of sampled start positions for the rollouts.
        seed: seed for the start positions and every π-sample / shuffle.
        loss_chunk_size: positions per ``lm_head`` chunk in the decoder loss.
        bank_chunk_size: query rows per bank chunk; None bounds memory
            automatically.

    Returns:
        A JSON-serialisable dict with one nested dict per section:
        ``context``, ``representation``, ``loss``, ``heads``, ``policies``,
        ``rollouts``, ``decoder`` and ``verdicts``.
    """
    if depths < 1:
        raise ValueError("depths must be at least 1")
    if rollout_positions < 1:
        raise ValueError("rollout_positions must be at least 1")

    model.eval()
    batches = list(batches)
    if not batches:
        raise ValueError("at least one query batch is required")
    horizon = model.pred_horizon
    for labels, attention_mask in batches:
        if attention_mask is None:
            raise ValueError("measure needs attention masks (1 = real token)")
        if labels.shape[1] <= horizon:
            raise ValueError(
                f"query length {labels.shape[1]} must exceed pred_horizon "
                f"{horizon}"
            )

    device = next(model.parameters()).device
    batches = [
        (labels.to(device), attention_mask.to(device))
        for labels, attention_mask in batches
    ]
    bank = bank.float().to(device)
    if bank.ndim != 2 or bank.shape[0] < 1:
        raise ValueError(
            "bank must be a non-empty (N, D) tensor of real latents"
        )
    bank = F.normalize(bank, dim=-1)

    latents = [
        model.encoder(labels, attention_mask=attention_mask)
        for labels, attention_mask in batches
    ]

    sections = {
        "representation": _representation_section(model, batches, latents),
        "loss": _loss_section(model, batches),
        "heads": _heads_section(model, batches, latents),
        "policies": _policies_section(model, batches, latents, seed=seed),
        "rollouts": _rollouts_section(
            model,
            batches,
            latents,
            bank,
            depths=depths,
            positions=rollout_positions,
            seed=seed,
            bank_chunk_size=bank_chunk_size,
        ),
        "decoder": _decoder_section(
            model,
            batches,
            latents,
            seed=seed,
            loss_chunk_size=loss_chunk_size,
        ),
    }

    context = {
        "checkpoint": None,
        "device": None,
        "examples": int(sum(labels.shape[0] for labels, _ in batches)),
        "max_length": int(max(labels.shape[1] for labels, _ in batches)),
        "tokens": int(sum(int(mask.sum()) for _, mask in batches)),
        "pred_horizon": int(horizon),
        "candidates": int(model.config.predictor.prediction_count),
        "depths": int(depths),
        "rollout_positions": int(sections["rollouts"]["positions"]),
        "bank_size": int(bank.shape[0]),
        "latent_dim": int(bank.shape[1]),
        "seed": int(seed),
    }

    metrics = {"context": context, **sections}
    metrics["verdicts"] = _verdicts(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(value) -> str:
    return f"{float(value):.4f}"


def _table(rows, headers) -> None:
    widths = [
        max([len(str(header))] + [len(str(row[i])) for row in rows])
        for i, header in enumerate(headers)
    ]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print(
            "  ".join(
                str(cell).ljust(width) for cell, width in zip(row, widths)
            )
        )


def print_report(metrics: dict) -> None:
    """Print every metric and verdict in ``collect_metrics``'s dict."""
    context = metrics["context"]
    print("=" * 78)
    checkpoint = context.get("checkpoint") or "(in-memory model)"
    print(f"measure: {checkpoint}")
    print(
        f"device: {context.get('device') or '?'} | "
        f"examples: {context['examples']} x {context['max_length']} | "
        f"tokens: {context['tokens']:,} | "
        f"k: {context['pred_horizon']} | "
        f"candidates: {context['candidates']} | "
        f"bank: {context['bank_size']:,}"
    )
    print("=" * 78)

    representation = metrics["representation"]
    print("\n--- representation ---")
    print(f"per-dim std mean:   {_fmt(representation['per_dim_std'])}")
    print(f"mean vector norm:   {_fmt(representation['mean_vector_norm'])}")
    print(
        f"effective rank:     {_fmt(representation['effective_rank'])} / "
        f"{context['latent_dim']}  ({representation['latents']:,} latents)"
    )
    print(f"cos(z_t, z_t+k):    {_fmt(representation['temporal_cos'])}")
    print(
        "VISReg:             "
        + " | ".join(
            f"{name} {_fmt(value)}"
            for name, value in representation["visreg"].items()
        )
    )

    loss = metrics["loss"]
    print("\n--- jepa loss ---")
    rows = [
        (name, _fmt(term["weight"]), _fmt(term["value"]), _fmt(term["weighted"]))
        for name, term in loss["terms"].items()
    ]
    _table(rows, ("term", "weight", "value", "weighted"))
    print(f"{'total':<12}{_fmt(loss['total']):>24}")
    print(
        f"pi floor: H(r)={_fmt(loss['floor']['assignment_entropy'])}  "
        f"KL(r||pi)={_fmt(loss['floor']['kl_to_pi'])}  "
        "(only the KL part is learnable)"
    )

    heads = metrics["heads"]
    print("\n--- heads ---")
    print(f"best_d:                   {_fmt(heads['best_d'])}")
    print(f"candidate pair cos:       {_fmt(heads['candidate_pair_cos'])}")
    print(f"|delta| / |z|:            {_fmt(heads['delta_ratio'])}")
    print(
        "win shares (hindsight):   "
        + " ".join(_fmt(share) for share in heads["win_shares"])
    )
    print(
        f"live slots:               {heads['live_slots']} / "
        f"{len(heads['win_shares'])}  "
        f"(effective {_fmt(heads['effective_slots'])})"
    )
    print(
        f"pi NLL:                   {_fmt(heads['pi_nll'])}  vs constant "
        f"{_fmt(heads['constant_nll'])}  "
        f"(gain {_fmt(heads['pi_gain'])} nats)"
    )
    print(
        f"pi accuracy:              {_fmt(heads['pi_accuracy'])}  vs majority "
        f"{_fmt(heads['majority_accuracy'])}"
    )
    print(f"|mu|:                     {_fmt(heads['mu_norm'])}")
    print(
        f"mu direction distance:    {_fmt(heads['mu_dir_distance_measured'])} "
        f"measured vs {_fmt(heads['mu_dir_distance_predicted'])} predicted "
        f"(2 - 2|mu|)"
    )

    policies = metrics["policies"]
    print("\n--- policies (depth 1, distance to the real z_t+k) ---")
    rows = [
        (name, _fmt(policies[name]))
        for name in (
            "copy",
            "hindsight",
            "pi_argmax",
            "pi_sample",
            "mu_safe",
            "mu_direction",
        )
    ]
    rows.append(
        (
            f"fixed slot {policies['fixed_slot']['index']}",
            _fmt(policies["fixed_slot"]["distance"]),
        )
    )
    _table(rows, ("policy", "distance"))
    print(
        f"oracle gain over mu direction: "
        f"{_fmt(policies['oracle_gain_over_mu_direction'])}"
    )
    print(
        f"selection headroom (pi-argmax - hindsight): "
        f"{_fmt(policies['selection_headroom'])}"
    )

    rollouts = metrics["rollouts"]
    print(
        f"\n--- rollouts ({rollouts['positions']} starts, "
        f"k={rollouts['stride']}) ---"
    )
    print("rollout error (distance to the real latent):")
    rows = [
        (
            depth,
            *(
                _fmt(rollouts["error"][name][depth - 1])
                for name in ROLLOUT_PATHS
            ),
        )
        for depth in range(1, rollouts["depths"] + 1)
    ]
    _table(rows, ("depth", *ROLLOUT_PATHS))
    print("manifold distance (nearest real latent in the bank):")
    rows = [
        (
            depth,
            *(
                _fmt(rollouts["manifold"][name][depth - 1])
                for name in (*ROLLOUT_PATHS, "real")
            ),
        )
        for depth in range(1, rollouts["depths"] + 1)
    ]
    _table(rows, ("depth", *ROLLOUT_PATHS, "real"))

    decoder = metrics["decoder"]
    print(
        f"\n--- decoder (token-weighted CE over {decoder['tokens']:,} "
        "tokens) ---"
    )
    rows = [
        (name, _fmt(value))
        for name, value in decoder["ce"].items()
    ]
    _table(rows, ("hint", "CE"))
    print(f"hint use (shuffled - pi-argmax):       {_fmt(decoder['hint_use'])}")
    print(
        "selection headroom (pi-argmax - hindsight): "
        f"{_fmt(decoder['selection_headroom'])}"
    )
    print(
        "specificity: shuffled-hindsight "
        f"{_fmt(decoder['specificity']['shuffled_minus_hindsight'])}  vs "
        "pi-argmax-hindsight "
        f"{_fmt(decoder['specificity']['pi_argmax_minus_hindsight'])}"
    )

    print("\n--- verdicts ---")
    for verdict in metrics["verdicts"]:
        state = "PASS" if verdict["ok"] else "FAIL"
        print(
            f"[{state}] {verdict['name']}: {verdict['metric']} = "
            f"{_fmt(verdict['value'])} "
            f"({verdict['comparison']} {_fmt(verdict['threshold'])})"
        )

    print("\n--- how to read this ---")
    print("representation: std near 1 and effective rank > 15 = no collapse;")
    print("                cos(z_t, z_t+k) 0.5-0.85 is healthy, >0.95 too smooth.")
    print("loss:           total is exactly the weighted sum of the terms; the")
    print("                pi floor H(r) is fixed by the targets, so only KL(r||pi)")
    print("                can improve.")
    print("heads:          best_d lower is better; pi gain > 0 = pi predicts the")
    print("                closest candidate from z_t; effective slots = exp(H(win")
    print("                shares)); mu calibration compares the measured mu")
    print("                direction distance with 2 - 2|mu|.")
    print("policies:       hindsight is the oracle. Oracle gain over mu direction is")
    print("                what P candidates buy over one mean prediction;")
    print("                selection headroom is what a better selector could add")
    print("                without better candidates.")
    print("rollouts:       error is distance to the real z at t+i*k; manifold is")
    print("                distance to the nearest real latent. The claim: at")
    print("                depths >= 2, pi-sample stays closer to the manifold than")
    print("                the mu path (compare with the real row).")
    print("decoder:        hint use = shuffled - pi-argmax (>0: the decoder uses the")
    print("                committed future); specificity compares")
    print("                shuffled-hindsight (topic) with pi-argmax-hindsight (the")
    print("                specific future).")
    print("verdicts:       PASS/FAIL on the checks above, each naming its metric.")
    print("                Old checkpoints need the old code: run it in a worktree.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m jepa measure", description=__doc__
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    parser.add_argument(
        "--fineweb-dir",
        type=Path,
        default=FINEWEB_DIR,
        help="Prepared token chunks: the validation split is queried and the "
        "train split supplies the real-latent bank.",
    )
    parser.add_argument("--num-examples", type=int, default=DEFAULT_QUERY_EXAMPLES)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--bank-size",
        type=int,
        default=DEFAULT_BANK_SIZE,
        help="Real train-split latents in the manifold bank.",
    )
    parser.add_argument(
        "--rollout-positions",
        type=int,
        default=DEFAULT_ROLLOUT_POSITIONS,
        help="Sampled start positions for the rollout table.",
    )
    parser.add_argument("--depths", type=int, default=DEFAULT_DEPTHS)
    parser.add_argument(
        "--loss-chunk-size",
        type=int,
        default=DEFAULT_LOSS_CHUNK_SIZE,
        help="Decoder positions per lm_head chunk (bounds peak memory).",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json", type=Path, default=None, help="Write every metric as JSON."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    tokenizer = load_tokenizer(args.tokenizer)
    model = JepaModel.from_pretrained(str(args.model)).to(device).eval()
    if len(tokenizer) != model.config.encoder.vocab_size:
        raise ValueError(
            f"{args.model} was trained with vocab_size="
            f"{model.config.encoder.vocab_size}, but {args.tokenizer} has "
            f"{len(tokenizer)} tokens. Pass the matching --tokenizer."
        )

    train_dataset, validation = load_datasets(args.fineweb_dir)
    batches = load_query_batches(
        validation,
        tokenizer,
        num_examples=args.num_examples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    bank = build_bank(
        model,
        train_dataset,
        tokenizer,
        size=args.bank_size,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    metrics = collect_metrics(
        model,
        batches,
        bank,
        depths=args.depths,
        rollout_positions=args.rollout_positions,
        seed=args.seed,
        loss_chunk_size=args.loss_chunk_size,
    )
    metrics["context"]["checkpoint"] = str(args.model)
    metrics["context"]["device"] = str(device)

    print_report(metrics)

    if args.json is not None:
        args.json.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
