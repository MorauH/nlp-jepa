"""The paper's numbers: representation, commitment, rollouts and decoding.

    python -m jepa measure --model runs/jepa
    python -m jepa measure --model runs/jepa --json runs/jepa/metrics.json

Queries are validation chunks; the manifold bank samples real encoder latents
from the train split (whole documents are held out, so they do not overlap).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ..config import EncoderConfig
from ..data.tokenizer import load_tokenizer
from ..models.commitment import (
    Hindsight,
    MuSafe,
    PiArgmax,
    PiSample,
    distance,
    pairwise_distance,
    rollout,
)
from ..models.jepa import JepaModel, build_block_hints
from ..paths import DEFAULT_RUN_DIR, FINEWEB_DIR, TOKENIZER_DIR
from ..runtime import select_device
from ..training.data import load_datasets, tokenize_batch

DEFAULT_QUERY_EXAMPLES = 64
DEFAULT_MAX_LENGTH = EncoderConfig().max_position_embeddings
DEFAULT_BATCH_SIZE = 4
DEFAULT_BANK_SIZE = 50_000
DEFAULT_ROLLOUT_POSITIONS = 2_048
DEFAULT_DEPTHS = 4
DEFAULT_BEST_OF = (1, 4, 16)
DEFAULT_LOSS_CHUNK_SIZE = None
REPRESENTATION_LATENTS = 20_000  # effective rank is sample-size sensitive
ROLLOUT_PATHS = ("hindsight", "pi_argmax", "pi_sample", "mu")
DECODER_CONDITIONS = (
    "pi_argmax", "pi_sample", "mu_safe", "hindsight", "real", "shuffled", "null",
    "no_hints",
)
# Lags beyond the horizon, as multiples of it, for the long end of the profile.
LONG_LAG_MULTIPLES = (2, 5, 10, 20)
# A drifting latent has rebound <= 0; the period-10 shortcut of the first
# two-stage run scored about 1.1.  The margin allows for sampling noise.
REBOUND_THRESHOLD = 0.05

def _value(value) -> float:
    return float(value.detach().float().item()) if isinstance(
        value, torch.Tensor
    ) else float(value)

def _generator(device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator

def _prediction_mask(mask, horizon: int):
    """A real token with a real token ``horizon`` ahead."""
    return mask[:, :-horizon].bool() & mask[:, horizon:].bool()

def _walk(latents, masks, horizon: int):
    for z, mask in zip(latents, masks):
        yield z[:, :-horizon], z[:, horizon:], _prediction_mask(mask, horizon)

def _commit_distance(distances, futures, policy, valid, target=None):
    """Distance of the candidate ``policy`` commits to, one per valid position."""
    index = futures.commit(policy, target).index[valid]
    return distances.gather(-1, index[:, None]).squeeze(-1)

def mu_path(predictor, z, steps: int):
    """The μ path: feed back ``normalize(μ)`` at the latent scale.

    ``z`` is ``(..., D)``; the path is ``(..., steps, D)``.  Shared with the
    decoded plans in :mod:`jepa.inference.generate`.
    """
    path = []
    for _ in range(steps):
        mu = predictor(z).mu
        z = F.normalize(mu.float(), dim=-1).to(mu.dtype) * math.sqrt(z.shape[-1])
        path.append(z)
    return torch.stack(path, dim=-2)

def lag_profile(latents, masks, lags) -> dict[int, float | None]:
    """Mean cos(z_t, z_{t+lag}) over every pair of real tokens, per lag.

    ``latents`` and ``masks`` are matching lists of ``(batch, seq, D)`` and
    ``(batch, seq)``.  A lag with no pair in any sequence maps to ``None``.
    """
    profile = {}
    for lag in lags:
        lag = int(lag)
        total, count = 0.0, 0
        for z, mask in zip(latents, masks):
            if z.shape[1] <= lag:
                continue
            pairs = _prediction_mask(mask, lag)
            if pairs.any():
                cos = F.cosine_similarity(
                    z[:, :-lag][pairs].float(), z[:, lag:][pairs].float(), dim=-1
                )
                total += cos.sum().item()
                count += cos.numel()
        profile[lag] = total / count if count else None
    return profile

def periodic_rebound(profile, horizon: int) -> float | None:
    """cos at lag ``horizon`` minus the lowest cos at lags 1..horizon-1.

    A latent that drifts loses similarity with lag, so its rebound is <= 0.
    A latent with a period dividing the horizon dips below and returns at the
    horizon, a shortcut the prediction loss cannot see.  ``None`` when the
    profile lacks the horizon or every shorter lag.
    """
    shorter = [profile.get(lag) for lag in range(1, horizon)]
    shorter = [value for value in shorter if value is not None]
    if profile.get(horizon) is None or not shorter:
        return None
    return profile[horizon] - min(shorter)

def _centred_kernel(window: int) -> torch.Tensor:
    """A centred moving average spanning exactly ``window`` steps.

    An even window gets ``window + 1`` taps with half weight at both ends, so
    it averages exactly one period of length ``window``; an odd window is
    ``window`` equal taps.  Either way a linear drift passes unchanged.
    """
    if window % 2 == 0:
        taps = torch.ones(window + 1, dtype=torch.float64)
        taps[0] = taps[-1] = 0.5
    else:
        taps = torch.ones(window, dtype=torch.float64)
    return taps / taps.sum()

def fast_share(latents, masks, window: int) -> float | None:
    """Share of the within-document energy that varies faster than ``window``.

    Unit latents are split into a centred ``window``-step moving average (the
    slow part) and the residual (the fast part).  The share is the fast
    energy over fast + slow, each the mean over documents of the squared norm
    per position (the slow part around its document mean).  ``None`` when no
    document is longer than the kernel.  Token-to-token change counts as fast
    too, so the share has no healthy reference value; it is reported, not
    judged.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2; got {window}")
    kernel = _centred_kernel(window)
    half = kernel.numel() // 2
    fast_sum = slow_sum = 0.0
    documents = 0
    for z, mask in zip(latents, masks):
        lengths = mask.sum(dim=-1).tolist()
        for row, length in enumerate(int(n) for n in lengths):
            if length <= kernel.numel():
                continue
            unit = F.normalize(z[row, :length].double(), dim=-1)
            slow = F.conv1d(
                unit.T.unsqueeze(1), kernel.to(unit.device).view(1, 1, -1)
            ).squeeze(1).T  # (length - 2 * half, D)
            fast = unit[half : length - half] - slow
            fast_sum += (fast**2).sum(dim=-1).mean().item()
            slow_sum += ((slow - slow.mean(dim=0)) ** 2).sum(dim=-1).mean().item()
            documents += 1
    total = fast_sum + slow_sum
    if documents == 0 or total == 0:
        return None
    return fast_sum / total

def _manifold_scores(points, bank, chunk_size):
    """Nearest-neighbour distance of each row to the bank."""
    chunk_size = chunk_size or max(1, 4_000_000 // bank.shape[0])
    rows = [
        pairwise_distance(points[i : i + chunk_size], bank).min(dim=-1).values
        for i in range(0, points.shape[0], chunk_size)
    ]
    return torch.cat(rows)

def load_query_batches(
    dataset, tokenizer, *, num_examples=DEFAULT_QUERY_EXAMPLES,
    max_length=DEFAULT_MAX_LENGTH, batch_size=DEFAULT_BATCH_SIZE, seed=0,
):
    """Sample validation chunks as padded ``(labels, attention_mask)`` batches."""
    generator = torch.Generator().manual_seed(seed)
    count = min(num_examples, len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    examples = [
        {"input_ids": list(dataset[i]["input_ids"])[:max_length]} for i in indices
    ]
    return [
        tokenize_batch(examples[i : i + batch_size], tokenizer)
        for i in range(0, count, batch_size)
    ]

@torch.no_grad()
def build_bank(
    model, dataset, tokenizer, *, size=DEFAULT_BANK_SIZE,
    max_length=DEFAULT_MAX_LENGTH, batch_size=DEFAULT_BATCH_SIZE, seed=0,
) -> torch.Tensor:
    """Sample real encoder latents from the train split for the manifold bank."""
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    chunks, collected = [], 0
    for start in range(0, len(order), batch_size):
        examples = [
            {"input_ids": list(dataset[i]["input_ids"])[:max_length]}
            for i in order[start : start + batch_size]
        ]
        labels, mask = tokenize_batch(examples, tokenizer)
        latents = model.encode(labels.to(device), attention_mask=mask.to(device))
        valid = latents[mask.bool().to(device)]
        chunks.append(valid.float().cpu())
        collected += valid.shape[0]
        if collected >= size:
            break
    bank = torch.cat(chunks)
    if bank.shape[0] > size:
        bank = bank[torch.randperm(bank.shape[0], generator=generator)[:size]]
    return bank

def _representation_section(model, batches, latents) -> dict:
    """Section 1: no-collapse checks on the encoder latents."""
    horizon = model.pred_horizon
    pooled, seen = [], 0
    token_count = 0
    visreg_sums = {name: 0.0 for name in model.visreg.weights}
    for (_, mask), z in zip(batches, latents):
        valid = mask.bool()
        tokens = int(valid.sum())
        token_count += tokens
        if seen < REPRESENTATION_LATENTS:
            take = z[valid].float()[: REPRESENTATION_LATENTS - seen]
            pooled.append(take)
            seen += take.shape[0]
        for name, value in model.visreg(z, mask=mask).terms.items():
            visreg_sums[name] += _value(value) * tokens
    masks = [mask for _, mask in batches]
    lags = (*range(1, horizon + 1), *(m * horizon for m in LONG_LAG_MULTIPLES))
    profile = lag_profile(latents, masks, lags)
    z_all = torch.cat(pooled)
    mean, centered = z_all.mean(dim=0), z_all - z_all.mean(dim=0)
    covariance = centered.T @ centered / max(1, z_all.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    shares = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    rank = torch.exp(-(shares * shares.clamp_min(1e-12).log()).sum())
    return {
        "per_dim_std": _value(centered.std(dim=0, unbiased=False).mean()),
        "mean_vector_norm": _value(mean.norm()),
        "effective_rank": _value(rank),
        "latents": int(z_all.shape[0]),
        "temporal_cos": profile[horizon] if profile[horizon] is not None else 0.0,
        # String keys: the dict is written to JSON as is.
        "lag_cos": {str(lag): value for lag, value in profile.items()},
        "periodic_rebound": periodic_rebound(profile, horizon),
        "fast_share": fast_share(latents, masks, window=max(2, horizon // 2)),
        "visreg": {name: visreg_sums[name] / max(1, token_count) for name in visreg_sums},
    }

def _loss_section(model, batches) -> dict:
    """Section 2: every JEPA term, its weight, and its contribution."""
    horizon = model.pred_horizon
    weights = model.loss_weights
    term_sums, diagnostic_sums, count = {}, {}, 0
    for labels, mask in batches:
        valid = int(_prediction_mask(mask, horizon).sum())
        if valid == 0:
            continue
        out = model.jepa_loss(labels, attention_mask=mask, progress=None)
        for name, value in out.terms.items():
            term_sums[name] = term_sums.get(name, 0.0) + _value(value) * valid
        for name, value in out.diagnostics.items():
            diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + _value(value) * valid
        count += valid
    if count == 0:
        raise ValueError("no valid prediction positions in the query batches")
    terms = {}
    for name, total in term_sums.items():
        value, weight = total / count, float(weights[name])
        terms[name] = {"weight": weight, "value": value, "weighted": weight * value}
    entropy = diagnostic_sums.get("assignment_entropy", 0.0) / count
    return {
        "terms": terms,
        "total": sum(term["weighted"] for term in terms.values()),
        "floor": {"assignment_entropy": entropy, "kl_to_pi": terms["pi"]["value"] - entropy},
        "diagnostics": {name: value / count for name, value in diagnostic_sums.items()},
    }

def _heads_section(model, batches, latents) -> dict:
    """Section 3: candidates, π, μ and their calibration."""
    horizon = model.pred_horizon
    masks = [mask for _, mask in batches]
    prediction_count = model.config.predictor.prediction_count
    count = 0
    pair_sim_sum = delta_sum = z_norm_sum = 0.0
    pi_nll_sum = mu_norm_sum = mu_distance_sum = 0.0
    pair_count = correct = 0
    hindsight_counts = torch.zeros(prediction_count, dtype=torch.long)
    for z_t, future, valid in _walk(latents, masks, horizon):
        if not valid.any():
            continue
        futures = model.predictor(z_t)
        future_valid, z_valid = future[valid].float(), z_t[valid].float()
        n = int(valid.sum())
        count += n
        # The hindsight commit decides the reference slot and the π target.
        hindsight_index = futures.commit(Hindsight(), future).index[valid]
        if prediction_count > 1:
            sim = futures.candidate_similarity()[valid]  # (N, P, P)
            off_diagonal = 1.0 - torch.eye(
                prediction_count, device=sim.device, dtype=sim.dtype
            )
            pair_sim_sum += (sim * off_diagonal).sum().item()
            pair_count += n * prediction_count * (prediction_count - 1)
        delta_sum += (
            futures.candidates[valid].float() - z_valid.unsqueeze(-2)
        ).norm(dim=-1).sum().item()
        z_norm_sum += z_valid.norm(dim=-1).sum().item()
        hindsight_counts += torch.bincount(
            hindsight_index, minlength=prediction_count
        ).cpu()
        log_pi = F.log_softmax(futures.pi[valid].float(), dim=-1)
        pi_nll_sum += -log_pi.gather(-1, hindsight_index[:, None]).sum().item()
        correct += (log_pi.argmax(dim=-1) == hindsight_index).sum().item()
        mu = futures.mu[valid].float()
        mu_norm_sum += mu.norm(dim=-1).sum().item()
        mu_distance_sum += distance(mu, future_valid).sum().item()
    if count == 0:
        raise ValueError("no valid prediction positions in the query batches")
    shares = hindsight_counts.float() / hindsight_counts.sum().clamp_min(1)
    entropy = float(-(shares * shares.clamp_min(1e-12).log()).sum())
    pi_nll = pi_nll_sum / count
    mu_norm = mu_norm_sum / count
    mu_distance = mu_distance_sum / count
    # If μ is calibrated, E[d(μ̂, ŷ)] = 2 - 2|μ| (μ is a mean of unit futures,
    # not renormalized).
    mu_distance_predicted = 2.0 - 2.0 * mu_norm
    choices = {
        "candidate_pair_cos": pair_sim_sum / pair_count if pair_count else None,
        "live_slots": int((shares > 1.0 / (4 * prediction_count)).sum()),
        "effective_slots": math.exp(entropy),
        "pi_nll": pi_nll,
        "constant_nll": entropy,
        "pi_gain": entropy - pi_nll,
        "pi_accuracy": correct / count,
        "majority_accuracy": float(shares.max()),
    }
    if prediction_count == 1:  # nothing to choose between: n/a, not a number
        choices = dict.fromkeys(choices)
    return {
        **choices,
        "delta_ratio": (delta_sum / (count * prediction_count)) / max(z_norm_sum / count, 1e-12),
        "hindsight_shares": [float(x) for x in shares.tolist()],
        "mu_norm": mu_norm,
        "mu_distance_measured": mu_distance,
        "mu_distance_predicted": mu_distance_predicted,
        "mu_calibration_error": mu_distance - mu_distance_predicted,
    }

def _policies_section(model, batches, latents, *, seed: int) -> dict:
    """Section 4: distance-to-real at depth 1 for every commit rule."""
    horizon = model.pred_horizon
    masks = [mask for _, mask in batches]
    prediction_count = model.config.predictor.prediction_count
    generator, count = _generator(latents[0].device, seed + 1), 0
    sums = dict.fromkeys(
        ("copy", "hindsight", "pi_argmax", "pi_sample", "mu_safe", "mu"),
        0.0,
    )
    slot_sums = torch.zeros(prediction_count, dtype=torch.float64)
    for z_t, future, valid in _walk(latents, masks, horizon):
        if not valid.any():
            continue
        futures = model.predictor(z_t)
        distances = futures.distances(future)[valid]
        future_valid, z_valid = future[valid].float(), z_t[valid].float()
        count += distances.shape[0]
        sums["copy"] += distance(z_valid, future_valid).sum().item()
        sums["hindsight"] += _commit_distance(
            distances, futures, Hindsight(), valid, target=future
        ).sum().item()
        for name, rule in (
            ("pi_argmax", PiArgmax()),
            ("pi_sample", PiSample(generator=generator)),
            ("mu_safe", MuSafe()),
        ):
            sums[name] += _commit_distance(
                distances, futures, rule, valid
            ).sum().item()
        sums["mu"] += distance(futures.mu[valid].float(), future_valid).sum().item()
        slot_sums += distances.double().cpu().sum(dim=-2)
    if count == 0:
        raise ValueError("no valid prediction positions in the query batches")
    fixed_index = int(slot_sums.argmin())
    hindsight, pi_argmax = sums["hindsight"] / count, sums["pi_argmax"] / count
    return {
        **{name: value / count for name, value in sums.items()},
        "fixed_slot": {"index": fixed_index, "distance": slot_sums[fixed_index].item() / count},
        "hindsight_gain_over_mu": sums["mu"] / count - hindsight,
        "commit_headroom": pi_argmax - hindsight,
    }

def _rollout_starts(batches, stride, steps, limit, generator):
    """Sample ``(batch, row, t)`` starts where ``t + steps * k`` is valid."""
    candidates = []
    for index, (_, mask) in enumerate(batches):
        last = mask.sum(dim=-1) - 1 - steps * stride
        for row, last_t in enumerate(last.tolist()):
            candidates.extend((index, row, t) for t in range(int(last_t) + 1))
    if not candidates:
        raise ValueError(
            "no rollout start has t + depths * pred_horizon valid; "
            "the query sequences are too short"
        )
    if len(candidates) > limit:
        sampled = torch.randperm(len(candidates), generator=generator)[:limit]
        candidates = [candidates[i] for i in sampled.tolist()]
    groups = {}
    for index, row, t in candidates:
        groups.setdefault(index, []).append((row, t))
    return len(candidates), groups

def _mean_rollout_error(errors):
    """Mean error over depths 1..i at each depth i, ``(..., depths)``."""
    depth = torch.arange(1, errors.shape[-1] + 1, device=errors.device)
    return errors.cumsum(dim=-1) / depth

def _best_of_k(predictor, z_start, targets, best_of, generator):
    """Best-of-K rollout error per start, ``{K: (N, depths)}``.

    Draws ``max(best_of)`` independent π-sample rollouts in one batch;
    best-of-K keeps the first K, so the draws are nested and the result is
    non-increasing in K.  At depth i each rollout is scored by its mean error
    over depths 1..i, so the minimum is one whole rollout, never a mix of
    rollouts picked per depth.
    """
    draws = max(best_of)
    starts = z_start.unsqueeze(0).expand(draws, *z_start.shape)
    paths = rollout(
        predictor, starts, PiSample(generator=generator), targets.shape[-2]
    ).future
    scores = _mean_rollout_error(distance(paths, targets.unsqueeze(0)))  # (K, N, depths)
    return {k: scores[:k].min(dim=0).values for k in best_of}

def _rollouts_section(
    model, batches, latents, bank, *, depths, positions, seed, bank_chunk_size,
    best_of,
) -> dict:
    """Section 5: rollout error, best-of-K error and manifold distance per depth."""
    stride = model.pred_horizon
    device = latents[0].device
    sampled, groups = _rollout_starts(
        batches, stride, depths, positions, _generator(torch.device("cpu"), seed)
    )
    sample_generator = _generator(device, seed + 2)
    best_generator = _generator(device, seed + 5)
    best_sums = {k: torch.zeros(depths, dtype=torch.float64) for k in best_of}
    error_sums = {name: [0.0] * depths for name in ROLLOUT_PATHS}
    manifold_sums = {name: [0.0] * depths for name in (*ROLLOUT_PATHS, "real")}
    counts = [0] * depths
    for index, pairs in groups.items():
        z = latents[index]
        rows = torch.tensor([row for row, _ in pairs], dtype=torch.long, device=device)
        starts = torch.tensor([t for _, t in pairs], dtype=torch.long, device=device)
        z_start = z[rows, starts]
        targets = torch.stack(
            [z[rows, starts + (depth + 1) * stride] for depth in range(depths)], dim=1
        )
        for depth in range(depths):
            counts[depth] += z_start.shape[0]
        paths = {
            "hindsight": rollout(
                model.predictor, z_start, Hindsight(), depths, targets=targets
            ).future,
            "pi_argmax": rollout(model.predictor, z_start, PiArgmax(), depths).future,
            "pi_sample": rollout(
                model.predictor, z_start, PiSample(generator=sample_generator), depths
            ).future,
            "mu": mu_path(model.predictor, z_start, depths),
        }
        for name, path in paths.items():
            for depth in range(depths):
                error_sums[name][depth] += distance(
                    path[:, depth], targets[:, depth]
                ).sum().item()
                manifold_sums[name][depth] += _manifold_scores(
                    path[:, depth], bank, bank_chunk_size
                ).sum().item()
        for depth in range(depths):
            manifold_sums["real"][depth] += _manifold_scores(
                targets[:, depth], bank, bank_chunk_size
            ).sum().item()
        # Same memory bound as the single rollouts: at most one group of
        # starts per predictor call, counting the K draws.
        chunk = max(1, z_start.shape[0] // max(best_of))
        for i in range(0, z_start.shape[0], chunk):
            best = _best_of_k(
                model.predictor, z_start[i : i + chunk], targets[i : i + chunk],
                best_of, best_generator,
            )
            for k, scores in best.items():
                best_sums[k] += scores.double().sum(dim=0).cpu()

    def means(sums):
        return [sums[depth] / max(1, counts[depth]) for depth in range(depths)]

    return {
        "positions": sampled,
        "depths": depths,
        "stride": stride,
        "error": {name: means(error_sums[name]) for name in ROLLOUT_PATHS},
        "best_of_k": {str(k): means(best_sums[k].tolist()) for k in best_of},
        "manifold": {name: means(manifold_sums[name]) for name in (*ROLLOUT_PATHS, "real")},
    }

def _decoder_section(model, batches, latents, *, seed, loss_chunk_size) -> dict:
    """Section 6: token-weighted CE per hint condition.

    Every condition uses the same mask: targets whose block runs past the end
    of the sequence have no committed future and are excluded.  The π-argmax
    row is the inference hint; the hindsight row is teacher-forced and its hint
    carries up to log2(P) bits about the target future from the choice among
    candidates, so it is a lower bound on CE rather than a leak-free condition.
    """
    horizon = model.pred_horizon
    device = latents[0].device
    sample_generator = _generator(device, seed + 3)
    shuffle_generator = _generator(device, seed + 4)
    totals = {name: 0.0 for name in DECODER_CONDITIONS}
    token_count = 0
    for (labels, mask), z in zip(batches, latents):
        futures = model.predictor(z[:, :-horizon])
        real_future = z[:, horizon:]
        z_hint, _, hint_valid = build_block_hints(z, horizon)
        committed = {
            "pi_argmax": futures.commit(PiArgmax()).future,
            "pi_sample": futures.commit(PiSample(generator=sample_generator)).future,
            "mu_safe": futures.commit(MuSafe()).future,
            "hindsight": futures.commit(Hindsight(), real_future).future,
        }
        order = torch.randperm(z.shape[0], generator=shuffle_generator, device=device)
        hints = {
            name: build_block_hints(z, horizon, committed=value)[1]
            for name, value in committed.items()
        }
        hints["real"] = build_block_hints(z, horizon)[1]
        hints["shuffled"] = build_block_hints(
            z, horizon, committed=committed["pi_argmax"][order]
        )[1]
        hints["null"] = torch.zeros_like(z_hint)
        hints["no_hints"] = hints["null"]
        # Same exclusion as decoder_loss: no committed future for the tail
        # block, so those targets are not counted.
        loss_mask = (
            hint_valid.unsqueeze(0)
            & mask[:, :-1].bool()
            & mask[:, 1:].bool()
        )
        for name in DECODER_CONDITIONS:
            # ``null`` keeps the real z_t and zeroes the future; ``no_hints``
            # zeroes both, which isolates the z_t channel.
            start = torch.zeros_like(z_hint) if name == "no_hints" else z_hint
            total, count = model.decoder.token_loss(
                start.detach(), hints[name].detach(), labels,
                attention_mask=mask, chunk_size=loss_chunk_size,
                loss_mask=loss_mask,
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
        # What each hint channel adds: z_t over nothing, the committed future
        # over a real z_t alone.
        "z_t_use": ce["no_hints"] - ce["null"],
        "future_use": ce["null"] - ce["pi_argmax"],
        "commit_headroom": ce["pi_argmax"] - ce["hindsight"],
        "specificity": {
            "shuffled_minus_hindsight": ce["shuffled"] - ce["hindsight"],
            "pi_argmax_minus_hindsight": ce["pi_argmax"] - ce["hindsight"],
        },
    }
def _verdict(name, metric, value, threshold, comparison) -> dict:
    value, threshold = float(value), float(threshold)
    ok = value > threshold if comparison == ">" else value < threshold
    return {
        "name": name, "metric": metric, "value": value, "threshold": threshold,
        "comparison": comparison, "ok": bool(ok),
    }
def _verdicts(metrics) -> list[dict]:
    """Section 7: pass/fail checks, each naming its metric and threshold."""
    heads, policies = metrics["heads"], metrics["policies"]
    decoder, representation, rollouts = (
        metrics["decoder"], metrics["representation"], metrics["rollouts"]
    )
    verdicts = [] if heads["pi_gain"] is None else [
        _verdict("pi_uses_z_t", "heads.pi_gain", heads["pi_gain"], 0.0, ">"),
    ]
    verdicts += [
        _verdict(
            "candidates_beat_mean", "policies.hindsight_gain_over_mu",
            policies["hindsight_gain_over_mu"], 0.01, ">",
        ),
        _verdict(
            "decoder_uses_commitment", "decoder.hint_use", decoder["hint_use"], 0.05, ">"
        ),
        _verdict(
            "mu_calibrated", "abs(heads.mu_calibration_error)",
            abs(heads["mu_calibration_error"]), 0.05, "<",
        ),
        _verdict(
            "encoder_center", "representation.visreg.center",
            representation["visreg"]["center"], 0.05, "<",
        ),
        _verdict(
            "encoder_effective_rank", "representation.effective_rank",
            representation["effective_rank"], 15.0, ">",
        ),
    ]
    if representation["periodic_rebound"] is not None:
        verdicts.append(
            _verdict(
                "no_periodic_shortcut", "representation.periodic_rebound",
                representation["periodic_rebound"], REBOUND_THRESHOLD, "<",
            )
        )
    for depth in range(2, rollouts["depths"] + 1):
        verdicts.append(
            _verdict(
                f"committed_on_manifold_depth_{depth}",
                f"rollouts.manifold.pi_sample[depth={depth}] - "
                f"rollouts.manifold.mu[depth={depth}]",
                rollouts["manifold"]["pi_sample"][depth - 1]
                - rollouts["manifold"]["mu"][depth - 1],
                0.0, "<",
            )
        )
    # Best-of-K scores whole rollouts (mean error over depths 1..i), so
    # compare it with the μ path scored the same way.
    k = max(rollouts["best_of_k"], key=int)
    depth = rollouts["depths"]
    mu = _mean_error_rows(rollouts)["mu"][depth - 1]
    verdicts.append(
        _verdict(
            f"best_of_{k}_beats_mu",
            f"rollouts.best_of_k[{k}][depth={depth}] - "
            f"mean(rollouts.error.mu[depth=1..{depth}])",
            rollouts["best_of_k"][k][depth - 1] - mu,
            0.0, "<",
        )
    )
    return verdicts

def _mean_error_rows(rollouts) -> dict:
    """The μ path and hindsight error rows as mean error over depths 1..i."""
    return {
        name: _mean_rollout_error(torch.tensor(rollouts["error"][name])).tolist()
        for name in ("mu", "hindsight")
    }

@torch.no_grad()
def collect_metrics(
    model, batches, bank, *, depths=DEFAULT_DEPTHS,
    rollout_positions=DEFAULT_ROLLOUT_POSITIONS, seed=0,
    loss_chunk_size: int | None = DEFAULT_LOSS_CHUNK_SIZE, bank_chunk_size=None,
    best_of=DEFAULT_BEST_OF,
) -> dict:
    """Measure ``model`` on ``batches`` against a bank of real latents.

    ``loss_chunk_size=None`` uses the checkpoint's ``DecoderConfig``.
    ``best_of`` lists the K values of the best-of-K rollout error.
    """
    if depths < 1 or rollout_positions < 1:
        raise ValueError("depths and rollout_positions must be at least 1")
    requested, best_of = best_of, tuple(sorted({int(k) for k in best_of}))
    if not best_of or best_of[0] < 1:
        raise ValueError(f"best_of must be positive K values; got {requested!r}")
    model.eval()
    batches = list(batches)
    if not batches:
        raise ValueError("at least one query batch is required")
    horizon = model.pred_horizon
    for labels, mask in batches:
        if mask is None:
            raise ValueError("measure needs attention masks (1 = real token)")
        if labels.shape[1] <= horizon:
            raise ValueError(f"query length {labels.shape[1]} must exceed pred_horizon {horizon}")
    device = next(model.parameters()).device
    batches = [(labels.to(device), mask.to(device)) for labels, mask in batches]
    bank = F.normalize(bank.float().to(device), dim=-1)
    loss_chunk_size = loss_chunk_size or model.config.decoder.loss_chunk_size
    latents = [model.encode(labels, attention_mask=mask) for labels, mask in batches]
    sections = {
        "representation": _representation_section(model, batches, latents),
        "loss": _loss_section(model, batches),
        "heads": _heads_section(model, batches, latents),
        "policies": _policies_section(model, batches, latents, seed=seed),
        "rollouts": _rollouts_section(
            model, batches, latents, bank, depths=depths, positions=rollout_positions,
            seed=seed, bank_chunk_size=bank_chunk_size, best_of=best_of,
        ),
        "decoder": _decoder_section(model, batches, latents, seed=seed,
                                    loss_chunk_size=loss_chunk_size),
    }
    context = {
        "checkpoint": None, "device": None,
        "examples": int(sum(labels.shape[0] for labels, _ in batches)),
        "max_length": int(max(labels.shape[1] for labels, _ in batches)),
        "tokens": int(sum(int(mask.sum()) for _, mask in batches)),
        "pred_horizon": int(horizon),
        "candidates": int(model.config.predictor.prediction_count),
        "depths": int(depths),
        "best_of": list(best_of),
        "rollout_positions": int(sections["rollouts"]["positions"]),
        "bank_size": int(bank.shape[0]), "latent_dim": int(bank.shape[1]),
        "seed": int(seed),
    }
    metrics = {"context": context, **sections}; metrics["verdicts"] = _verdicts(metrics)
    return metrics

def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"

def _table(rows, headers) -> None:
    widths = [max([len(str(h))] + [len(str(r[i])) for r in rows]) for i, h in enumerate(headers)]
    for row in (headers, *rows):
        print("  ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))

def _pairs(title, rows) -> None:
    print(f"\n--- {title} ---")
    for label, value in rows:
        print(f"{label + ':':<26}{value}")

def print_report(metrics: dict) -> None:
    """Print every metric and verdict in ``collect_metrics``'s dict."""
    c, r = metrics["context"], metrics["representation"]
    print("=" * 78)
    print(f"measure: {c.get('checkpoint') or '(in-memory model)'}")
    print(
        f"device: {c.get('device') or '?'} | examples: {c['examples']} x "
        f"{c['max_length']} | tokens: {c['tokens']:,} | k: {c['pred_horizon']} | "
        f"candidates: {c['candidates']} | bank: {c['bank_size']:,}"
    )
    print("=" * 78)

    _pairs("representation", [
        ("per-dim std mean", _fmt(r["per_dim_std"])),
        ("mean vector norm", _fmt(r["mean_vector_norm"])),
        ("effective rank", f"{_fmt(r['effective_rank'])} / {c['latent_dim']}  "
                           f"({r['latents']:,} latents)"),
        ("cos(z_t, z_t+k)", _fmt(r["temporal_cos"])),
        ("periodic rebound", f"{_fmt(r['periodic_rebound'])}  "
                             "(cos at k minus the lowest cos below k)"),
        ("fast share", f"{_fmt(r['fast_share'])}  "
                       "(within-document energy faster than k/2)"),
        ("VISReg", " | ".join(f"{n} {_fmt(v)}" for n, v in r["visreg"].items())),
    ])
    lags = list(r["lag_cos"].items())
    print("lag profile, mean cos(z_t, z_t+lag):")
    for start in range(0, len(lags), 10):
        row = lags[start : start + 10]
        print("  lag " + "".join(f"{lag:>8}" for lag, _ in row))
        print("  cos " + "".join(f"{_fmt(value):>8}" for _, value in row))

    loss = metrics["loss"]
    print("\n--- jepa loss ---")
    _table([(n, _fmt(t["weight"]), _fmt(t["value"]), _fmt(t["weighted"]))
            for n, t in loss["terms"].items()], ("term", "weight", "value", "weighted"))
    print(f"{'total':<12}{_fmt(loss['total']):>24}")
    print(f"pi floor: H(r)={_fmt(loss['floor']['assignment_entropy'])}  "
          f"KL(r||pi)={_fmt(loss['floor']['kl_to_pi'])}  "
          "(only the KL part is learnable)")

    h = metrics["heads"]
    _pairs("heads", [
        ("candidate pair cos", _fmt(h["candidate_pair_cos"])),
        ("|delta| / |z|", _fmt(h["delta_ratio"])),
        ("hindsight shares", " ".join(_fmt(x) for x in h["hindsight_shares"])),
        ("live slots", f"{'n/a' if h['live_slots'] is None else h['live_slots']} / {len(h['hindsight_shares'])}  "
                       f"(effective {_fmt(h['effective_slots'])})"),
        ("pi NLL", f"{_fmt(h['pi_nll'])}  vs constant {_fmt(h['constant_nll'])}  "
                   f"(gain {_fmt(h['pi_gain'])} nats)"),
        ("pi accuracy", f"{_fmt(h['pi_accuracy'])}  vs majority "
                        f"{_fmt(h['majority_accuracy'])}"),
        ("|mu|", _fmt(h["mu_norm"])),
        ("mu distance", f"{_fmt(h['mu_distance_measured'])} measured vs "
                       f"{_fmt(h['mu_distance_predicted'])} predicted "
                       "(2 - 2|mu|)"),
    ])

    p = metrics["policies"]
    print("\n--- policies (depth 1, distance to the real z_t+k) ---")
    _table([(n, _fmt(p[n])) for n in ("copy", "hindsight", "pi_argmax",
                                      "pi_sample", "mu_safe", "mu")]
           + [(f"fixed slot {p['fixed_slot']['index']}", _fmt(p["fixed_slot"]["distance"]))],
           ("commit rule", "distance"))
    print(f"hindsight gain over mu: {_fmt(p['hindsight_gain_over_mu'])}")
    print(f"commit headroom (pi-argmax - hindsight): {_fmt(p['commit_headroom'])}")

    ro = metrics["rollouts"]
    print(f"\n--- rollouts ({ro['positions']} starts, k={ro['stride']}) ---")
    print("rollout error (distance to the real latent):")
    _table([(d, *(_fmt(ro["error"][n][d - 1]) for n in ROLLOUT_PATHS))
            for d in range(1, ro["depths"] + 1)], ("depth", *ROLLOUT_PATHS))
    print("best-of-K pi-sample rollouts (mean error over depths 1..i, best of K):")
    mean_rows = _mean_error_rows(ro)
    depth_headers = [f"d={d}" for d in range(1, ro["depths"] + 1)]
    _table([(f"K={k}", *map(_fmt, row)) for k, row in ro["best_of_k"].items()]
           + [(name, *map(_fmt, row)) for name, row in mean_rows.items()],
           ("path", *depth_headers))
    print("manifold distance (nearest real latent in the bank):")
    _table([(d, *(_fmt(ro["manifold"][n][d - 1]) for n in (*ROLLOUT_PATHS, "real")))
            for d in range(1, ro["depths"] + 1)],
           ("depth", *ROLLOUT_PATHS, "real"))

    dec = metrics["decoder"]
    print(f"\n--- decoder (token-weighted CE over {dec['tokens']:,} tokens) ---")
    _table([(n, _fmt(v)) for n, v in dec["ce"].items()], ("hint", "CE"))
    print(f"hint use (shuffled - pi-argmax):       {_fmt(dec['hint_use'])}")
    print(f"z_t use (no_hints - null):             {_fmt(dec['z_t_use'])}")
    print(f"future use (null - pi-argmax):         {_fmt(dec['future_use'])}")
    print(f"commit headroom (pi-argmax - hindsight): {_fmt(dec['commit_headroom'])}")
    print(f"specificity: shuffled-hindsight "
          f"{_fmt(dec['specificity']['shuffled_minus_hindsight'])}  vs "
          f"pi-argmax-hindsight {_fmt(dec['specificity']['pi_argmax_minus_hindsight'])}")

    _pairs("verdicts", [
        (f"[{'PASS' if v['ok'] else 'FAIL'}] {v['name']}",
         f"{v['metric']} = {_fmt(v['value'])} ({v['comparison']} {_fmt(v['threshold'])})")
        for v in metrics["verdicts"]
    ])
    print("\n--- how to read this ---")
    for line in (
        "representation: std near 1 and effective rank > 15 = no collapse;",
        "                cos(z_t, z_t+k) 0.5-0.85 healthy, >0.95 too smooth.",
        "                periodic rebound > 0 = the latent dips and returns by",
        "                lag k, a periodic code the prediction loss cannot see.",
        "loss:           total is the weighted sum of the terms; H(r) is fixed",
        "                by the targets, so only KL(r||pi) can improve.",
        "heads:          pi gain > 0 = pi predicts the closest candidate from z_t.",
        "policies:       hindsight needs the real future; hindsight gain over mu",
        "                is what candidates buy, commit headroom is the rest.",
        "rollouts:       error is distance to the real z at t+i*k; manifold is",
        "                distance to the nearest real latent. The claim: at",
        "                depths >= 2, pi-sample stays closer to the manifold than",
        "                the mu path (compare with the real row).",
        "                best-of-K should fall from near mu toward hindsight as",
        "                K grows if the committed futures cover the real one.",
        "decoder:        hint use > 0 means the decoder uses the committed future.",
        "                null keeps the real z_t; future use is what the",
        "                committed future adds beyond it.",
        "verdicts:       PASS/FAIL on the checks above, each naming its metric.",
    ):
        print(line)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m jepa measure", description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    parser.add_argument(
        "--fineweb-dir", type=Path, default=FINEWEB_DIR,
        help="Prepared chunks: validation is queried, train supplies the bank.",
    )
    parser.add_argument("--num-examples", type=int, default=DEFAULT_QUERY_EXAMPLES)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--bank-size", type=int, default=DEFAULT_BANK_SIZE,
        help="Real train-split latents in the manifold bank.",
    )
    parser.add_argument(
        "--rollout-positions", type=int, default=DEFAULT_ROLLOUT_POSITIONS,
        help="Sampled start positions for the rollout table.",
    )
    parser.add_argument("--depths", type=int, default=DEFAULT_DEPTHS)
    parser.add_argument(
        "--best-of", type=int, nargs="+", default=list(DEFAULT_BEST_OF),
        help="K values for the best-of-K pi-sample rollout error.",
    )
    parser.add_argument(
        "--loss-chunk-size", type=int, default=DEFAULT_LOSS_CHUNK_SIZE,
        help="Decoder positions per lm_head chunk (default: checkpoint config).",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="Write every metric as JSON.")
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
        validation, tokenizer, num_examples=args.num_examples,
        max_length=args.max_length, batch_size=args.batch_size, seed=args.seed,
    )
    bank = build_bank(
        model, train_dataset, tokenizer, size=args.bank_size,
        max_length=args.max_length, batch_size=args.batch_size, seed=args.seed,
    )
    metrics = collect_metrics(
        model, batches, bank, depths=args.depths,
        rollout_positions=args.rollout_positions, seed=args.seed,
        loss_chunk_size=args.loss_chunk_size, best_of=args.best_of,
    )
    metrics["context"]["checkpoint"] = str(args.model)
    metrics["context"]["device"] = str(device)
    print_report(metrics)
    if args.json is not None:
        args.json.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nwrote {args.json}")

if __name__ == "__main__":
    main()
