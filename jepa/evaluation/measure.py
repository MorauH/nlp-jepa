"""The paper's numbers: representation, commitment, rollouts and decoding.

    python -m jepa measure --model runs/jepa
    python -m jepa measure --model runs/jepa --json runs/jepa/metrics.json

Queries are validation chunks; the manifold bank samples real encoder latents
from the train split (whole documents are held out, so they do not overlap),
and the next-block probe is fit on other train-split chunks.

The horizon is random (``docs/adr/0003-random-hidden-horizon.md``), so everything
that compares a prediction with the future is scored at a fixed set of
**evaluation horizons** and averaged.  Latents, candidates and every rollout
path except hindsight do not depend on k, so each section computes them once
and scores them against each (the loss section calls ``jepa_loss`` per k).
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
    Futures,
    Hindsight,
    MuSafe,
    PiArgmax,
    PiSample,
    distance,
    pairwise_distance,
    rollout,
)
from ..models.jepa import JepaModel, blur, build_block_hints, hint_noise_scale
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
    "pi_argmax", "pi_sample", "mu_safe", "hindsight", "real", "real_noisy",
    "shuffled", "null", "no_hints",
)
# Latents the next-block probe reads, and the train-split chunks it is fit on.
PROBE_FEATURES = ("z_t", "mu", "pi_argmax", "hindsight")
# Latents vary mostly between documents, so the probe needs many of them:
# with 256 it overfits the fit documents (negative R² on random-joint-seed1),
# with 2048 it is stable for ridges from 0.1 to 1.
DEFAULT_PROBE_EXAMPLES = 2048
# Ridge penalty, relative to the mean feature variance times the row count.
PROBE_RIDGE = 0.1
POLICY_NAMES = ("copy", "hindsight", "pi_argmax", "pi_sample", "mu_safe", "mu")
# Period of the cycle the fixed-k encoders learned (ADR 0003).
OBSERVED_PERIOD = 10
# Fixed, not multiples of k, so runs with different horizons compare.
LONG_LAGS = (40, 100, 200, 400)
FAST_SHARE_WINDOW = OBSERVED_PERIOD
# Positions the document start no longer dominates.
STEADY_STATE_START = 80
DOCUMENT_SHARE_START = 20
# A drifting latent has rebound <= 0; the period-10 shortcut of the first
# two-stage run scored about 1.0.  The margin allows for sampling noise.
REBOUND_THRESHOLD = 0.05

def evaluation_horizons(horizon_range) -> list[int]:
    """The k values measurement scores: no two alike modulo the observed period.

    A short range is scored in full.  A longer one is sampled evenly around its
    centre, with the smallest step coprime with the period that covers it in at
    most one period's worth of values.  Steps of 5 would always sit on the
    observed cycle.
    """
    lo, hi = horizon_range
    span = hi - lo
    if span < OBSERVED_PERIOD:
        return list(range(lo, hi + 1))
    step = 3
    while math.gcd(step, OBSERVED_PERIOD) > 1 or span // step >= OBSERVED_PERIOD:
        step += 1
    count = span // step + 1
    first = lo + (span - step * (count - 1)) // 2
    return [first + step * i for i in range(count)]

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

def _proposals(model, batches, latents, horizons):
    """Yield ``(z, mask, futures)`` per batch: the predictor runs once per batch.

    ``futures`` covers every base position some horizon can use.  Batches
    without any such position are skipped.
    """
    shortest = min(horizons)
    for (_, mask), z in zip(batches, latents):
        if _prediction_mask(mask, shortest).any():
            yield z, mask, model.predictor(z[:, :-shortest])

def _bases(futures, z, horizon: int) -> Futures:
    """The proposals at the base positions ``horizon`` can use."""
    return Futures(*(x[:, : z.shape[1] - horizon] for x in futures))

def _steady(mask, horizon: int):
    """``_prediction_mask`` restricted to the steady state."""
    valid = _prediction_mask(mask, horizon)
    valid[:, :STEADY_STATE_START] = False
    return valid

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

def max_rebound(profile, longest: int) -> float | None:
    """Largest return of similarity at any lag up to ``longest``.

    Over every lag L in 2..longest: cos at L minus the lowest cos at a shorter
    lag, maximized over L.  A latent that drifts loses similarity with lag, so
    its rebound is <= 0.  A latent with a period dips and returns, whether or
    not the period divides a horizon.  ``None`` without two usable lags.
    """
    values = [profile.get(lag) for lag in range(1, longest + 1)]
    lowest, rebound = None, None
    for value in (v for v in values if v is not None):
        if lowest is None:
            lowest = value
            continue
        rise = value - lowest
        rebound = rise if rebound is None else max(rebound, rise)
        lowest = min(lowest, value)
    return rebound

def document_share(latents, masks, start: int = DOCUMENT_SHARE_START) -> float | None:
    """Between-document share of the latent variance (0..1).

    A latent that is constant within a document stays free to predict at any
    horizon.  Positions before ``start`` are left out: every document starts
    alike.  A query chunk counts as a document.  ``None`` with fewer than two
    documents longer than ``start`` (or no variance at all).
    """
    counts, sums, square = [], [], 0.0
    for z, mask in zip(latents, masks):
        for row, length in enumerate(int(n) for n in mask.sum(dim=-1).tolist()):
            if length <= start:
                continue
            x = z[row, start:length].double()
            counts.append(x.shape[0])
            sums.append(x.sum(dim=0))
            square += (x**2).sum().item()
    if len(counts) < 2:
        return None
    sums = torch.stack(sums)
    n = torch.tensor(counts, dtype=sums.dtype, device=sums.device)
    grand = (sums.sum(dim=0) ** 2).sum() / n.sum()
    between = ((sums**2).sum(dim=-1) / n).sum() - grand
    total = square - grand
    return _value(between / total) if total > 0 else None

def slot_horizon_information(counts) -> float:
    """Mutual information in nats between slot and horizon.

    ``counts`` is ``(horizons, slots)``: how often each slot is the hindsight
    commit at each horizon.  Near its maximum the slots are phase copies, one
    per k, rather than different futures.  Near 0 no slot is tied to a distance
    from ``z_t``.  That does not rule out slots tied to an absolute phase: such
    a slot wins by ``t + k``, and averaging over ``t`` hides it.
    """
    joint = counts.double() / counts.sum()
    independent = joint.sum(dim=1, keepdim=True) * joint.sum(dim=0, keepdim=True)
    terms = joint * (joint.clamp_min(1e-300) / independent.clamp_min(1e-300)).log()
    return max(0.0, float(terms.sum()))

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

def _representation_section(model, batches, latents, horizons) -> dict:
    """Section 1: no-collapse checks on the encoder latents."""
    longest = model.config.horizon_range[1]
    pooled, seen = [], 0
    token_count = 0
    latent_sum = 0.0
    visreg_sums = {name: 0.0 for name in model.visreg.weights}
    for (_, mask), z in zip(batches, latents):
        valid = mask.bool()
        tokens = int(valid.sum())
        token_count += tokens
        latent_sum = latent_sum + z[valid].double().sum(dim=0)
        if seen < REPRESENTATION_LATENTS:
            take = z[valid].float()[: REPRESENTATION_LATENTS - seen]
            pooled.append(take)
            seen += take.shape[0]
        for name, value in model.visreg(z, mask=mask).terms.items():
            visreg_sums[name] += _value(value) * tokens
    masks = [mask for _, mask in batches]
    profile = lag_profile(latents, masks, sorted({*range(1, longest + 1), *LONG_LAGS}))
    at_horizons = [profile[k] for k in horizons if profile[k] is not None]
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
        "temporal_cos": sum(at_horizons) / len(at_horizons) if at_horizons else 0.0,
        # String keys: the dict is written to JSON as is.
        "lag_cos": {str(lag): value for lag, value in profile.items()},
        "max_rebound": max_rebound(profile, longest),
        "fast_share": fast_share(latents, masks, window=FAST_SHARE_WINDOW),
        "document_share": document_share(latents, masks),
        # VISReg as training sees it, one measure batch at a time.  Its center
        # term grows with document share / batch size, so the verdict reads
        # the center of every query latent pooled instead.
        "visreg": {name: visreg_sums[name] / max(1, token_count) for name in visreg_sums},
        "pooled_center": _value(((latent_sum / max(1, token_count)) ** 2).mean()),
    }

def _loss_section(model, batches, horizons) -> dict:
    """Section 2: every JEPA term, its weight, and its contribution."""
    weights = model.loss_weights
    term_sums, diagnostic_sums, count = {}, {}, 0
    for labels, mask in batches:
        for horizon in horizons:
            valid = int(_prediction_mask(mask, horizon).sum())
            if valid == 0:
                continue
            out = model.jepa_loss(labels, attention_mask=mask, horizon=horizon)
            for sums, values in (
                (term_sums, out.terms), (diagnostic_sums, out.diagnostics)
            ):
                for name, value in values.items():
                    sums[name] = sums.get(name, 0.0) + _value(value) * valid
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

def _shares(counts):
    """Share of the hindsight commits per slot, and their entropy in nats."""
    pooled = counts.sum(dim=0).float()
    shares = pooled / pooled.sum().clamp_min(1)
    return shares, float(-(shares * shares.clamp_min(1e-12).log()).sum())

def _slot_usage(counts, matched) -> dict:
    """How many slots the hindsight commit uses, and whether they encode k.

    ``counts`` and ``matched`` are ``(horizons, slots)``.  ``matched`` only has
    positions every horizon can use, so a slot that follows the position cannot
    look like one that follows k.  Everything is n/a for a single slot.
    """
    horizons, slots = counts.shape
    if slots == 1:
        return dict.fromkeys(("live_slots", "effective_slots", "slot_horizon_mi"))
    shares, entropy = _shares(counts)
    return {
        "live_slots": int((shares > 1.0 / (4 * slots)).sum()),
        "effective_slots": math.exp(entropy),
        "slot_horizon_mi": slot_horizon_information(matched)
        if horizons > 1 and matched.sum() > 0 else None,
    }

def _heads_section(model, batches, latents, horizons) -> dict:
    """Section 3: candidates, π, μ and their calibration."""
    prediction_count = model.config.predictor.prediction_count
    count = 0
    pair_sim_sum = delta_sum = z_norm_sum = 0.0
    pi_nll_sum = mu_norm_sum = mu_distance_sum = 0.0
    pair_count = correct = 0

    def slot_counts():
        return torch.zeros(len(horizons), prediction_count, dtype=torch.long)

    hindsight_counts, matched_counts = slot_counts(), slot_counts()
    steady_counts, steady_matched = slot_counts(), slot_counts()
    last = max(horizons)
    for z, mask, proposals in _proposals(model, batches, latents, horizons):
        similarity = (
            proposals.candidate_similarity() if prediction_count > 1 else None
        )
        for row, horizon in enumerate(horizons):
            valid = _prediction_mask(mask, horizon)
            if not valid.any():
                continue
            futures = _bases(proposals, z, horizon)
            z_t, future = z[:, :-horizon], z[:, horizon:]
            future_valid, z_valid = future[valid].float(), z_t[valid].float()
            n = int(valid.sum())
            count += n
            # The hindsight commit decides the reference slot and the π target.
            index = futures.commit(Hindsight(), future).index
            hindsight_index = index[valid]
            if prediction_count > 1:
                sim = similarity[:, : z_t.shape[1]][valid]  # (N, P, P)
                off_diagonal = 1.0 - torch.eye(
                    prediction_count, device=sim.device, dtype=sim.dtype
                )
                pair_sim_sum += (sim * off_diagonal).sum().item()
                pair_count += n * prediction_count * (prediction_count - 1)
            delta_sum += (
                futures.candidates[valid].float() - z_valid.unsqueeze(-2)
            ).norm(dim=-1).sum().item()
            z_norm_sum += z_valid.norm(dim=-1).sum().item()
            # Positions the last horizon can use are usable at every one.
            shared = index[:, : z.shape[1] - last]
            steady = _steady(mask, horizon)
            for counts, chosen in (
                (hindsight_counts, hindsight_index),
                (matched_counts, shared[_prediction_mask(mask, last)]),
                (steady_counts, index[steady]),
                (steady_matched, shared[_steady(mask, last)]),
            ):
                counts[row] += torch.bincount(chosen, minlength=prediction_count).cpu()
            log_pi = F.log_softmax(futures.pi[valid].float(), dim=-1)
            pi_nll_sum += -log_pi.gather(-1, hindsight_index[:, None]).sum().item()
            correct += (log_pi.argmax(dim=-1) == hindsight_index).sum().item()
            mu = futures.mu[valid].float()
            mu_norm_sum += mu.norm(dim=-1).sum().item()
            mu_distance_sum += distance(mu, future_valid).sum().item()
    if count == 0:
        raise ValueError("no valid prediction positions in the query batches")
    shares, entropy = _shares(hindsight_counts)
    pi_nll = pi_nll_sum / count
    mu_norm = mu_norm_sum / count
    mu_distance = mu_distance_sum / count
    # If μ is calibrated, E[d(μ̂, ŷ)] = 2 - 2|μ| (μ is a mean of unit futures,
    # not renormalized).
    mu_distance_predicted = 2.0 - 2.0 * mu_norm
    choices = {
        "candidate_pair_cos": pair_sim_sum / pair_count if pair_count else None,
        "pi_nll": pi_nll,
        "constant_nll": entropy,
        "pi_gain": entropy - pi_nll,
        "pi_accuracy": correct / count,
        "majority_accuracy": float(shares.max()),
        # What MI could reach at most: it is bounded by both entropies.
        "slot_horizon_mi_max": math.log(min(len(horizons), prediction_count))
        if len(horizons) > 1 else None,
    }
    if prediction_count == 1:  # nothing to choose between: n/a, not a number
        choices = dict.fromkeys(choices)
    return {
        **choices,
        **_slot_usage(hindsight_counts, matched_counts),
        "steady_state": {
            "from_position": STEADY_STATE_START,
            **_slot_usage(steady_counts, steady_matched),
        } if steady_counts.sum() > 0 else None,
        "delta_ratio": (delta_sum / (count * prediction_count)) / max(z_norm_sum / count, 1e-12),
        "hindsight_shares": [float(x) for x in shares.tolist()],
        "mu_norm": mu_norm,
        "mu_distance_measured": mu_distance,
        "mu_distance_predicted": mu_distance_predicted,
        "mu_calibration_error": mu_distance - mu_distance_predicted,
    }

class _PolicySums:
    """Per-horizon distance sums of every commit rule, over one set of positions."""

    def __init__(self, horizons, prediction_count: int):
        self.counts = dict.fromkeys(horizons, 0)
        self.sums = {k: dict.fromkeys(POLICY_NAMES, 0.0) for k in horizons}
        self.slots = {
            k: torch.zeros(prediction_count, dtype=torch.float64) for k in horizons
        }

    def add(self, horizon, named, slot_distances) -> None:
        """``named`` is one distance per position per rule, slots ``(N, P)``."""
        self.counts[horizon] += slot_distances.shape[0]
        for name, values in named.items():
            self.sums[horizon][name] += values.sum().item()
        self.slots[horizon] += slot_distances.double().cpu().sum(dim=0)

    def rows(self) -> dict | None:
        """The mean over horizons, with one row per horizon; None if empty."""
        used = [k for k, count in self.counts.items() if count]
        if not used:
            return None
        slots = {k: self.slots[k] / self.counts[k] for k in used}
        # One slot for every horizon: the best single one on average.
        fixed = int(torch.stack(list(slots.values())).mean(dim=0).argmin())

        def row(k):
            values = {n: self.sums[k][n] / self.counts[k] for n in POLICY_NAMES}
            return {
                **values,
                "hindsight_gain_over_mu": values["mu"] - values["hindsight"],
                "commit_headroom": values["pi_argmax"] - values["hindsight"],
            }, {"index": fixed, "distance": slots[k][fixed].item()}

        rows = {k: row(k) for k in used}

        def mean(values):
            return sum(values) / len(used)

        return {
            **{name: mean([rows[k][0][name] for k in used]) for name in rows[used[0]][0]},
            "fixed_slot": {
                "index": fixed,
                "distance": mean([rows[k][1]["distance"] for k in used]),
            },
            # String keys: the dict is written to JSON as is.
            "by_horizon": {
                str(k): {**rows[k][0], "fixed_slot": rows[k][1]} if k in rows else None
                for k in self.counts
            },
        }

def _policies_section(model, batches, latents, horizons, *, seed: int) -> dict:
    """Section 4: distance-to-real at depth 1 for every commit rule.

    Each number is the mean over the evaluation horizons, which are reported
    one by one too, for all positions and for the steady state.
    """
    prediction_count = model.config.predictor.prediction_count
    generator = _generator(latents[0].device, seed + 1)
    everywhere = _PolicySums(horizons, prediction_count)
    steady = _PolicySums(horizons, prediction_count)
    for z, mask, proposals in _proposals(model, batches, latents, horizons):
        # These rules do not see the future, so they commit once for every k.
        committed = {
            name: proposals.commit(rule).index
            for name, rule in (
                ("pi_argmax", PiArgmax()),
                ("pi_sample", PiSample(generator=generator)),
                ("mu_safe", MuSafe()),
            )
        }
        for horizon in horizons:
            valid = _prediction_mask(mask, horizon)
            if not valid.any():
                continue
            futures = _bases(proposals, z, horizon)
            z_t, future = z[:, :-horizon], z[:, horizon:]
            distances = futures.distances(future)[valid]
            future_valid = future[valid].float()
            named = {
                "copy": distance(z_t[valid].float(), future_valid),
                "hindsight": distances.min(dim=-1).values,
                "mu": distance(futures.mu[valid].float(), future_valid),
            }
            for name, index in committed.items():
                index = index[:, : z_t.shape[1]][valid]
                named[name] = distances.gather(-1, index[:, None]).squeeze(-1)
            everywhere.add(horizon, named, distances)
            keep = _steady(mask, horizon)[valid]
            steady.add(
                horizon, {n: v[keep] for n, v in named.items()}, distances[keep]
            )
    rows, steady_rows = everywhere.rows(), steady.rows()
    if rows is None:
        raise ValueError("no valid prediction positions in the query batches")
    if steady_rows is not None:
        steady_rows = {"from_position": STEADY_STATE_START, **steady_rows}
    return {**rows, "steady_state": steady_rows}

def _rollout_starts(batches, longest, steps, limit, generator):
    """Sample ``(batch, row, t)`` starts where ``t + steps * longest`` is valid.

    ``longest`` is the longest horizon, so every horizon uses the same starts.
    """
    candidates = []
    for index, (_, mask) in enumerate(batches):
        last = mask.sum(dim=-1) - 1 - steps * longest
        for row, last_t in enumerate(last.tolist()):
            candidates.extend((index, row, t) for t in range(int(last_t) + 1))
    if not candidates:
        raise ValueError(
            f"no rollout start has t + depths * {longest} (the longest horizon) valid; "
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

    ``targets`` is ``(horizons, N, depths, D)``; the error is the mean over
    horizons of the best rollout at each.  Draws ``max(best_of)`` independent
    π-sample rollouts in one batch; best-of-K keeps the first K, so the draws
    are nested and the result is non-increasing in K.  At depth i each rollout
    is scored by its mean error over depths 1..i, so the minimum is one whole
    rollout, never a mix of rollouts picked per depth.
    """
    draws = max(best_of)
    starts = z_start.unsqueeze(0).expand(draws, *z_start.shape)
    paths = rollout(
        predictor, starts, PiSample(generator=generator), targets.shape[-2]
    ).future
    scores = torch.stack([  # (horizons, K, N, depths)
        _mean_rollout_error(distance(paths, target.unsqueeze(0))) for target in targets
    ])
    return {k: scores[:, :k].min(dim=1).values.mean(dim=0) for k in best_of}

def _rollouts_section(
    model, batches, latents, bank, horizons, *, depths, positions, seed,
    bank_chunk_size, best_of,
) -> dict:
    """Section 5: rollout error, best-of-K error and manifold distance per depth.

    Depth i is scored at ``t + i * k`` and averaged over the evaluation
    horizons.  Only the hindsight path depends on k.
    """
    device = latents[0].device
    sampled, groups = _rollout_starts(
        batches, model.config.horizon_range[1], depths, positions,
        _generator(torch.device("cpu"), seed),
    )
    sample_generator = _generator(device, seed + 2)
    best_generator = _generator(device, seed + 5)

    def zeros(names):
        return {name: torch.zeros(depths, dtype=torch.float64) for name in names}

    best_sums = zeros(best_of)
    error_sums = zeros(ROLLOUT_PATHS)
    manifold_sums = zeros((*ROLLOUT_PATHS, "real"))

    def add(sums, name, scores):
        """Add ``(..., N, depths)`` scores, averaged over any horizon axis."""
        per_start = scores.reshape(-1, *scores.shape[-2:]).mean(dim=0)
        sums[name] += per_start.double().sum(dim=0).cpu()

    def manifold(points):
        flat = points.reshape(-1, points.shape[-1])
        return _manifold_scores(flat, bank, bank_chunk_size).view(points.shape[:-1])

    for index, pairs in groups.items():
        z = latents[index]
        rows = torch.tensor([row for row, _ in pairs], dtype=torch.long, device=device)
        starts = torch.tensor([t for _, t in pairs], dtype=torch.long, device=device)
        z_start = z[rows, starts]
        targets = torch.stack([  # (horizons, N, depths, D)
            torch.stack(
                [z[rows, starts + (depth + 1) * k] for depth in range(depths)], dim=1
            )
            for k in horizons
        ])
        paths = {
            "hindsight": rollout(
                model.predictor, z_start.expand(len(horizons), *z_start.shape),
                Hindsight(), depths, targets=targets,
            ).future,
            "pi_argmax": rollout(model.predictor, z_start, PiArgmax(), depths).future,
            "pi_sample": rollout(
                model.predictor, z_start, PiSample(generator=sample_generator), depths
            ).future,
            "mu": mu_path(model.predictor, z_start, depths),
        }
        for name, path in paths.items():
            add(error_sums, name, distance(path, targets))
            add(manifold_sums, name, manifold(path))
        add(manifold_sums, "real", manifold(targets))
        # Same memory bound as the single rollouts: at most one group of
        # starts per predictor call, counting the K draws.
        chunk = max(1, z_start.shape[0] // max(best_of))
        for i in range(0, z_start.shape[0], chunk):
            best = _best_of_k(
                model.predictor, z_start[i : i + chunk], targets[:, i : i + chunk],
                best_of, best_generator,
            )
            for k, scores in best.items():
                add(best_sums, k, scores)

    def means(sums):
        return {str(name): (total / sampled).tolist() for name, total in sums.items()}

    return {
        "positions": sampled,
        "depths": depths,
        "horizons": list(horizons),
        "error": means(error_sums),
        "best_of_k": means(best_sums),
        "manifold": means(manifold_sums),
    }

def _decoder_section(
    model, batches, latents, horizons, *, seed, loss_chunk_size
) -> dict:
    """Section 6: token-weighted CE per hint condition.

    Blocks of k tokens for each evaluation horizon, pooled over all of them.
    Every condition uses the same mask: targets whose block runs past the end
    of the sequence have no committed future and are excluded.  The π-argmax
    row is the inference hint; the hindsight row is teacher-forced and its hint
    carries up to log2(P) bits about the target future from the choice among
    candidates (and may reveal k), so it is a lower bound on CE rather than a leak-free condition.
    """
    device = latents[0].device
    sample_generator = _generator(device, seed + 3)
    shuffle_generator = _generator(device, seed + 4)
    noise_generator = _generator(device, seed + 6)
    totals = {name: 0.0 for name in DECODER_CONDITIONS}
    token_count = 0
    shortest = min(horizons)
    for (labels, mask), z in zip(batches, latents):
        proposals = model.predictor(z[:, :-shortest])
        # Only the hindsight and real hints depend on k.
        committed = {
            "pi_argmax": proposals.commit(PiArgmax()).future,
            "pi_sample": proposals.commit(PiSample(generator=sample_generator)).future,
            "mu_safe": proposals.commit(MuSafe()).future,
        }
        order = torch.randperm(z.shape[0], generator=shuffle_generator, device=device)
        committed["shuffled"] = committed["pi_argmax"][order]
        for horizon in horizons:
            z_hint, real_hint, hint_valid = build_block_hints(z, horizon)
            futures = _bases(proposals, z, horizon)
            hindsight = futures.commit(Hindsight(), z[:, horizon:]).future
            # The real-hint training condition: the real future blurred to
            # this batch's hindsight distance at this k.
            valid = _prediction_mask(mask, horizon)
            hindsight_d = futures.distances(z[:, horizon:]).min(dim=-1).values[valid]
            scale = hint_noise_scale(
                hindsight_d.float().mean() if hindsight_d.numel()
                else z.new_zeros((), dtype=torch.float32)
            )
            noisy = blur(z, scale, generator=noise_generator)
            hints = {
                name: build_block_hints(z, horizon, committed=value)[1]
                for name, value in (*committed.items(), ("hindsight", hindsight))
            }
            hints["real"] = real_hint
            hints["real_noisy"] = build_block_hints(noisy, horizon)[1]
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
        # What decoding a predicted future costs against the real one.
        "render_gap": ce["pi_argmax"] - ce["real"],
        "specificity": {
            "shuffled_minus_hindsight": ce["shuffled"] - ce["hindsight"],
            "pi_argmax_minus_hindsight": ce["pi_argmax"] - ce["hindsight"],
        },
    }


class _BagOfWordsProbe:
    """Ridge regression from a unit latent to the next block's bag of words.

    The target of a row is the next block's token frequencies (counts / k)
    over the vocabulary.  Only sufficient statistics are kept, so neither the
    targets nor the predictions are ever materialized as ``(N, vocab)``.  The
    features are centred, so the unpenalized intercept is the mean bag of
    words, which is also the reference the R² is measured against.
    """

    def __init__(self, dim: int, vocab: int, device):
        self.count = 0
        self.sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
        self.xx = torch.zeros(dim, dim, dtype=torch.float64, device=device)
        # fp64: centring subtracts n · mean_x · mean_yᵀ later, and a latent
        # with a large document share has a large mean.
        self.xy = torch.zeros(dim, vocab, dtype=torch.float64, device=device)
        self.sum_y = torch.zeros(vocab, dtype=torch.float64, device=device)

    def add(self, x, tokens) -> None:
        """Fit rows: ``x`` is ``(N, D)``, ``tokens`` the ``(N, k)`` next block."""
        k = tokens.shape[1]
        x64 = x.double()
        self.count += x.shape[0]
        self.sum_x += x64.sum(dim=0)
        self.xx += x64.T @ x64
        source = (x64 / k).T.repeat(1, k)  # (D, k * N), j-major
        self.xy.index_add_(1, tokens.T.reshape(-1), source)
        self.sum_y.index_add_(
            0, tokens.reshape(-1),
            torch.full((tokens.numel(),), 1.0 / k, dtype=torch.float64,
                       device=tokens.device),
        )

    def solve(self) -> None:
        n = max(1, self.count)
        self.mean_x, self.mean_y = self.sum_x / n, self.sum_y / n
        cxx = self.xx - n * torch.outer(self.mean_x, self.mean_x)
        cxy = self.xy - n * torch.outer(self.mean_x, self.mean_y)
        dim = cxx.shape[0]
        ridge = PROBE_RIDGE * cxx.trace().clamp_min(1e-12) / dim
        eye = torch.eye(dim, dtype=cxx.dtype, device=cxx.device)
        weight = torch.linalg.solve(cxx + ridge * eye, cxy)  # (D, vocab)
        self.weight_t = weight.T.float().contiguous()  # (vocab, D), for gathers
        self.gram = weight @ weight.T  # (D, D)
        self.weight_mean_y = weight @ self.mean_y  # (D,)
        self.mean_y_sq = (self.mean_y**2).sum()
        del self.xy

    def errors(self, x, tokens):
        """Per-row squared error of the probe and of the mean bag of words."""
        k = tokens.shape[1]
        xc = x.double() - self.mean_x
        y_sq = (tokens.unsqueeze(-1) == tokens.unsqueeze(-2)).sum(dim=(-1, -2)) / k**2
        y_mean = self.mean_y[tokens].sum(dim=-1) / k
        y_weight = F.embedding_bag(tokens, self.weight_t, mode="sum").double() / k
        y_pred = y_mean + (xc * y_weight).sum(dim=-1)
        pred_sq = (
            self.mean_y_sq + 2 * xc @ self.weight_mean_y
            + ((xc @ self.gram) * xc).sum(dim=-1)
        )
        return y_sq - 2 * y_pred + pred_sq, y_sq - 2 * y_mean + self.mean_y_sq


def _probe_rows(model, batches, latents, horizons):
    """Yield ``(features, tokens)`` per batch and horizon.

    ``features`` maps each name in :data:`PROBE_FEATURES` to ``(N, D)`` unit
    latents at the base positions with a real token ``k`` ahead; ``tokens``
    is the ``(N, k)`` block after each of them.
    """
    shortest = min(horizons)
    for (labels, mask), z in zip(batches, latents):
        if labels.shape[1] <= max(horizons):
            continue
        proposals = model.predictor(z[:, :-shortest])
        for horizon in horizons:
            valid = _prediction_mask(mask, horizon)
            if not valid.any():
                continue
            futures = _bases(proposals, z, horizon)
            features = {
                "z_t": z[:, :-horizon],
                "mu": futures.mu,
                "pi_argmax": futures.commit(PiArgmax()).future,
                "hindsight": futures.commit(Hindsight(), z[:, horizon:]).future,
            }
            width = z.shape[1] - horizon
            tokens = torch.stack(
                [labels[:, j : j + width][valid] for j in range(1, horizon + 1)],
                dim=-1,
            )
            yield (
                {n: F.normalize(f[valid].float(), dim=-1) for n, f in features.items()},
                tokens,
            )


def _probe_section(model, fit_batches, batches, latents, horizons) -> dict:
    """Section 7: how much of the next block's words a latent predicts.

    One linear probe per latent, fit on train-split chunks and scored on the
    query batches, pooled over the evaluation horizons.  The score is R²
    against the mean bag of words.  Nothing here is trained into the model.
    """
    device = latents[0].device
    vocab = model.config.encoder.vocab_size
    dim = latents[0].shape[-1]
    probes = {name: _BagOfWordsProbe(dim, vocab, device) for name in PROBE_FEATURES}
    # Encoded one batch at a time: 2048 chunks of latents would not fit at once.
    fit_latents = (model.encode(labels, attention_mask=mask) for labels, mask in fit_batches)
    for features, tokens in _probe_rows(model, fit_batches, fit_latents, horizons):
        for name, x in features.items():
            probes[name].add(x, tokens)
    if probes["z_t"].count == 0:
        raise ValueError("no probe fit positions in the train-split batches")
    for probe in probes.values():
        probe.solve()
    sse = dict.fromkeys(PROBE_FEATURES, 0.0)
    sst, positions = 0.0, 0
    for features, tokens in _probe_rows(model, batches, latents, horizons):
        positions += tokens.shape[0]
        for name, x in features.items():
            errors, reference = probes[name].errors(x, tokens)
            sse[name] += errors.sum().item()
        sst += reference.sum().item()
    r2 = {name: 1.0 - sse[name] / sst if sst > 0 else None for name in PROBE_FEATURES}

    def gain(name, over):
        return None if r2[name] is None else r2[name] - r2[over]

    return {
        "r2": r2,
        "gain_over_z_t": {n: gain(n, "z_t") for n in PROBE_FEATURES if n != "z_t"},
        "hindsight_over_mu": gain("hindsight", "mu"),
        "fit_positions": probes["z_t"].count,
        "positions": positions,
    }


def _verdict(name, metric, value, threshold, comparison) -> dict:
    value, threshold = float(value), float(threshold)
    ok = value > threshold if comparison == ">" else value < threshold
    return {
        "name": name, "metric": metric, "value": value, "threshold": threshold,
        "comparison": comparison, "ok": bool(ok),
    }
def _verdicts(metrics) -> list[dict]:
    """Section 8: pass/fail checks, each naming its metric and threshold."""
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
            "encoder_center", "representation.pooled_center",
            representation["pooled_center"], 0.05, "<",
        ),
        _verdict(
            "encoder_effective_rank", "representation.effective_rank",
            representation["effective_rank"], 15.0, ">",
        ),
    ]
    probe = metrics["probe"]
    if probe is not None and probe["hindsight_over_mu"] is not None:
        verdicts.append(
            _verdict(
                "hindsight_carries_content", "probe.hindsight_over_mu",
                probe["hindsight_over_mu"], 0.0, ">",
            )
        )
    if representation["max_rebound"] is not None:
        verdicts.append(
            _verdict(
                "no_periodic_shortcut", "representation.max_rebound",
                representation["max_rebound"], REBOUND_THRESHOLD, "<",
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
    best_of=DEFAULT_BEST_OF, probe_batches=None,
) -> dict:
    """Measure ``model`` on ``batches`` against a bank of real latents.

    ``loss_chunk_size=None`` uses the checkpoint's ``DecoderConfig``.
    ``best_of`` lists the K values of the best-of-K rollout error.
    ``probe_batches`` are train-split ``(labels, attention_mask)`` batches the
    next-block probe is fit on; without them the ``probe`` section is None.
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
    horizon_range = model.config.horizon_range
    horizons = evaluation_horizons(horizon_range)
    for labels, mask in batches:
        if mask is None:
            raise ValueError("measure needs attention masks (1 = real token)")
        if labels.shape[1] <= horizon_range[1]:
            raise ValueError(
                f"query length {labels.shape[1]} must exceed the longest "
                f"horizon {horizon_range[1]}"
            )
    device = next(model.parameters()).device
    batches = [(labels.to(device), mask.to(device)) for labels, mask in batches]
    bank = F.normalize(bank.float().to(device), dim=-1)
    loss_chunk_size = loss_chunk_size or model.config.decoder.loss_chunk_size
    latents = [model.encode(labels, attention_mask=mask) for labels, mask in batches]
    sections = {
        "representation": _representation_section(model, batches, latents, horizons),
        "loss": _loss_section(model, batches, horizons),
        "heads": _heads_section(model, batches, latents, horizons),
        "policies": _policies_section(model, batches, latents, horizons, seed=seed),
        "rollouts": _rollouts_section(
            model, batches, latents, bank, horizons, depths=depths,
            positions=rollout_positions, seed=seed,
            bank_chunk_size=bank_chunk_size, best_of=best_of,
        ),
        "decoder": _decoder_section(
            model, batches, latents, horizons, seed=seed,
            loss_chunk_size=loss_chunk_size,
        ),
        "probe": None if probe_batches is None else _probe_section(
            model,
            [(labels.to(device), mask.to(device)) for labels, mask in probe_batches],
            batches, latents, horizons,
        ),
    }
    context = {
        "checkpoint": None, "device": None,
        "examples": int(sum(labels.shape[0] for labels, _ in batches)),
        "batch_size": int(max(labels.shape[0] for labels, _ in batches)),
        "max_length": int(max(labels.shape[1] for labels, _ in batches)),
        "tokens": int(sum(int(mask.sum()) for _, mask in batches)),
        "horizon_range": [int(k) for k in horizon_range],
        "horizons": horizons,
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

def _steady_heads(steady) -> list:
    """The steady-state slot usage, as ``_pairs`` rows."""
    if steady is None:
        return []
    live = "n/a" if steady["live_slots"] is None else steady["live_slots"]
    return [(
        f"positions {steady['from_position']} and up",
        f"live slots {live}  (effective {_fmt(steady['effective_slots'])})  "
        f"slot-horizon MI {_fmt(steady['slot_horizon_mi'])}",
    )]

def _policy_table(p) -> None:
    """One row per commit rule: the mean over k, then each k."""
    per_k = {k: row for k, row in p["by_horizon"].items() if row is not None}
    if len(per_k) == 1:  # a fixed horizon: the mean is the only row
        per_k = {}

    def cells(get):
        return [_fmt(get(row)) for row in (p, *per_k.values())]

    _table(
        [(n, *cells(lambda row, n=n: row[n])) for n in POLICY_NAMES]
        + [(f"fixed slot {p['fixed_slot']['index']}",
            *cells(lambda row: row["fixed_slot"]["distance"]))]
        + [("hindsight gain over mu", *cells(lambda row: row["hindsight_gain_over_mu"])),
           ("commit headroom", *cells(lambda row: row["commit_headroom"]))],
        ("commit rule", "distance", *(f"k={k}" for k in per_k)),
    )

def print_report(metrics: dict) -> None:
    """Print every metric and verdict in ``collect_metrics``'s dict."""
    c, r = metrics["context"], metrics["representation"]
    print("=" * 78)
    print(f"measure: {c.get('checkpoint') or '(in-memory model)'}")
    print(
        f"device: {c.get('device') or '?'} | examples: {c['examples']} x "
        f"{c['max_length']} | tokens: {c['tokens']:,} | "
        f"candidates: {c['candidates']} | bank: {c['bank_size']:,}"
    )
    print(
        f"horizon range: {c['horizon_range']} | scored at k = "
        f"{', '.join(map(str, c['horizons']))}"
    )
    print("=" * 78)

    _pairs("representation", [
        ("per-dim std mean", _fmt(r["per_dim_std"])),
        ("mean vector norm", _fmt(r["mean_vector_norm"])),
        ("effective rank", f"{_fmt(r['effective_rank'])} / {c['latent_dim']}  "
                           f"({r['latents']:,} latents)"),
        ("cos(z_t, z_t+k)", f"{_fmt(r['temporal_cos'])}  (mean over k)"),
        ("max rebound", f"{_fmt(r['max_rebound'])}  "
                        "(largest return of cos at any lag in the range)"),
        ("fast share", f"{_fmt(r['fast_share'])}  "
                       f"(within-document energy faster than {FAST_SHARE_WINDOW} steps)"),
        ("document share", f"{_fmt(r['document_share'])}  "
                           "(between-document share of the variance)"),
        ("VISReg", " | ".join(f"{n} {_fmt(v)}" for n, v in r["visreg"].items())
                   + f"  (per batch of {c['batch_size']}; center grows with "
                     "document share / batch size)"),
        ("pooled center", f"{_fmt(r['pooled_center'])}  (mean over every query latent)"),
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
        ("slot-horizon MI", f"{_fmt(h['slot_horizon_mi'])}  of at most "
                            f"{_fmt(h['slot_horizon_mi_max'])} nats "
                            "(near 0 = no slot is tied to a distance)"),
        *_steady_heads(h["steady_state"]),
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
    print("\n--- policies (depth 1, distance to the real z_t+k, mean over k) ---")
    _policy_table(p)
    if p["steady_state"] is None:
        print(f"steady state: n/a (no query reaches position {STEADY_STATE_START})")
    else:
        print(f"steady state, positions {p['steady_state']['from_position']} and up:")
        _policy_table(p["steady_state"])

    ro = metrics["rollouts"]
    print(f"\n--- rollouts ({ro['positions']} starts, mean over k) ---")
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
    print(f"render gap (pi-argmax - real):         {_fmt(dec['render_gap'])}")
    print(f"specificity: shuffled-hindsight "
          f"{_fmt(dec['specificity']['shuffled_minus_hindsight'])}  vs "
          f"pi-argmax-hindsight {_fmt(dec['specificity']['pi_argmax_minus_hindsight'])}")

    pr = metrics["probe"]
    if pr is None:
        print("\n--- next-block probe: n/a (no train-split batches) ---")
    else:
        print(f"\n--- next-block probe (bag of words of the next k tokens, R², "
              f"fit on {pr['fit_positions']:,}, scored on {pr['positions']:,}) ---")
        _table([(n, _fmt(pr["r2"][n]), _fmt(pr["gain_over_z_t"].get(n)))
                for n in PROBE_FEATURES], ("latent", "R2", "gain over z_t"))
        print(f"hindsight over mu: {_fmt(pr['hindsight_over_mu'])}")

    _pairs("verdicts", [
        (f"[{'PASS' if v['ok'] else 'FAIL'}] {v['name']}",
         f"{v['metric']} = {_fmt(v['value'])} ({v['comparison']} {_fmt(v['threshold'])})")
        for v in metrics["verdicts"]
    ])
    print("\n--- how to read this ---")
    for line in (
        "representation: std near 1 and effective rank > 15 = no collapse;",
        "                cos(z_t, z_t+k) 0.5-0.85 healthy, >0.95 too smooth.",
        "                max rebound > 0 = the latent dips and returns at some lag,",
        "                a periodic code. document share near 1 = a per-document",
        "                constant, which is free to predict at any horizon.",
        "loss:           total is the weighted sum of the terms; H(r) is fixed",
        "                by the targets, so only KL(r||pi) can improve.",
        "heads:          pi gain > 0 = pi predicts the closest candidate from z_t.",
        "                slot-horizon MI near its maximum = the slots are phase",
        "                copies (which k), not different futures.",
        "policies:       hindsight needs the real future; hindsight gain over mu",
        "                is what candidates buy, commit headroom is the rest. The",
        "                document start flatters the averages: read the steady state.",
        "rollouts:       error is distance to the real z at t+i*k; manifold is",
        "                distance to the nearest real latent. The claim: at",
        "                depths >= 2, pi-sample stays closer to the manifold than",
        "                the mu path (compare with the real row).",
        "                best-of-K should fall from near mu toward hindsight as",
        "                K grows if the committed futures cover the real one.",
        "decoder:        hint use > 0 means the decoder uses the committed future.",
        "                null keeps the real z_t; future use is what the",
        "                committed future adds beyond it. real is the noise-free",
        "                block-end latent, real_noisy the decoder_hint=real training",
        "                condition; render gap is what a predicted future costs.",
        "                A decoder_hint=real decoder trained on real latents, so its",
        "                real, hint use and z_t use differ in meaning from a",
        "                committed-hint run; compare runs through pi_argmax.",
        "probe:          R² of a linear map to the next block's words; a hindsight",
        "                gain over mu means the candidates carry content.",
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
    parser.add_argument(
        "--probe-examples", type=int, default=DEFAULT_PROBE_EXAMPLES,
        help="Train-split chunks the next-block probe is fit on (0 skips it).",
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
    probe_batches = load_query_batches(
        train_dataset, tokenizer, num_examples=args.probe_examples,
        max_length=args.max_length, batch_size=args.batch_size, seed=args.seed + 1,
    ) if args.probe_examples > 0 else None
    metrics = collect_metrics(
        model, batches, bank, depths=args.depths,
        rollout_positions=args.rollout_positions, seed=args.seed,
        loss_chunk_size=args.loss_chunk_size, best_of=args.best_of,
        probe_batches=probe_batches,
    )
    metrics["context"]["checkpoint"] = str(args.model)
    metrics["context"]["device"] = str(device)
    print_report(metrics)
    if args.json is not None:
        args.json.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nwrote {args.json}")

if __name__ == "__main__":
    main()
