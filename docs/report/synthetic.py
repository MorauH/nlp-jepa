"""Synthetic experiment for the report (section 5.1).

Trains the P = 4 model and the P = 1 baseline on the two-successor Markov
chain of :mod:`jepa.evaluation.synthetic`, over several seeds, and measures
rollouts at depths 1-4 from the start states of sampled real paths:
hindsight, π-argmax, π-sample and the μ path.

Writes ``synthetic.json`` (every metric, mean and sample std over seeds, plus
the per-seed values) and ``synthetic.svg`` (Figure 1) next to this file.
Both are written directly: under ``nix develop`` the shell banner goes to
stdout, so redirecting stdout would corrupt them.

    python docs/report/synthetic.py

CPU only, about a minute.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# `jepa` is not an installed package; make the repository root importable.
sys.path.insert(0, str(HERE.parents[1]))

import torch  # noqa: E402

from jepa.evaluation.synthetic import (  # noqa: E402
    DIM,
    N_STATES,
    ON_STATE_DISTANCE,
    TRAIN_STEPS,
    World,
    branch_entropy,
    build_world,
    manifold_distance,
    mode_coverage,
    mu_rollout,
    path_validity,
    sample_path_indices,
    train_predictor,
)
from jepa.models import Hindsight, PiArgmax, PiSample, distance, rollout  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
PREDICTION_COUNTS = [4, 1]
DEPTH = 4
PATHS = 1024  # sampled real paths per seed
BRANCH_SAMPLES = 512  # depth-1 π-samples per state for branch entropy
PATHS_MEASURED = ["hindsight", "pi_argmax", "pi_sample", "mu"]
PER_DEPTH = ["rollout_error", "manifold_distance", "path_validity"]
SCALARS = ["branch_entropy", "mode_coverage", "mu_norm"]

JSON_PATH = HERE / "synthetic.json"
SVG_PATH = HERE / "synthetic.svg"


def model_key(prediction_count: int) -> str:
    return f"P={prediction_count}"


@torch.no_grad()
def measure(predictor, world: World, seed: int) -> dict:
    """All metrics for one trained predictor on one seed."""
    states = world.states
    generator = torch.Generator().manual_seed(seed + 1000)
    indices = sample_path_indices(world.successors, PATHS, DEPTH, generator)
    path = states[indices]  # (B, DEPTH + 1, D)
    start, real = path[:, 0], path[:, 1:]

    rollouts = {
        "hindsight": rollout(
            predictor, start, Hindsight(), DEPTH, targets=real
        ).future,
        "pi_argmax": rollout(predictor, start, PiArgmax(), DEPTH).future,
        "pi_sample": rollout(
            predictor, start, PiSample(generator=generator), DEPTH
        ).future,
        "mu": mu_rollout(predictor, start, DEPTH),
    }
    paths = {
        name: {
            "rollout_error": distance(latents, real).mean(0).tolist(),
            "manifold_distance": manifold_distance(latents, states).mean(0).tolist(),
            "path_validity": path_validity(latents, indices[:, 0], world).tolist(),
        }
        for name, latents in rollouts.items()
    }

    from_every_state = states.unsqueeze(1).expand(N_STATES, BRANCH_SAMPLES, DIM)
    commits = predictor(from_every_state).commit(PiSample(generator=generator))
    futures = predictor(states)
    return {
        "paths": paths,
        "branch_entropy": branch_entropy(commits.future, states),
        "mode_coverage": mode_coverage(futures.candidates, world),
        "mu_norm": futures.mu.norm(dim=-1).mean().item(),
    }


def summarise(values: list) -> dict:
    """Mean and sample std over seeds (element-wise for per-depth lists)."""
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": tensor.mean(0).tolist(),
        "std": tensor.std(0, unbiased=True).tolist(),
        "per_seed": values,
    }


def run() -> dict:
    per_seed: dict[str, list[dict]] = {model_key(p): [] for p in PREDICTION_COUNTS}
    for seed in SEEDS:
        world = build_world(seed)
        for count in PREDICTION_COUNTS:
            predictor = train_predictor(world, count, seed)
            per_seed[model_key(count)].append(measure(predictor, world, seed))
            print(f"seed {seed} {model_key(count)} done", file=sys.stderr)

    models = {}
    for key, runs in per_seed.items():
        models[key] = {
            "paths": {
                name: {
                    metric: summarise([r["paths"][name][metric] for r in runs])
                    for metric in PER_DEPTH
                }
                for name in PATHS_MEASURED
            },
            **{name: summarise([r[name] for r in runs]) for name in SCALARS},
        }

    return {
        "setup": {
            "n_states": N_STATES,
            "dim": DIM,
            "train_steps": TRAIN_STEPS,
            "seeds": SEEDS,
            "depths": list(range(1, DEPTH + 1)),
            "paths_per_seed": PATHS,
            "branch_samples_per_state": BRANCH_SAMPLES,
            "on_state_distance": ON_STATE_DISTANCE,
            "std": "sample std over seeds",
            "note": (
                "P=1 has one candidate, so its hindsight, pi_argmax and "
                "pi_sample rollouts are the same rollout."
            ),
        },
        "models": models,
    }


# --------------------------------------------------------------------------
# Figure 1: a dependency-free SVG line chart, one panel per metric.

SERIES = [  # (label, model, path, colour)
    ("P=4 π-sample", "P=4", "pi_sample", "#1f6f8b"),
    ("P=4 hindsight", "P=4", "hindsight", "#3a9d5d"),
    ("P=4 μ path", "P=4", "mu", "#c0392b"),
    ("P=1", "P=1", "pi_argmax", "#8e6c3a"),
]
PANELS = [
    ("manifold distance", "manifold_distance"),
    ("rollout error", "rollout_error"),
    ("path validity", "path_validity"),
]


def figure(results: dict) -> str:
    width, height, pad = 230, 190, 42
    total = len(PANELS) * width + 20
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total} {height + 50}" '
        f'font-family="Georgia, serif" font-size="11">'
    ]
    depths = results["setup"]["depths"]
    for column, (title, metric) in enumerate(PANELS):
        x0 = column * width + pad
        plot_w, plot_h = width - pad - 12, height - 50
        y0 = 24
        series = [
            results["models"][model]["paths"][path][metric]
            for _, model, path, _ in SERIES
        ]
        top = max(m + s for d in series for m, s in zip(d["mean"], d["std"]))
        top = 1.0 if metric == "path_validity" else top * 1.08

        def sx(depth):
            return x0 + (depth - depths[0]) / (depths[-1] - depths[0]) * plot_w

        def sy(value):
            return y0 + plot_h - value / top * plot_h

        parts.append(
            f'<text x="{x0 + plot_w / 2}" y="14" text-anchor="middle">{title}</text>'
            f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" '
            f'fill="none" stroke="#999"/>'
        )
        for tick in (0.0, top / 2, top):
            parts.append(
                f'<text x="{x0 - 4}" y="{sy(tick) + 4}" text-anchor="end">'
                f"{tick:.2f}</text>"
            )
        for depth in depths:
            parts.append(
                f'<text x="{sx(depth)}" y="{y0 + plot_h + 14}" '
                f'text-anchor="middle">{depth}</text>'
            )
        parts.append(
            f'<text x="{x0 + plot_w / 2}" y="{y0 + plot_h + 28}" '
            f'text-anchor="middle">depth</text>'
        )
        for (_, _, _, colour), data in zip(SERIES, series):
            points = " ".join(
                f"{sx(d)},{sy(m)}" for d, m in zip(depths, data["mean"])
            )
            parts.append(
                f'<polyline points="{points}" fill="none" stroke="{colour}" '
                f'stroke-width="1.6"/>'
            )
            for d, m, s in zip(depths, data["mean"], data["std"]):
                parts.append(
                    f'<line x1="{sx(d)}" x2="{sx(d)}" y1="{sy(m - s)}" '
                    f'y2="{sy(m + s)}" stroke="{colour}"/>'
                    f'<circle cx="{sx(d)}" cy="{sy(m)}" r="2.4" fill="{colour}"/>'
                )

    legend_y = height + 30
    for index, (label, _, _, colour) in enumerate(SERIES):
        x = pad + index * 150
        parts.append(
            f'<line x1="{x}" x2="{x + 18}" y1="{legend_y}" y2="{legend_y}" '
            f'stroke="{colour}" stroke-width="2"/>'
            f'<text x="{x + 24}" y="{legend_y + 4}">{label}</text>'
        )
    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    began = time.perf_counter()
    results = run()
    JSON_PATH.write_text(json.dumps(results, indent=2) + "\n")
    SVG_PATH.write_text(figure(results))
    print(
        f"wrote {JSON_PATH} and {SVG_PATH} in {time.perf_counter() - began:.0f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
