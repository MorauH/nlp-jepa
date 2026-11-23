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

CPU only, about 15 seconds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# `jepa` is not an installed package; make the repository root importable.
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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
    path_validity,
    sample_path_indices,
    train_predictor,
)
from jepa.evaluation.measure import mu_path  # noqa: E402
from jepa.models import Hindsight, PiArgmax, PiSample, distance, rollout  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
PREDICTION_COUNTS = [4, 1]
DEPTH = 4
PATHS_PER_SEED = 1024  # sampled real paths
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
    indices = sample_path_indices(world.successors, PATHS_PER_SEED, DEPTH, generator)
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
        "mu": mu_path(predictor, start, DEPTH),
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
        "branch_entropy": branch_entropy(commits.future, world),
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
            "paths_per_seed": PATHS_PER_SEED,
            "branch_samples_per_state": BRANCH_SAMPLES,
            "on_state_distance": ON_STATE_DISTANCE,
            "std": "sample std over seeds",
            "note": (
                "P=1 has one candidate, so its hindsight, pi_argmax and "
                "pi_sample rollouts are the same rollout. branch_entropy "
                "counts only depth-1 commits that land on a successor."
            ),
        },
        "models": models,
    }


# --------------------------------------------------------------------------
# Figure 1: one panel per metric, mean with sample-std error bars.

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


def figure(results: dict, path: Path) -> None:
    depths = results["setup"]["depths"]
    # Keep text as text, so the report's font applies and the file stays small.
    with plt.rc_context({"svg.fonttype": "none", "font.family": "serif"}):
        fig, axes = plt.subplots(1, len(PANELS), figsize=(10, 3.5))
        for ax, (title, metric) in zip(axes, PANELS):
            for label, model, rollout_path, colour in SERIES:
                data = results["models"][model]["paths"][rollout_path][metric]
                ax.errorbar(
                    depths, data["mean"], yerr=data["std"], label=label,
                    color=colour, marker="o", markersize=3, capsize=2,
                )
            ax.set_title(title)
            ax.set_xlabel("depth")
            ax.set_xticks(depths)
            ax.set_ylim(bottom=0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(SERIES), frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(path, metadata={"Date": None})
        plt.close(fig)


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    began = time.perf_counter()
    results = run()
    JSON_PATH.write_text(json.dumps(results, indent=2) + "\n")
    figure(results, SVG_PATH)
    print(
        f"wrote {JSON_PATH} and {SVG_PATH} in {time.perf_counter() - began:.0f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
