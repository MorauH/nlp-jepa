# 004: Synthetic experiment script for the report

Status: done

## Why

The report's first results section shows the mechanism on a world where the
true futures are known. `tests/test_research_claim.py` already contains that
world and asserts the claim, but a test gives pass/fail at one seed, not the
mean ± std numbers and the baseline comparison a report needs.

## What

A script `docs/report/synthetic.py` that writes `docs/report/synthetic.json`.
Write the file directly: the `nix develop` banner goes to stdout, so
redirecting stdout corrupts the JSON.

- **World:** the same as the test. N = 8 states at the latent scale (norm √D,
  D = 32), each with two distinct successors chosen at random, each taken with
  probability 1/2. Reuse the test's world construction rather than copying it,
  e.g. by moving `build_world` and the μ path into a small shared helper.
- **Models:** P = 4 (as in the test) and **P = 1** (the single-prediction
  baseline), trained identically with `CommitmentLoss`. 5 seeds each.
- **Rollouts, depths 1–4**, from the start states of sampled real paths:
  hindsight (targets = the real path), π-argmax, π-sample, and the μ path.
- **Metrics per depth (mean and std over seeds):**
  - **rollout error**: distance to the real path's latent;
  - **manifold distance**: nearest-state distance;
  - **path validity**: the fraction of rollouts whose steps 1..i are all real
    transitions. A step counts when it lies on a state (distance < 0.1) and
    that state is a successor of the previous step's state.
  - **branch entropy** (bits) of π-sample at depth 1: how evenly it splits
    between the two successors (ideal: 1 bit).
  - **mode coverage** (the worst-case distance from a successor to its nearest
    candidate) and |μ|.

## Expected reading (to check, not to assume)

On this world the mean minimizes the expected distance to the realized
future, so rollout error against one realized path should *favour* μ and P = 1
over committed π rollouts. The report's argument rests on validity, manifold
distance and branch entropy, plus hindsight rollout error. The script must
report all of them, so the report can make this point with numbers.

## Acceptance criteria

- `python docs/report/synthetic.py` runs on CPU in under a few minutes and
  writes the JSON with all metrics for P = 4 and P = 1.
- The test and the script share the world and μ-path code, so there is only
  one copy.
- The report's synthetic table and figure are filled from the JSON, and its
  `TODO` markers for section 5.1 are removed.
- `AGENTS.md` mentions the script under `docs/report/`.
