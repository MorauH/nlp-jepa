# 005: Best-of-K rollout error in `measure`

Status: open

## Why

On real text there is no list of valid futures, so the report cannot measure
path validity directly. Rollout error against the one realized continuation
penalizes commitment by construction: a committed rollout picks *one*
plausible branch, while the mean minimizes the expected distance to whichever
branch happens. A fairer test of "the committed futures include the real one"
is the standard multi-hypothesis metric from trajectory forecasting
(minADE_K / oracle-of-K):

> Draw K independent π-sample rollouts. Score each one by its rollout error
> against the real continuation and keep the best.

This should fall below the μ path's error as K grows if the committed futures
cover the real one. If it never does, the candidates are not covering the
real futures, and the report must say so.

## What

- In the `rollouts` section of `jepa/evaluation/measure.py`, add
  `best_of_k[K][depth]` for K ∈ {1, 4, 16} (a `--best-of` flag). Use
  independent π-sample rollouts from the same starts with one seeded
  generator. Report it at each depth, along with the μ path and hindsight rows
  it should sit between.
- Score whole trajectories, not each depth separately. For depth i, the best
  rollout is the one with the lowest mean error over depths 1..i. Picking the
  best per depth would mix several rollouts and overstate coverage.
- Batch the K samples (repeat the starts K times) rather than looping, and
  keep the existing memory bound.
- Add a verdict: at the deepest depth, best-of-16 < μ path.

## Acceptance criteria

- `collect_metrics` returns `rollouts.best_of_k` and `print_report` shows it as
  a table (rows K, columns depth).
- The smoke test checks the keys, and checks that best-of-K is non-increasing
  in K on the tiny model (it must be, with nested samples: use the first K of
  the 16 draws).
- `docs/architecture.md` explains why best-of-K exists, in two sentences.
