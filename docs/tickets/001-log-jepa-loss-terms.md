# 001: Log each JEPA loss term separately

Status: open

## Why

`history.json` and the epoch summary only record the weighted JEPA total
(`train_jepa_loss` / `val_jepa_loss`). The total is a poor progress signal:

- About 60% of it is π's cross-entropy (`loss_score`), which is bounded below
  by the entropy of the soft winner assignment and so never approaches 0.
- Terms were added over time (`loss_mean` in the π+μ split), so totals from
  different runs are not comparable. In the 2026-11-22 evaluation the new
  run's total was 2.37 vs 1.59 for `runs/jepa-bk`, and 0.41 of that gap was
  the new `loss_mean` term alone.
- The real regression in that run (encoder off-center: `loss_center` 0.20 vs
  0.02; `loss_shape` 0.09 vs 0.04) was invisible in the history. So was the
  cause of `val_jepa_loss` rising in epoch 4 (1.96 → 2.16).

## What

Record every scalar in `jepa_out["stats"]` (from `VISRegLoss`) in the
training and validation history, next to the existing totals.

Terms (unweighted, as the loss reports them):
`loss_move`, `loss_score`, `loss_mean`, `loss_diversity`, `loss_scale`,
`loss_shape`, `loss_center`.

Diagnostics (cheap, already computed): `best_d`, `selected_d`, `mean_norm`,
`mean_dir_d`, `temperature`.

## Implementation notes

- `JepaTrainer._losses` (`jepa/training/trainer.py`) builds the per-batch
  `values` dict. Add the scalar stats from `jepa_out["stats"]` when
  `jepa_out["loss"] is not None`. Skip non-scalars (`win`) and keep the
  existing detach-and-accumulate-on-GPU path (`_accumulate`/`_means`), so
  logging still synchronizes only every `log_every` steps.
- Key names: prefix with `jepa_` and drop `loss_`, e.g. `jepa_move`,
  `jepa_center`, `jepa_best_d`. History keys then become
  `train_jepa_move`, `val_jepa_center`, and so on. Existing tests rely on two
  rules this naming keeps:
  - frozen (`train-decoder`) runs have no key containing `jepa`;
  - `--jepa-only` runs have no key containing `decoder`/`perplexity`.
- Store unweighted values. The weights are in the checkpoint's `visreg`
  config, and `total = Σ weight_i · term_i` (`VISRegLoss._multi_hypothesis_loss`).
- Console: keep the tqdm postfix compact (JEPA total only). Print the
  per-term breakdown as a second line of the epoch summary (`describe` in
  `JepaTrainer.train`).
- Mid-epoch checkpoints write `last_checkpoint` with the running train means.
  The new keys appear there automatically through `_means`.
- `--resume` on an older history: new keys start with fewer entries than
  `train_jepa_loss`. Either accept that (document it) or pad the missing
  epochs with `null` when a key first appears. Pick one, and handle it in the
  history-writing loop.
- `torch.compile`: `stats` is already returned from the compiled forward, so
  no graph change is needed.

## Acceptance criteria

- A `--jepa-only` run writes `train_*` and `val_*` lists for every term and
  diagnostic above, one entry per epoch.
- For one validation pass, the weighted sum of the logged terms equals
  `val_jepa_loss` (up to float error). Add a test with the tiny config, in
  the style of `test_jepa_only_never_runs_the_decoder`.
- Frozen-decoder runs log no `jepa_*` keys (existing test still passes).
- The epoch summary prints the per-term line.
- `docs/architecture.md` (Training section) lists the logged keys.
