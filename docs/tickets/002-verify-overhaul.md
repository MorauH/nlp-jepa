# 002: Make `python -m jepa verify` reliable and decision-oriented

Status: open

## Why

The 2026-11-22 evaluation of `runs/jepa` needed a separate script, because
`verify` (`jepa/evaluation/verify.py`) gave misleading or too-noisy answers:

- **Misleading verdict.** It prints "SCORER PROBLEM" when the selected
  waypoint is not closer than copying `z_t`. That rule predates π: π is a
  sampling distribution over futures, not an expected-distance selector, so
  "π argmax is no closer than copy" is expected, not a bug (see
  `docs/vision.md`).
- **Too small and too short.** The default 8 examples × 256 tokens gave
  `best_d` 0.18 and effective rank 15. On 64 × 1024 tokens the same
  checkpoint gave `best_d` 0.09 and effective rank 29. Short sequences
  understate predictability, and 2k latents understate the rank.
- **Missing the conditions that matter.** The hint ablation has no
  π-sampled hint (what generation uses) and no hindsight/oracle hint (what
  the decoder is trained on). Those two showed the key finding: π argmax
  4.444, oracle 4.436, real block 4.425, shuffled 4.628 nats. In other words
  the decoder takes topic from the waypoint but almost nothing about the
  specific future.
- **No loss breakdown.** It prints only the JEPA total.
- **Not machine-readable**, so runs can't be compared without copying
  numbers by hand.

## What

### Sampling

- Defaults: 64 validation examples at full length (1024), evaluated in
  mini-batches (`--batch-size`, default 4, which fits 16 GB), with
  token-weighted averages. Keep `--num-examples` / `--max-length` overrides.
- Effective rank: computed on up to 20k valid latents gathered across batches.

### Sections and metrics

1. **Representation:** per-dim std, mean-vector norm, effective rank (20k),
   cos(z_t, z_{t+k}), plus the VISReg `center`/`scale`/`shape` terms.
2. **JEPA loss breakdown:** each term, its weight, its weighted contribution,
   and the total. For π, also show the floor: `H(r)`, the mean entropy of the
   soft assignment, and `KL(r‖π) = loss_score − H(r)`. The KL part is what π
   can still learn. Needs `r`; expose it from `VISRegLoss` stats (detached,
   masked) or recompute it with the same temperature.
3. **Candidates and heads:**
   - `best_d`, candidate pair cosine, delta/z;
   - win histogram with live-slot count (share > 1/(4P)) and effective slot
     count `exp(H(win shares))`;
   - π: NLL vs constant (gain in nats) and accuracy vs majority slot;
   - μ: `|μ|`, measured μ-direction distance vs the calibrated prediction
     `2 − 2|μ|`. If they disagree, μ is miscalibrated.
4. **Waypoint distances (and retrieval):** copy z_t, π argmax, π sampled,
   μ-safe, μ direction, best fixed slot, oracle. Derived: oracle gain over μ
   direction (what P candidates buy over one prediction).
5. **Decoder hint ablation** (token-weighted CE): π argmax, π sampled,
   μ-safe, oracle (hindsight), real block (leaky), shuffled, null. Derived:
   - hint use = shuffled − π argmax (does the decoder use the waypoint?);
   - selection headroom = π argmax − oracle (would a better selector help?);
   - specificity = shuffled − oracle vs π argmax − oracle (topic vs specific
     future).

   Per-offset columns move behind `--per-offset` (20 columns are noise by
   default).
6. **Verdicts:** replace the old SCORER/CANDIDATE logic with checks on:
   π gain > 0, oracle gain over μ direction, hint use > 0.05, selection
   headroom, μ calibration, encoder center/rank. Each verdict line names the
   metric and threshold it used.

### Output

- `--json PATH` writes every metric (nested by section) for diffing runs.
- Keep the reference guide, updated to the new metrics.

### Structure

- Split into `collect_metrics(model, batches, ...) -> dict` and
  `print_report(metrics)`. The printed report and the JSON come from the same
  dict.
- Old checkpoints (with a trunk) can't load with the current code
  (`PredictorConfig` raises). To compare against them, run the old commit's
  code in a worktree. Note this in the module docstring rather than adding
  version shims.

## Acceptance criteria

- `python -m jepa verify --model runs/jepa --json out.json` finishes on a
  16 GB GPU with the defaults and prints all six sections.
- "SCORER PROBLEM" and the copy-based selection verdict are gone.
- A smoke test runs `collect_metrics` on the tiny test config with a saved
  tiny dataset (pattern: `test_train_decoder_entry_point_writes_loadable_checkpoint`)
  and checks the JSON keys, plus one identity: the weighted sum of the loss
  terms equals the reported total.
- `docs/architecture.md` describes the new output briefly.

## Reference numbers (2026-11-22, 64 × 1024 validation tokens)

| Metric | `runs/jepa-bk` (trunk, CE) | `runs/jepa` (z_t, π+μ) |
|---|---|---|
| best_d | 0.095 | 0.090 |
| π NLL / constant | 1.244 / 1.485 | 1.384 / 1.523 |
| effective rank (20k) | 41.6 | 29.0 |
| loss_center | 0.022 | 0.195 |
| CE π argmax / oracle / shuffled / null | 4.268 / 4.265 / 4.288 / 4.282 | 4.444 / 4.436 / 4.628 / 5.757 |
| μ direction distance / \|μ\| | — | 0.479 / 0.762 |
