# Runs for the report

Every training run the report needs, with the exact command. Each
configuration runs with **3 seeds (1, 2, 3)**. Every configuration uses the
same seeds, so the P = 1 runs pair with the main run of the same seed
(report §4.5).

Totals: **18 JEPA-stage runs** (6 configurations × 3 seeds) and **9
decoder-stage runs** (main, P = 1 and π-argmax decoder, × 3 seeds).

Conventions:

- Training goes through the cluster launcher: `sbatch scripts/slurm_train.sh <args>`,
  which is meant to forward `<args>` to `python -m jepa train` (it does not yet;
  see the launcher warning in section 0).
- A JEPA run writes `runs/<config>-seed<N>`. Its decoder writes
  `runs/<config>-seed<N>-decoder`.
- A full JEPA stage (3 epochs, ~90k steps) may not fit in one 12 h job. If a
  job hits the time limit, resubmit with the **resume** command shown for
  that run. `--resume` keeps the checkpoint's config and seed, so don't pass
  `--config` or `--seed` with it.
- Start a decoder stage only after its JEPA run has finished all 3 epochs.
- `measure` and `generate` need a GPU but aren't in the launcher. Run them in
  a GPU job, e.g. `srun --gres=gpu:L40s:1 --time=02:00:00 <command>`.

---

## 0. First: the random-horizon gate (ticket 007)

Two single-seed runs with a fixed k = 20 learned a periodic shortcut: the
two-stage run (`runs/main-seed1-decoder`, max rebound 1.03) and the joint run
(`runs/joint-seed1`, `decoder_w = 0.3`, max rebound 0.31). The horizon is now
drawn from `[10, 30]` once per batch and hidden from the predictor (ADR 0003).
That is the config default, **and it is not validated yet**: any plain `train`
is a random-k run. **Sections 1–7 are on hold** until this section passes.

Both old checkpoints still load, and measure as fixed-k models. Loss values are
not comparable across horizon ranges; compare `measure` numbers only. The new
runs reuse some old names below, so move the old directories out of the way
first (e.g. `mv runs/main-seed1 runs/fixed20-two-stage-seed1`).

**Check the launcher first.** `scripts/slurm_train.sh` currently does not forward
its arguments: it trains the defaults into `runs/main-seed1`, then a decoder
stage, whatever is passed. Until it forwards `"$@"`, the commands below do not do
what they say.

### Gate (1 epoch each, seed 1, full data)

| Run | horizon | `decoder_w` | P |
|---|---|---|---|
| control | fixed 20 | 0.3 | 8 |
| A | random | 0.3 | 8 |
| B | random | 0.3 | 1 |
| C | random | 0 | 8 |

A is `joint-decoder.json`. Write the other presets outside the repo:

```sh
echo '{"decoder_w": 0.3, "horizon_range": [20, 20]}' > runs/gate-control.json
echo '{"decoder_w": 0.3, "predictor": {"prediction_count": 1}}' > runs/gate-b.json
echo '{"decoder_w": 0}' > runs/gate-c.json
sbatch scripts/slurm_train.sh --config runs/gate-control.json --epochs 1 --seed 1 --output-dir runs/gate-control
sbatch scripts/slurm_train.sh --config docs/experiments/joint-decoder.json --epochs 1 --seed 1 --output-dir runs/gate-a
sbatch scripts/slurm_train.sh --config runs/gate-b.json --epochs 1 --seed 1 --output-dir runs/gate-b
sbatch scripts/slurm_train.sh --config runs/gate-c.json --epochs 1 --seed 1 --output-dir runs/gate-c
```

Measure each (C's decoder is untrained; ignore its decoder section):

```sh
python -m jepa measure --model runs/gate-a --json runs/gate-a/metrics.json   # likewise for control, b, c
```

B isolates the mechanism: with one candidate nothing can absorb the cycle. A
against B shows whether winner-takes-all does. C shows what the change does
without the decoder gradient.

**Go for a full run of A** if, reading the `steady_state` rows (positions 80
and up) where there are any:

1. `no_periodic_shortcut` passes (max rebound < 0.05);
2. μ clearly beats copy, averaged over k;
3. live slots > 2 **and** `slot_horizon_mi` is low (phase copies would make all
   eight slots live). A low value excludes slots tied to a distance from z_t,
   not slots tied to an absolute phase (they win by t + k, which averages out
   over t), so it is necessary but not sufficient;
4. `document_share` is no higher than the control's;
5. `future_use` > 0.

If B passes and A fails on 1 or 3, winner-takes-all is absorbing the cycle: open
a separate ticket to score each candidate against two independently drawn
horizons and assign on the average distance. If B fails as well, the shortcut
is not about the fixed k. Stop and revisit.

### Full run (seed 1), then the `decoder_w` sweep

```sh
sbatch scripts/slurm_train.sh --config docs/experiments/joint-decoder.json --seed 1 --output-dir runs/random-joint-seed1
python -m jepa measure --model runs/random-joint-seed1 --json runs/random-joint-seed1/metrics.json
```

Read it with the same five conditions. If 0.3 is in doubt, sweep the weight with
1-epoch runs under the random horizon (C is already `w = 0` and A is `w = 0.3`):

```sh
for w in 0.1 1.0; do echo "{\"decoder_w\": $w}" > runs/decoder-w$w.json; done
sbatch scripts/slurm_train.sh --config runs/decoder-w0.1.json --epochs 1 --seed 1 --output-dir runs/sweep-w0.1
sbatch scripts/slurm_train.sh --config runs/decoder-w1.0.json --epochs 1 --seed 1 --output-dir runs/sweep-w1.0
```

Choose the largest weight where, compared with C, the JEPA terms (`move`, `pi`,
`mu`) still fall, effective rank stays > 15, `no_periodic_shortcut` passes and
`fast_share` drops clearly.

Then make the weight the default in `jepa/config.py` and delete
`joint-decoder.json`. Every section below then trains jointly without changing
its commands, and the separate decoder stages in sections 1 and 2 drop out,
because the joint run already trains its decoder. The two-stage ablation is
`runs/gate-c.json` (`decoder_w = 0`) with the same random horizon, seeds 1 to 3.
The two old fixed-k runs stay as the history of the shortcut, not as ablations.

P = 1 and P = 8 must use the same `horizon_range` and `decoder_w`. Their encoders
may still differ in periodicity, so report each one's max rebound next to the
comparison.

---

## 1. Main run (P = 8, config defaults)

Feeds Tables 4 and 5, Figures 2 and 3, and section 5.2 (H1–H4).

### JEPA stage

```sh
sbatch scripts/slurm_train.sh --seed 1 --output-dir runs/main-seed1
sbatch scripts/slurm_train.sh --seed 2 --output-dir runs/main-seed2
sbatch scripts/slurm_train.sh --seed 3 --output-dir runs/main-seed3
```

Resume:

```sh
sbatch scripts/slurm_train.sh --resume --output-dir runs/main-seed1   # seed2, seed3 likewise
```

### Decoder stage (hindsight hints, the default)

```sh
sbatch scripts/slurm_train.sh --stage decoder --from runs/main-seed1 --seed 1 --output-dir runs/main-seed1-decoder
sbatch scripts/slurm_train.sh --stage decoder --from runs/main-seed2 --seed 2 --output-dir runs/main-seed2-decoder
sbatch scripts/slurm_train.sh --stage decoder --from runs/main-seed3 --seed 3 --output-dir runs/main-seed3-decoder
```

Resume:

```sh
sbatch scripts/slurm_train.sh --stage decoder --resume --from runs/main-seed1 --output-dir runs/main-seed1-decoder
```

### Measure

The decoder checkpoint contains the frozen JEPA, so one measurement covers
the representation, heads, rollouts and decoder.

```sh
python -m jepa measure --model runs/main-seed1-decoder --json runs/main-seed1-decoder/metrics.json
python -m jepa measure --model runs/main-seed2-decoder --json runs/main-seed2-decoder/metrics.json
python -m jepa measure --model runs/main-seed3-decoder --json runs/main-seed3-decoder/metrics.json
```

---

## 2. P = 1 baseline (`p1-baseline.json`)

This is the external baseline. It supplies the P = 1 rows of Table 4 and the
P = 1 column of Figure 3, so it needs a decoder too.

### JEPA stage

```sh
sbatch scripts/slurm_train.sh --config docs/experiments/p1-baseline.json --seed 1 --output-dir runs/p1-seed1
sbatch scripts/slurm_train.sh --config docs/experiments/p1-baseline.json --seed 2 --output-dir runs/p1-seed2
sbatch scripts/slurm_train.sh --config docs/experiments/p1-baseline.json --seed 3 --output-dir runs/p1-seed3
```

Resume:

```sh
sbatch scripts/slurm_train.sh --resume --output-dir runs/p1-seed1
```

### Decoder stage

```sh
sbatch scripts/slurm_train.sh --stage decoder --from runs/p1-seed1 --seed 1 --output-dir runs/p1-seed1-decoder
sbatch scripts/slurm_train.sh --stage decoder --from runs/p1-seed2 --seed 2 --output-dir runs/p1-seed2-decoder
sbatch scripts/slurm_train.sh --stage decoder --from runs/p1-seed3 --seed 3 --output-dir runs/p1-seed3-decoder
```

Resume:

```sh
sbatch scripts/slurm_train.sh --stage decoder --resume --from runs/p1-seed1 --output-dir runs/p1-seed1-decoder
```

### Measure

```sh
python -m jepa measure --model runs/p1-seed1-decoder --json runs/p1-seed1-decoder/metrics.json
python -m jepa measure --model runs/p1-seed2-decoder --json runs/p1-seed2-decoder/metrics.json
python -m jepa measure --model runs/p1-seed3-decoder --json runs/p1-seed3-decoder/metrics.json
```

---

## 3. Decoder trained on π-argmax hints (`decoder-pi-argmax.json`)

This is a decoder-only ablation on top of the main run's JEPA. No new JEPA
training is needed. It compares with the main decoder in Table 5 and §5.6.

### Decoder stage

```sh
sbatch scripts/slurm_train.sh --stage decoder --from runs/main-seed1 --config docs/experiments/decoder-pi-argmax.json --seed 1 --output-dir runs/main-seed1-decoder-pi-argmax
sbatch scripts/slurm_train.sh --stage decoder --from runs/main-seed2 --config docs/experiments/decoder-pi-argmax.json --seed 2 --output-dir runs/main-seed2-decoder-pi-argmax
sbatch scripts/slurm_train.sh --stage decoder --from runs/main-seed3 --config docs/experiments/decoder-pi-argmax.json --seed 3 --output-dir runs/main-seed3-decoder-pi-argmax
```

Resume:

```sh
sbatch scripts/slurm_train.sh --stage decoder --resume --from runs/main-seed1 --output-dir runs/main-seed1-decoder-pi-argmax
```

### Measure

```sh
python -m jepa measure --model runs/main-seed1-decoder-pi-argmax --json runs/main-seed1-decoder-pi-argmax/metrics.json
python -m jepa measure --model runs/main-seed2-decoder-pi-argmax --json runs/main-seed2-decoder-pi-argmax/metrics.json
python -m jepa measure --model runs/main-seed3-decoder-pi-argmax --json runs/main-seed3-decoder-pi-argmax/metrics.json
```

---

## 4. P = 4 (`p4.json`)

This is a JEPA ablation. The §5.6 table only uses latent metrics (depth-4
manifold distance, best-of-16 error, live slots), so **no decoder stage**.
`measure` runs on the JEPA checkpoint, and its decoder section is
meaningless there (untrained decoder). Ignore it.

### JEPA stage

```sh
sbatch scripts/slurm_train.sh --config docs/experiments/p4.json --seed 1 --output-dir runs/p4-seed1
sbatch scripts/slurm_train.sh --config docs/experiments/p4.json --seed 2 --output-dir runs/p4-seed2
sbatch scripts/slurm_train.sh --config docs/experiments/p4.json --seed 3 --output-dir runs/p4-seed3
```

Resume:

```sh
sbatch scripts/slurm_train.sh --resume --output-dir runs/p4-seed1
```

### Measure

```sh
python -m jepa measure --model runs/p4-seed1 --json runs/p4-seed1/metrics.json
python -m jepa measure --model runs/p4-seed2 --json runs/p4-seed2/metrics.json
python -m jepa measure --model runs/p4-seed3 --json runs/p4-seed3/metrics.json
```

---

## 5. No annealing (`no-anneal.json`)

This is a JEPA ablation with no decoder stage (see section 4).

### JEPA stage

```sh
sbatch scripts/slurm_train.sh --config docs/experiments/no-anneal.json --seed 1 --output-dir runs/no-anneal-seed1
sbatch scripts/slurm_train.sh --config docs/experiments/no-anneal.json --seed 2 --output-dir runs/no-anneal-seed2
sbatch scripts/slurm_train.sh --config docs/experiments/no-anneal.json --seed 3 --output-dir runs/no-anneal-seed3
```

Resume:

```sh
sbatch scripts/slurm_train.sh --resume --output-dir runs/no-anneal-seed1
```

### Measure

```sh
python -m jepa measure --model runs/no-anneal-seed1 --json runs/no-anneal-seed1/metrics.json
python -m jepa measure --model runs/no-anneal-seed2 --json runs/no-anneal-seed2/metrics.json
python -m jepa measure --model runs/no-anneal-seed3 --json runs/no-anneal-seed3/metrics.json
```

---

## 6. No diversity (`no-diversity.json`)

This is a JEPA ablation with no decoder stage (see section 4).

### JEPA stage

```sh
sbatch scripts/slurm_train.sh --config docs/experiments/no-diversity.json --seed 1 --output-dir runs/no-diversity-seed1
sbatch scripts/slurm_train.sh --config docs/experiments/no-diversity.json --seed 2 --output-dir runs/no-diversity-seed2
sbatch scripts/slurm_train.sh --config docs/experiments/no-diversity.json --seed 3 --output-dir runs/no-diversity-seed3
```

Resume:

```sh
sbatch scripts/slurm_train.sh --resume --output-dir runs/no-diversity-seed1
```

### Measure

```sh
python -m jepa measure --model runs/no-diversity-seed1 --json runs/no-diversity-seed1/metrics.json
python -m jepa measure --model runs/no-diversity-seed2 --json runs/no-diversity-seed2/metrics.json
python -m jepa measure --model runs/no-diversity-seed3 --json runs/no-diversity-seed3/metrics.json
```

---

## 7. Fixed horizon (`fixed-horizon.json`)

The ablation of the random horizon: `horizon_range = [20, 20]`, everything else
as the main run. Is the shortcut caused by a fixed k? It is measured at k = 20
only, so compare its representation section (max rebound, fast share, document
share) with the main run's, not its losses. Its max rebound scans lags up to 20
and the main run's up to 30, which can only raise the main run's; compare the
lag profiles too. Its rollout starts need 80 tokens ahead instead of 120, so its
rollouts use different starts.

### JEPA stage

```sh
sbatch scripts/slurm_train.sh --config docs/experiments/fixed-horizon.json --seed 1 --output-dir runs/fixed-horizon-seed1
sbatch scripts/slurm_train.sh --config docs/experiments/fixed-horizon.json --seed 2 --output-dir runs/fixed-horizon-seed2
sbatch scripts/slurm_train.sh --config docs/experiments/fixed-horizon.json --seed 3 --output-dir runs/fixed-horizon-seed3
```

Resume:

```sh
sbatch scripts/slurm_train.sh --resume --output-dir runs/fixed-horizon-seed1
```

### Measure

```sh
python -m jepa measure --model runs/fixed-horizon-seed1 --json runs/fixed-horizon-seed1/metrics.json
python -m jepa measure --model runs/fixed-horizon-seed2 --json runs/fixed-horizon-seed2/metrics.json
python -m jepa measure --model runs/fixed-horizon-seed3 --json runs/fixed-horizon-seed3/metrics.json
```

---

## 8. Figure 3: decoded plans (no training)

This needs sections 1 and 2 finished for seed 1. It decodes the main run's
plans (π-sample ×3, π-argmax, μ) next to the P = 1 model's plan, which the
P = 1 model's own decoder decodes. Each prompt draws one k from the seed, shared
by every column.

```sh
python -m jepa generate --model runs/main-seed1-decoder --baseline-model runs/p1-seed1-decoder --plan-depth 4 --samples 3 --seed 0 --json runs/main-seed1-decoder/plans.json --html runs/main-seed1-decoder/plans.html
```

---

## Not planned

- **MDN predictor** (Table 2, optional): needs code that doesn't exist yet.
  Decide on it before adding runs.
- **Synthetic experiment** (§5.1): already done. It runs on CPU with
  `python docs/report/synthetic.py`.
