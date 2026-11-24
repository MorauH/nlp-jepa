# Runs for the report

Every training run the report needs, with the exact command. Each
configuration runs with **3 seeds (1, 2, 3)**. Every configuration uses the
same seeds, so the P = 1 runs pair with the main run of the same seed
(report §4.5).

Totals: **15 JEPA-stage runs** (5 configurations × 3 seeds) and **9
decoder-stage runs** (main, P = 1 and π-argmax decoder, × 3 seeds).

Conventions:

- Training goes through the cluster launcher: `sbatch scripts/slurm_train.sh <args>`,
  which forwards `<args>` to `python -m jepa train`.
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

## 0. First: `decoder_w` sweep and one joint main run (ticket 006)

The first two-stage main run learned a periodic shortcut (ticket 006, ADR
0002). **Sections 1–6 are on hold** until this section passes. They are
written for the two-stage recipe (`decoder_w = 0`, the current default). Once
the sweep has picked a weight, it becomes the default in `jepa/config.py`, and
every section below then trains jointly without changing its commands. The
separate decoder stages in sections 1 and 2 then drop out: the joint JEPA run
already trains its decoder.

### Sweep (1 epoch each, seed 1)

`--subset` is only 4 examples, too few to judge a representation, so the
sweep uses 1-epoch runs on the full data. `w = 0` is the control at the same
length. Write the weights you need as presets outside the repo
(`joint-decoder.json` is 0.3):

```sh
for w in 0 0.1 1.0; do echo "{\"decoder_w\": $w}" > runs/decoder-w$w.json; done
sbatch scripts/slurm_train.sh --config runs/decoder-w0.json --epochs 1 --seed 1 --output-dir runs/sweep-w0
sbatch scripts/slurm_train.sh --config runs/decoder-w0.1.json --epochs 1 --seed 1 --output-dir runs/sweep-w0.1
sbatch scripts/slurm_train.sh --config docs/experiments/joint-decoder.json --epochs 1 --seed 1 --output-dir runs/sweep-w0.3
sbatch scripts/slurm_train.sh --config runs/decoder-w1.0.json --epochs 1 --seed 1 --output-dir runs/sweep-w1.0
```

Measure each (`runs/sweep-w0` measures an untrained decoder; read only its
representation, loss, heads, policies and rollouts sections):

```sh
python -m jepa measure --model runs/sweep-w0.3 --json runs/sweep-w0.3/metrics.json   # likewise for each w
```

Choose the largest weight where, compared with `w = 0`, the JEPA terms
(`move`, `pi`, `mu`) still fall, effective rank stays > 15,
`no_periodic_shortcut` passes and `fast_share` drops clearly.

### Joint main run (seed 1, go/no-go)

Use the chosen weight as `--config`, or `joint-decoder.json` if 0.3 won:

```sh
sbatch scripts/slurm_train.sh --config docs/experiments/joint-decoder.json --seed 1 --output-dir runs/joint-seed1
python -m jepa measure --model runs/joint-seed1 --json runs/joint-seed1/metrics.json
```

Go for the grid if `no_periodic_shortcut` passes, μ clearly beats copy at
depth 1, live slots > 2 and `decoder.future_use` > 0. Then make the weight
the default, delete `joint-decoder.json`, and run sections 1–7. Keep
`runs/main-seed1` (two-stage) as the `decoder_w = 0` ablation, and add seeds
2 and 3 for it with `--config runs/decoder-w0.json`.

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

## 7. Figure 3: decoded plans (no training)

This needs sections 1 and 2 finished for seed 1. It decodes the main run's
plans (π-sample ×3, π-argmax, μ) next to the P = 1 model's plan, which the
P = 1 model's own decoder decodes.

```sh
python -m jepa generate --model runs/main-seed1-decoder --baseline-model runs/p1-seed1-decoder --plan-depth 4 --samples 3 --seed 0 --json runs/main-seed1-decoder/plans.json --html runs/main-seed1-decoder/plans.html
```

---

## Not planned

- **MDN predictor** (Table 2, optional): needs code that doesn't exist yet.
  Decide on it before adding runs.
- **Synthetic experiment** (§5.1): already done. It runs on CPU with
  `python docs/report/synthetic.py`.
