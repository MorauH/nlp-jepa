# Experiment presets

The config defaults are the main experiment. Each file here is a variant: a
JSON override nested like a checkpoint's `config.json`, holding only the
fields that change. Everything else, including the training recipe, is the
main run's.

```sh
python -m jepa train --config docs/experiments/p1-baseline.json --output-dir runs/p1-seed1 --seed 1
python -m jepa train --stage decoder --from runs/jepa --config docs/experiments/decoder-pi-argmax.json
```

| Preset | Stage | Change | Question it answers |
|---|---|---|---|
| `p1-baseline.json` | jepa | `predictor.prediction_count = 1` | Does committing among candidates beat an ordinary single-prediction JEPA? (the external baseline) |
| `p4.json` | jepa | `predictor.prediction_count = 4` | How much does the number of candidates matter? |
| `no-anneal.json` | jepa | `commitment.temperature_start = 0.1` (= `temperature`) | Do candidates need annealed winner-takes-all to spread over the modes? |
| `no-diversity.json` | jepa | `commitment.diversity_w = 0` | Does the diversity penalty prevent duplicate candidates? |
| `decoder-pi-argmax.json` | decoder | `decoder.train_policy = "pi_argmax"` | Does training the decoder on the inference hint instead of hindsight close the commit headroom? |

Run each configuration with at least 3 seeds (`--seed`) for error bars.
`--resume` keeps the run's config and seed, so neither flag is allowed with it.

With `prediction_count = 1`, `measure` reports π, live slots and candidate
pair cosine as n/a, and skips the `pi_uses_z_t` verdict.
