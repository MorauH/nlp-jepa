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
| `fixed-horizon.json` | jepa | `horizon_range = [20, 20]` | Is the periodic shortcut caused by a fixed k? The default draws k from `[10, 30]` once per batch and hides it from the predictor. See ADR 0003 and ticket 007. |
| `joint-decoder.json` | jepa | `decoder_w = 0.3` | Joint training with a random horizon (the default range). Does a decoder gradient into the encoder (through the hint z_t), together with a hidden random k, remove the periodic shortcut and make the latent carry content? With a fixed k it only weakened it. See ADR 0002, ADR 0003 and tickets 006 and 007. Temporary: once the `decoder_w` sweep picks a weight, it becomes the default and this preset is deleted. Until then, the other presets stay two-stage. |
| `joint-real-hint.json` | jepa | `decoder_w = 0.3`, `decoder_hint = "real"` | Does decoding the real future latent, blurred to the predictor's hindsight distance, make the latent carry content instead of document identity? The token loss reaches the encoder through the block end, not z_t. See ADR 0004 and ticket 008; the gate is in `RUNS.md` §0b. Compare with `joint-decoder.json` (same seed). |
| `decoder-pi-argmax.json` | decoder | `decoder.train_policy = "pi_argmax"` | Does training the decoder on the inference hint instead of hindsight close the commit headroom? |

Every preset except `fixed-horizon.json` inherits the random horizon. Compare
runs across horizon ranges with `measure` only: their loss values are not
comparable.

Run each configuration with at least 3 seeds (`--seed`) for error bars.
`--resume` keeps the run's config and seed, so neither flag is allowed with it.

With `prediction_count = 1`, `measure` reports π, live slots, candidate
pair cosine and the slot-horizon mutual information as n/a, and skips the `pi_uses_z_t` verdict.
