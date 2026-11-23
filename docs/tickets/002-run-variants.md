# 002: Run baselines, ablations and seeds without editing code

Status: done

## Why

"Config defaults are the experiment" keeps the main run honest, but the report
needs other runs that are trained identically except for one setting:

- **P = 1 baseline** (`predictor.prediction_count = 1`): an ordinary
  single-prediction JEPA. This is the external baseline the claim is judged
  against. On the synthetic chain it collapses to the blurry mean, exactly like
  μ (`docs/report/synthetic.json`).
- **Ablations**: P = 4, no annealing (`commitment.temperature_start` equal to
  `temperature`), no diversity (`commitment.diversity_w = 0`), and a decoder
  trained on π-argmax hints (`decoder.train_policy = "pi_argmax"`).
- **Seeds**: at least 3 per configuration for error bars. `SEED` is currently
  a trainer constant, so seeds can't vary without editing code.

## What

1. **`--config PATH`** on `python -m jepa train`: a JSON file with the same
   nesting as a checkpoint's `config.json`, holding only the fields to change.

   ```json
   {"predictor": {"prediction_count": 1}}
   ```

   - **JEPA stage, fresh run:** overrides apply to the default `JepaConfig`
     before the model is built. They end up in the saved checkpoint
     automatically, so every checkpoint stays a complete record of its
     experiment.
   - **Decoder stage, fresh decoder:** only `decoder.*` overrides are allowed
     (e.g. `train_policy`). Anything else is an error, because the JEPA is
     frozen and already defined by `--from`.
   - **With `--resume`:** passing `--config` is an error. The checkpoint's
     config is the run's config.
   - **Unknown keys fail loudly.** `PretrainedConfig` silently accepts
     arbitrary kwargs, so a typo like `prediction_cout` would otherwise train
     the default model. Check every override key against the target config's
     existing attributes, and let `JepaConfig.link()` validate the result.
   - Keep the override logic in one small function, `apply_overrides(config,
     overrides) -> config`, next to `JepaConfig` in `jepa/config.py`.

2. **`--seed N`** on `train`: becomes a `TrainConfig` field, defaulting to
   `SEED`. It seeds model construction in `train.py` and the trainer's stream.
   Save it in `trainer_state_<stage>.pt` and in the history, so `--resume`
   keeps the original seed.

3. **Presets** in `docs/experiments/` (one JSON file each):
   `p1-baseline.json`, `p4.json`, `no-anneal.json`, `no-diversity.json`,
   `decoder-pi-argmax.json`. Add a short README table that maps each preset to
   the question it answers in the report.

4. **Check `measure` on a P = 1 checkpoint.** π, live slots and candidate pair
   cosine are degenerate with a single candidate. The report should show them
   as n/a rather than as misleading numbers, so either print them that way or
   leave them out of the P = 1 report.

## Acceptance criteria

- `train --config docs/experiments/p1-baseline.json --subset --output-dir
  /tmp/x` writes a checkpoint whose `config.json` has `prediction_count: 1`.
- A typo in an override key raises before any training starts. So do a
  non-decoder override in the decoder stage and `--config` with `--resume`.
- Two `--subset` runs with different `--seed` values give different weights,
  and two runs with the same seed give identical weights.
- `measure` completes on a tiny P = 1 checkpoint (test with the tiny config).
- `AGENTS.md` and `docs/architecture.md` describe the flags. Update the AGENTS
  rule "config defaults are the experiment" to "config defaults are the main
  experiment; variants are override files in `docs/experiments/`".
