# 008: The decoder reads the real future latent, blurred to the predictor's accuracy

Status: implemented (pending on the cluster: the gate run in `RUNS.md` §0b).
The optional independent probe (a frozen sentence encoder) is not
implemented.

## Why

The random-horizon joint run (`runs/random-joint-seed1`, `decoder_w = 0.3`,
seed 1, 3 epochs) did what ticket 007 asked: the periodic shortcut is gone and
the candidates do not encode k. It did not make the latent carry content.
All three runs measured with the current `measure` (64 validation documents):

| | two-stage, k = 20 | joint, k = 20 | joint, random k |
|---|---|---|---|
| max rebound (limit 0.05) | 1.03 | 0.31 | 0.0009 |
| document share | 0.22 | 0.60 | 0.81 |
| effective rank | 41 | 38 | 26 |
| cos at lag 400 | 0.38 | 0.56 | 0.77 |
| steady state: copy / μ / hindsight / π-argmax | .021 / .024 / .020 / .020 | .030 / .036 / .030 / .031 | .123 / .077 / .032 / .109 |
| π gain (nats) | 0.108 | 0.094 | 0.071 |
| decoder: no hints − π-argmax / commit headroom | 1.47 / 0.0014 | 0.077 / 0.0006 | 0.119 / 0.0021 |

- **The candidates now beat μ in hindsight** (0.032 vs 0.077 from position
  80 on), and the best of 16 rollouts beats μ. But π cannot choose: accuracy
  0.536 against 0.522 for the majority slot, about 2 of 8 slots live. Committing
  with π-argmax (0.109) is worse than μ.
- **The latent is turning into a per-document constant.** It is free to predict
  at any k, and nothing pays the encoder for anything else (ADR 0004 traces the
  three gradient paths). The decoder gradient through z_t adds to it: z_t is
  rewarded for what the 32-token window lacks, which is document context.
- **The decoder ignores which future was committed.** Hindsight and π-argmax
  give the same loss within 0.002 nats, while a shuffled hint costs 0.128. The
  decoder reads the hint for document identity, not for the future.
- **`encoder_center` fails as a side effect.** `measure` uses batches of 4
  documents (`jepa/evaluation/measure.py:45`). With 81% of the variance between
  documents, the expected center term is about 0.81 × 0.96 / 4 ≈ 0.19 (0.23
  measured), against 0.026 in training batches. VISReg is not broken.

## Design

- **Hint mode.** A top-level `JepaConfig.decoder_hint`, next to `decoder_w`:
  `"committed"` (default, ADR 0002, today's behavior) or `"real"` (ADR 0004).
  It only matters in the joint JEPA stage (`decoder_w > 0`). It is not a
  `decoder.*` field, because the decoder stage never uses it. Old checkpoints
  have no field and load as `"committed"`.
- **Gradient paths with `"real"`.** The hint is (z_t detached, ẑ_{t+k}).
  ẑ_{t+k} is the online block-end latent plus noise, not detached, so the
  token loss reaches the encoder only through the block end. The committed
  future does not enter the objective, and the predictor is trained by the
  commitment loss alone.
- **Leak.** The block-end latent has read the whole block. That is the point
  (reconstruction), and it never exists at inference. `build_block_hints`
  already documents that the raw block-end latent leaks the block. The
  no-leak tests for z_t stay; add a test that z_t carries no gradient and
  ẑ_{t+k} does.
- **Noise.** ẑ = z + s · (‖z‖ / √D) · ε, ε ~ N(0, I) per position, with
  s² = 1 / (1 − d/2)² − 1 and d the batch's detached `hindsight_d` (the
  commitment loss already computes it), clamped to [0, 1]. This gives
  E[distance(ẑ, z)] ≈ d. Draw ε from the global RNG: it is restored on resume
  like the rest of training. Log `s` as a diagnostic (`hint_noise`).
  `"real"` with `decoder_w == 0` is an error, as is noise without the real hint.
- **Validation** keeps the π-argmax hint (the inference condition) for
  perplexity, as now, and adds the noise-free real hint as a second number, so
  the gap between rendering a predicted and a real future is visible per epoch.
- **Compile.** `s` is a 0-dim tensor computed in the graph; no new shapes.

## What

1. **Config** (`jepa/config.py`): `decoder_hint` with validation, the legacy
   default, and `apply_overrides` support.
2. **Model** (`jepa/models/jepa.py`): the `"real"` branch in `jepa_loss`,
   reusing `build_block_hints(latents, horizon, committed=None)` for the
   block-end latent. Detach z_t only in this branch. The `"committed"` branch
   is unchanged.
3. **Trainer** (`jepa/training/trainer.py`): log `hint_noise`, the extra
   validation number.
4. **Measurement** (`jepa/evaluation/measure.py`):
   - **Decoder conditions.** Add `real_noisy` (the training condition at the
     measured hindsight distance). Add `render_gap = pi_argmax − real`: what
     decoding a predicted future costs against the real one. Keep
     `commit_headroom = pi_argmax − hindsight`; this is the number the
     commitment claim needs to grow.
   - **Next-block probe.** A linear probe, fit on the train split and scored on
     validation, from a latent to the bag of words of the next block (k tokens,
     at each evaluation horizon). Score z_t, μ, the π-argmax and the hindsight
     candidate. Report the gain of each over z_t, and of the hindsight candidate
     over μ. This is the first metric for "the future carries content". It is
     not trained into the model in either hint mode.
   - **Independent probe (optional).** A frozen pretrained sentence encoder
     embeds the real next block. Does the hindsight candidate retrieve it among
     the document's other blocks better than μ does? Fit only as an evaluation;
     report n/a when the model is not available offline.
   - **Center verdict.** Compute `encoder_center` from the pooled latents, not
     per 4-document batch, or say in the output that it scales with document
     share / batch size. The current number mixes the two.
5. **Preset** (`docs/experiments/`): `joint-real-hint.json`
   (`{"decoder_w": 0.3, "decoder_hint": "real"}`) and its README row.
   `RUNS.md`: the gate below.
6. **Docs.** `GLOSSARY.md` (**Hint**, **Joint JEPA stage**: two hint modes),
   `docs/architecture.md`, AGENTS.md's joint-stage gotcha.
7. **Tests.** Gradient paths for `"real"` (z_t none, block-end latent yes,
   predictor only through the commitment loss) next to `tests/test_joint.py`.
   The noise level matches d on random vectors. `"committed"` still passes the
   existing tests unchanged.

## Gate

One run, 3 epochs, seed 1, against `runs/random-joint-seed1` (same everything
except `decoder_hint`). Limits are provisional; set them once there are runs.

| Check | Today | Pass |
|---|---|---|
| document share | 0.81 | below 0.60 (the fixed-k joint run) |
| effective rank | 26 | above 38 |
| max rebound, slot-horizon MI | 0.0009, 0.0001 | stay below 0.05 and near 0 |
| next-block probe, hindsight candidate over μ | n/a | above 0 |
| commit headroom | 0.002 nats | clearly above 0.002; report it with `render_gap` |
| steady state μ vs copy | 0.077 vs 0.123 | μ still beats copy |

If document share does not fall, there are two fallbacks, each a separate
decision recorded in a new ADR:

1. Detach z_t in the residual candidates (sg(z_t) + δ). The move loss then no
   longer pulls z_t toward z_{t+k} directly; it only trains the predictor and
   reaches the encoder through the trunk's input. Narrowest change: try first.
   Written up as ticket 009.
2. Let the JEPA gradient reach the target through the move term only (the
   winner-takes-all distance), with μ, π and diversity detached.

## Not in this ticket

- Choosing the winning candidate by decoder loss instead of latent distance,
  with decoder gradient into the predictor (ADR 0004, alternatives).
- Re-measuring π-sample with `top_k` restricted to live slots. Worth a
  `measure` flag independently, since the dead slots may explain the failed
  manifold verdict for π-sample (it is closer to the manifold than μ at depth 1,
  0.490 vs 0.504, and drifts after).
