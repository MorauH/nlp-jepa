# 007: Random horizon per batch, hidden from the predictor

Status: implemented (decision recorded in ADR 0003; pending on the cluster: the gate runs of the experiment plan)

## Why

The joint run (`runs/joint-seed1`, `decoder_w = 0.3`, seed 1, 3 epochs) weakened
the periodic shortcut of ticket 006 but did not remove it.

| 64 validation documents (`measure`) | two-stage | joint |
|---|---|---|
| cos at lag 5 / 10 / 20 | −0.07 / 0.96 / 0.95 | 0.65 / 0.96 / 0.95 |
| periodic rebound (limit 0.05) | 1.02 | 0.31 |
| fast share | 0.88 | 0.53 |

On 256 validation documents, past the first 20 positions, document identity
explains 58% of the joint latent's variance (two-stage: 23%).

Away from the document start, copying z_t is as good as anything the predictor
proposes. Distances to z_{t+20} in the joint run, positions 80 and up (88% of
positions):

| copy z_t | μ | hindsight, best of 8 |
|---|---|---|
| 0.032 | 0.038 | 0.032 |

The two live slots have cos 0.9999 with each other and with z_t. Positions
below 40 make up 6% of positions but about 70% of the measured μ-hindsight gap,
so the averaged verdicts look better than the steady state.

The cause is the fixed k: a component whose period divides k costs nothing to
predict. Ticket 006 rejected a variable k because a periodic feature is
predictable from (z_t, k). That argument needs the predictor to know k. If k is
random and hidden, z_t no longer determines the phase of z_{t+k}.

The decoder blocks had the same blind spot. With blocks of exactly 20 tokens,
the hint only ever read z_t at multiples of 20, which is always the same phase
within a document. With a random block length the hint reads z_t at many
positions and phases.

### What the change can and cannot do

Estimated on the existing encoders (not retrained), positions 80 and up, k
uniform on 10..30, 256 validation documents. "Phase copies" are the latents
z_t, z_{t−1}, ..., spread over one period: an idealized set of candidates, not
a trained predictor.

| Distance to z_{t+k} | joint | two-stage |
|---|---|---|
| copy z_t, fixed k = 20 (today) | 0.032 | 0.021 |
| copy z_t, random k | 0.426 | 1.348 |
| one prediction: the cycle's mean, random k | 0.265 | 0.987 |
| best of 4 phase copies in hindsight, random k | 0.076 | 0.280 |
| best of 8 phase copies in hindsight, random k | 0.043 | 0.085 |

- **A single prediction pays for the cycle.** For P = 1 and μ the cyclic code
  becomes 8 to 47 times more expensive, so there is a real reason to drop it.
- **Winner-takes-all can absorb it.** Eight candidates covering the phases
  bring the hindsight distance back to 0.043, close to today's 0.032. Candidates
  would then stand for "which k" instead of "which future". π does not fix this:
  its target r is detached, so the π loss cannot push the encoder to remove the
  phase. The main run (P = 8) is therefore at risk where the P = 1 baseline is
  not, and the two could end up with different encoders.
- **A code that is constant per document stays free.** It does not depend on k.
  Only the decoder gradient works against it.

## Design

- **One horizon for everything.** k is the prediction target's offset, the
  decoder block length, the hint's block end, the hindsight target, the rollout
  stride, and the number of tokens `generate` writes per block. There is no
  second, fixed horizon anywhere: `pred_horizon` goes away.
- **One k per sequence of work.** Training draws one k per batch, uniform over
  the inclusive range `horizon_range = [lo, hi]`, shared by every position,
  block and document in it. Inference draws one k per generation, plan or
  rollout, so the decoder sees equal blocks within a sequence, as in training.
- **Random is the default.** `JepaConfig.horizon_range` defaults to `[10, 30]`:
  the mean is the old k and the range covers more than two periods of the
  observed cycle. `[20, 20]` is the fixed-k recipe, kept as the ablation.
- **Hidden.** The predictor stays a function of one latent: `predictor(z)`, no
  k argument. The encoder gets no k either. The decoder is not told k; it only
  sees where its blocks fall, through the hints.
- **Old checkpoints load as what they are.** They store `pred_horizon: 20` and
  no `horizon_range`, so the new default would silently turn them into
  random-k models. `JepaConfig` therefore accepts `pred_horizon` as a legacy
  constructor argument, meaning `horizon_range = [pred_horizon, pred_horizon]`.
  Giving both is an error. New checkpoints write `horizon_range` only.
  `apply_overrides` rejects `pred_horizon`, as it does any unknown key.
- **Gradients are unchanged.** The target is detached, the committed future in
  the hint is detached (ADR 0002), and the predictor gets no decoder gradient.
  The hint reuses the JEPA loss's `Futures` and target, as now.
- **Where k is drawn.** In the trainer, for both stages, from
  (seed, global_step) with a dedicated generator. Resume replays it without
  extra state and the global RNG is untouched.
- **Compile.** k must not trigger a recompile per value (21 values; the limit is
  64 and other shapes use it too). Pass it as a 0-dim tensor, like `progress`.
  Predict from every base t, gather the target at t + k (clamped) and mask
  positions with t + k past the end. `build_block_hints` indexes with the same
  tensor; a block is valid when `block_start + k <= seq - 1`.
- **Validation** must not depend on a random draw. Cycle k deterministically
  through the range by batch index, so each epoch's validation covers the same k
  values. Do this for both stages.

## What

1. **Config** (`jepa/config.py`): replace `pred_horizon` with `horizon_range`
   in `JepaConfig` and `from_encoder`, validate `1 <= lo <= hi`, and add the
   legacy argument.
2. **Model** (`jepa/models/jepa.py`): `jepa_loss` and `decoder_loss` take the
   horizon. Remove the `pred_horizon` property.
3. **Trainer** (`jepa/training/trainer.py`): draw k per step, the validation
   schedule, and the too-short check against `hi`.
4. **Generation** (`jepa/inference/generate.py`):
   - `generate` draws one k per call and uses it for every block.
   - `--plan-depth` draws one k per prompt, seeded by `--seed`, and shares it
     across every condition, sample and the baseline model, so the columns line
     up. Prompts need room for `N × hi` reference tokens; the reference uses the
     drawn k. Record k per prompt in the JSON and show it in the HTML.
   - The baseline check compares `horizon_range`.
5. **Measurement** (`jepa/evaluation/measure.py`). Latents, candidates and
   every rollout path except hindsight are independent of k, so compute them
   once and score them against each k.
   - **Evaluation horizons.** A fixed set over the range whose residues modulo
     10 all differ: 11, 14, 17, 20, 23, 26, 29 for `[10, 30]`. A grid in steps
     of 5 would always sit on the observed cycle. `[k, k]` gives just k, so a
     fixed-k checkpoint keeps its current numbers. Loss, heads, policies,
     rollouts and decoder average over these, and policies also report each k.
   - **Rollouts**: depth i is scored at t + i·k. Use the same starts for every
     k, so a start needs `depths × hi` tokens ahead.
   - **Decoder**: blocks of k tokens for each evaluation k, token-weighted.
   - **Max rebound.** Over every lag L in 2..hi: the largest cos(L) minus the
     lowest cos at a shorter lag. The current check looks at lag k only, so it
     misses a period that does not divide it. Move the `no_periodic_shortcut`
     verdict to it, keep the 0.05 limit, and extend the lag profile to hi. The
     joint run scores about 0.31 on both, because its period is 10.
   - **Fixed diagnostics.** Keep the fast-share window at 10 and the long lags
     at 40, 100, 200 and 400, so old and new runs compare. `temporal_cos` is
     the mean over the evaluation horizons.
   - **Steady-state rows.** Policies for positions 80 and up, next to the
     all-position ones.
   - **Slot-horizon mutual information** between the hindsight slot and k. Near
     0: the slots do not encode k. Near its maximum, ln 7 ≈ 1.9 nats with the
     seven evaluation horizons: the slots are phase copies. Report it; set a
     limit once there are runs. Report n/a for a fixed k or P = 1.
   - **Document share.** Between-document share of latent variance (23%
     two-stage, 58% joint), to see the per-document code grow or shrink.
6. **Presets** (`docs/experiments/`): `fixed-horizon.json`
   (`{"horizon_range": [20, 20]}`) and its README row. The other presets inherit
   the random default. Update the `joint-decoder.json` row: it now means joint
   with a random horizon.
7. **Report** (`docs/report/report.html`), see below.
8. **Docs**: `GLOSSARY.md` (**Horizon**: the number of tokens, random within
   `horizon_range`, between a latent and the future it aims at),
   `docs/architecture.md` (the module list, stages, hints, configuration JSON,
   measurement, generation, decoded plans), `AGENTS.md` (the `pred_horizon` and
   block-hint bullets, the decoded-plan command),
   `docs/experiments/RUNS.md` (§0 and the fixed-horizon ablation).

### Report changes

Every statement that assumes one fixed k:

- **Abstract and §1**: "a horizon of k = 20 tokens", "(80 tokens)" and the
  "k-token horizon" of contribution 1. Say that k is random per batch and not
  shown to the predictor, and give depth 4 as a range (40 to 120 tokens for
  10..30).
- **§3.1**: the target is `sg(z_{t+k})` with k drawn per batch.
- **§3.2**: state that the predictor does not see k. \(|\mu|\) and the spread of
  the candidates now mix uncertainty about content with uncertainty about
  distance.
- **§3.5**: a step advances by k tokens, drawn once per rollout; drop "k = 20"
  and "80 tokens". The manifold distance does not depend on k.
- **§3.6**: blocks of k tokens, k drawn per batch. A hindsight choice can also
  reveal k to the decoder, on top of the 3 bits; a π-argmax choice cannot.
- **§3.7**: add the joint run to the shortcut account: the decoder gradient
  alone took the rebound from 1.02 to 0.31. Then the random hidden horizon as
  the second change, why a hidden k is different from a known one, and the
  winner-takes-all caveat. Replace the "confirm the shortcut is gone" TODO with
  the result, or with a TODO until the gate runs are in.
- **§4.2, Table 1**: the "Horizon" row.
- **§4.3, Table 2**: a "Fixed horizon" run (`horizon_range = [20, 20]`, question:
  is the shortcut caused by a fixed k?). State that every other run uses the
  same range.
- **§4.4**: rollout error at depth i is \(d(z^{(i)}, z_{t+ik})\) averaged over
  the evaluation horizons. Add max rebound, slot-horizon mutual information and
  steady-state rows.
- **§4.5**: H1 to H4 keep their wording. Add one line saying that the horizon
  protocol was changed after two single-seed runs (two-stage and joint) and
  before the grid, so the hypotheses were fixed but the recipe was not.
- **§5.2**: the health table also reports max rebound, document share, mutual
  information, and all-position against steady-state rows.
- **§5.3, Table 4 caption**: "Depth i plans 20i tokens ahead" is wrong.
- **§5.4, §5.5, Figure 3**: "80 tokens each" depends on k; state k per prompt.
- **§5.6**: the fixed-horizon run joins the ablation table.
- **§6, "Candidate usage"**: live slots alone do not show that the candidates
  stand for different futures; discuss with the mutual information.
- **§7**: replace "one horizon (k = 20)" with one range, and add that the
  predictor cannot say how far ahead a candidate lies.
- **Reproducibility**: the commands, once they have changed.

Section 5.1 is not affected: the synthetic world steps by one state and has no
horizon. Do not write results that have not been run.

## Experiment plan

Gate with 1-epoch runs, seed 1, full data:

| Run | horizon | decoder_w | P |
|---|---|---|---|
| control | fixed 20 | 0.3 | 8 |
| A | random | 0.3 | 8 |
| B | random | 0.3 | 1 |
| C | random | 0 | 8 |

A is `joint-decoder.json`. The control, B and C need two overrides or a
different weight; write those presets outside the repo, as `RUNS.md` §0 does.

B isolates the mechanism: with one candidate nothing can absorb the cycle. A
against B shows whether winner-takes-all does. C shows what the change does
without the decoder gradient (its decoder is untrained; ignore that section).

**Go/no-go for a full run** of A, read on positions 80 and up:

1. `no_periodic_shortcut` passes (max rebound < 0.05);
2. μ clearly beats copy, averaged over k;
3. live slots > 2, **and** slot-horizon mutual information is low. Alone, live
   slots is not evidence: phase copies would make all 8 live;
4. document share no higher than the control's;
5. `future_use` > 0.

If B passes and A fails on 1 or 3, winner-takes-all is absorbing the cycle. The
fix to try then (a separate ticket, not part of this one) is to score each
candidate against two independently drawn horizons and assign on the average of
the two distances: one phase copy cannot fit both. Note that this also rewards
slow latents.

If B fails as well, the shortcut is not about the fixed k. Stop and revisit.

## Risks

- **Candidates now carry uncertainty about k.** Spread among candidates and |μ|
  mix "which future" with "how far ahead". This weakens the reading of H1 to H4:
  the report must say so, and it is why the mutual information is measured.
- **A committed future has no known distance.** Rollouts and decoded blocks use
  a drawn k, but the committed latent stands for "some k tokens ahead". Rollout
  error and decoder cross-entropy get noisier for every path. The manifold
  distance does not depend on alignment.
- **The decoder has a harder job.** Its blocks vary in length and their hint
  ends at a point the predictor cannot pin down, so `future_use` may fall.
- **Confound with the baseline.** P = 1 and P = 8 must use the same
  `horizon_range` and `decoder_w`, and their encoders may still differ in
  periodicity. Report each encoder's rebound next to the comparison.
- **Loss values are not comparable** across fixed and random horizons. Compare
  `measure` numbers only.
- **The default changes before it is validated.** Any plain `train` is now a
  random-k run. Until the gate passes, do not start the grid.

## Acceptance criteria

- `horizon_range = [20, 20]`: JEPA and decoder losses and logged keys match the
  current code on a fixed seed on CPU (within 1e-5, since the predictor now
  also runs on the tail positions), and the global RNG stream is unchanged.
- A config with `pred_horizon: 20` and no `horizon_range` loads as `[20, 20]`.
  Both fields together raise. `runs/main-seed1-decoder` and `runs/joint-seed1`
  load and `measure` as fixed-k models.
- A fresh `JepaConfig()` has `horizon_range = [10, 30]`, and no code reads
  `pred_horizon` outside the legacy argument.
- Tests:
  - the predictor never receives k: with spies, the same latents give the same
    predictor call for different k;
  - target, hint block end, decoder block length and hindsight target all use
    the batch's k, the target is detached, and positions with no real token at
    t + k are masked;
  - the hint for a block does not depend on tokens after the block start, for
    several k (today's leakage tests, parametrized);
  - the draw is a function of (seed, global_step): a resumed run uses the same
    k sequence as an uninterrupted one, in both stages;
  - validation uses the same k sequence every epoch;
  - `generate` and plans use one k per sequence, and the same seed gives the
    same k;
  - on a synthetic period-10 latent, the copy distance at random k exceeds the
    one at k = 20;
  - max rebound detects period 7 and 10, and not a plain drift;
  - the mutual information is near 0 when slots ignore k and large when the
    slot is a function of k.
- No recompilation per k on CUDA: report the recompile count from a short run.
- The tests that use `pred_horizon` (`test_smoke.py`, `test_measure.py`,
  `test_joint.py`, `test_generate.py`) are updated, and the suite is green.
- `docs/report/report.html` has no statement left that assumes k = 20 for the
  main run (historical mentions of the two fixed-k runs excepted).
- ADR 0003 no longer says "Proposed", and the docs above are updated.
