# 006: Train the encoder through the decoder's block-start hint

Status: implemented (pending on the cluster: the peak-VRAM check, the decoder_w sweep and the go/no-go run in RUNS.md §0)

## Why

The first full run (`runs/main-seed1-decoder`, two-stage, P = 8) shows that the
encoder found a positional shortcut instead of encoding content.

- **The latent oscillates with a period of 10 tokens**, which divides
  `pred_horizon = 20`. Mean cos(z_t, z_t+lag) over 29 validation documents:

  | lag | 1 | 3 | 5 | 10 | 15 | 19 | 20 | 21 | 25 | 30 | 100 | 200 | 400 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | cos | 0.74 | −0.05 | −0.15 | 0.97 | −0.15 | 0.68 | 0.96 | 0.75 | −0.16 | 0.95 | 0.88 | 0.71 | 0.34 |

  93% of the variance within a document is this oscillation (z minus its
  10-token moving average). The slowly changing part, which would carry
  content, is 7%. `measure` only reports cos at lag k (0.948), which hides it.
- **Why the objective allows it:** VISReg wants the batch of latents spread
  isotropically over 448 dimensions. A batch holds 24 documents, and their
  means span an effective rank of about 15, so VISReg forces variation within
  each document. Content that changes costs prediction error. A component
  whose period divides k costs none, because the predictor copies it. Nothing
  in the JEPA stage rewards the encoder for carrying information about the
  tokens.
- **Consequences in the measured numbers:**
  - μ barely beats copying z_t (0.095 vs 0.104, depth 1);
  - only 2 of 8 slots are live, and the two live candidates have cos ≈ 0.97
    with each other and with z_t;
  - π is nearly constant (median π of the top slot 0.482, interquartile range
    0.480–0.483);
  - the decoder gains only 0.076 nats from the committed future over no future
    (`null` − `pi_argmax`).
- **Neither a variable k nor larger batches fixes this.** Any feature that
  depends on position is predictable from (z_t, k), so it stays cheap for the
  JEPA loss. More documents per batch remove the pressure to vary within a
  document but reward a code that is constant per document (topic) instead.
  The encoder needs a term that only content can satisfy.

The decoder loss is such a term. Letting its gradient reach the encoder
through the block-start hint z_t rewards z_t for carrying what a 32-token
window lacks and what helps predict the next k tokens. Blocks start at
multiples of k, so the oscillation phase is the same at every hint and
carries nothing the decoder can use.

This reverses ADR 0001 (no joint training). The evidence above is the reason.

## Design

- **Gradient paths.** The decoder loss reaches the encoder through z_t only.
  The committed future is detached. A hindsight commit is chosen using the
  real future, so a gradient through it would train the predictor and encoder
  on information that does not exist at inference, and it would pull the
  candidates away from the JEPA target. The predictor is trained only by the
  JEPA loss.
- **No leakage through z_t.** `build_block_hints` (`jepa/models/jepa.py`) pairs
  input position i, which predicts token i+1, with the latent at
  `block_start = (i // k) * k`. That latent has read tokens up to
  `block_start` only, and every target token comes after it. Keep a test that
  pins this down.
- **One objective:** `VISReg + CommitmentLoss + decoder_w × decoder token
  loss`, in one forward pass of the encoder. The hint's committed future
  reuses the JEPA loss's `Futures`: commit with the decoder's `train_policy`
  (hindsight by default) against the detached targets, then `.detach()`. Do not
  run the predictor a second time.
- **`decoder_w = 0` reproduces today's JEPA stage exactly**: no decoder
  forward, so no extra compute and identical logs. That setting is the
  ablation "does a decoder gradient remove the shortcut?".
- **Where the weight lives.** It is a top-level `JepaConfig` field
  (`decoder_w`), like `pred_horizon`, so `--config` can override it and the
  checkpoint records it. The default is 0 until the sweep below picks a value.
  That way the existing two-stage checkpoints, which have no `decoder_w` in
  their `config.json`, load with the weight they were trained with. The joint
  recipe is the preset `joint-decoder.json`. Once the sweep has chosen the
  weight, the default switches to it and the preset is deleted.
- **Stages.** With `decoder_w > 0` the `jepa` stage trains encoder, predictor
  and decoder together and logs `jepa_*` terms plus `jepa_token`. The
  `decoder` stage stays. It is still needed for the π-argmax-hint decoder
  ablation, and for refitting a fresh decoder on a frozen joint encoder if the
  report wants that comparison.

## What

1. **ADR 0002** superseding ADR 0001. It records the shortcut evidence, the
   detach rule, what `decoder_w = 0` means, and what happens to the decoder
   stage. Update ADR 0001's header to point at it.
2. **Model.** Add `decoder_w` to `JepaConfig`. In `JepaModel.jepa_loss`, when
   `decoder_w > 0`:
   - build the hints from the online latents, keeping the gradient on z_t;
   - commit on the existing `Futures` and detach the result;
   - add the decoder token loss as a weighted term;
   - apply the decoder-loss masking already used by `decoder_loss`.
3. **Trainer** (`jepa/training/trainer.py`). The `jepa` stage's trainable
   modules include `decoder` when `decoder_w > 0`.
   - Validation computes the token term with π-argmax hints, the inference
     condition, as the decoder stage does. Perplexity comes from the token
     term, not from the total loss. The hindsight condition is reported by
     `measure`.
   - `torch.compile`, AMP and the "save from the uncompiled module" rule apply
     as before.
   - Check peak VRAM at batch size 24 on an L40S. The last run peaked at 16 GB
     for the JEPA stage and 4.5 GB for the decoder stage.
4. **Measurement** (`jepa/evaluation/measure.py`):
   - **Lag profile:** mean cos(z_t, z_t+lag) for every lag 1..k, plus 2k,
     5k, 10k and 20k, in `representation`. The dense part is what the verdict
     below needs.
   - **Fast-oscillation share:** on unit latents, the energy of z minus its
     centred k/2-step moving average (fast), divided by fast plus the energy
     of that moving average around its document mean (slow). This is the
     split behind the 93% figure above. The window is centred so that a plain
     drift counts as slow, and it is k/2 rather than a hard-coded 10.
   - **A `no_hints` decoder condition** (z_t and future both zero) next to the
     existing `null` (real z_t, zero future). Report the z_t channel's use as
     `no_hints − null` and the future channel's use as `null − pi_argmax`. H4
     (`shuffled − pi_argmax`) is unchanged. Only `null − pi_argmax` shows
     whether the commitment adds anything beyond z_t.
   - **A verdict `no_periodic_shortcut`:** the rebound, cos(lag k) − min over
     lags 1..k−1 of cos(lag), is < 0.05. A latent that drifts loses
     similarity with lag, so its rebound is ≤ 0. A latent whose period divides
     k dips somewhere below k and rebounds at k. `runs/main-seed1-decoder`
     scores 0.96 − (−0.16) ≈ 1.1; the minimum is near lag 5, a quarter of k.
     The 0.05 allows for sampling noise. An earlier version compared lag k/2
     with lag k. That misses this run: the period is 10 = k/2, so lag k/2 is
     itself a full period (cos 0.97). The test checks for periodicity directly,
     so it does not depend on how much fast variation a healthy encoder has. The
     fast-oscillation share is reported only, with no verdict: that healthy
     level is unknown, and the share also counts legitimate changes from
     token to token.
5. **Presets** in `docs/experiments/`: `joint-decoder.json` (`decoder_w`
   above 0, the sweep's starting value), plus the README row mapping it to its
   report question. The default, `decoder_w = 0`, is the two-stage ablation.
6. **Docs.**
   - `docs/architecture.md`: two-stage → joint, the gradient paths.
   - `AGENTS.md`: the stage bullets, the decoder-stage freezing, "the JEPA
     stage never runs the decoder".
   - `docs/report/report.html` §3.7 and §4.2.
   - `docs/experiments/RUNS.md`: the main recipe now trains the decoder
     jointly, the separate decoder stage drops out of the main and P = 1
     sections, and the two-stage runs become the `decoder_w = 0` ablation.

## Experiment plan

1. Sweep `decoder_w` ∈ {0, 0.1, 0.3, 1.0} with 1-epoch runs on the full data
   (`--subset` is only 4 examples, too few to judge a representation; commands
   in `RUNS.md` §0). Choose the largest
   weight for which, compared with `decoder_w = 0`:
   - the JEPA terms (`move`, `pi`, `mu`) still fall;
   - effective rank stays > 15;
   - `no_periodic_shortcut` passes;
   - the fast-oscillation share drops clearly.
2. One full main run with seed 1. Go/no-go before the other configurations:
   - `no_periodic_shortcut` passes;
   - μ clearly beats copy at depth 1;
   - live slots > 2;
   - `null − pi_argmax` > 0.
3. Then run the full grid in `RUNS.md` under the joint recipe. P = 1 and every
   JEPA ablation must use the same `decoder_w`, otherwise differences between
   configurations are confounded with the recipe. Hold the remaining
   two-stage runs until step 2 passes.

## Risks

- **The decoder may ignore the committed future.** A z_t trained by the
  decoder can carry most of what the decoder needs, so H4 could get weaker.
  The `null − pi_argmax` row measures exactly this. If it is near zero, that
  is a finding for the report, not a bug to tune away.
- **The decoder gradient can take over the encoder** and turn it into a plain
  language-model encoder. The sweep and the JEPA-term checks guard against
  this.
- **The decoder sees a moving hint while the encoder trains**, which was ADR
  0001's original objection. That co-adaptation is now the intent. Note it in
  the ADR.

## Acceptance criteria

- With `decoder_w = 0`, the JEPA stage's losses and the logged keys match the
  current code on a fixed seed on CPU.
- Tests:
  - with `decoder_w > 0`, the decoder loss produces non-zero gradients on the
    encoder;
  - the predictor gets zero gradient from the decoder term (a test that
    isolates that term);
  - the committed hint does not require grad;
  - changing tokens at positions > `block_start` does not change the hint z_t
    for that block (no leakage).
- `measure` returns the lag profile, the fast-oscillation share, the
  `no_hints` condition and the new verdict. `tests/test_measure.py` covers the
  keys.
- `tests/test_research_claim.py` and `tests/test_measure.py` stay green.
- ADR 0002 is written, and the docs listed above are updated.
