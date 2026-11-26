# 009: Stop the move loss from pulling z_t toward z_{t+k}

Status: moot. Its condition was not met: ticket 008's run dropped document
share to 0.04 (`RUNS.md` §0b). ADR 0005 took the second fallback instead.

## Why

The predictor builds each candidate as a correction on top of the current
latent (`jepa/models/predictor.py:216`):

    candidate = z_t + δ(z_t)

The move loss pulls each candidate toward the detached target z_{t+k}. Its
gradient has two routes into the model:

1. **Through δ.** The predictor learns a better correction, and the encoder
   learns to make z_t useful for computing it. This is the intended learning.
2. **Through the "+ z_t" itself (identity path).** The encoder moves z_t
   closer to z_{t+k}, so the candidate gets closer without the predictor
   learning anything.

Route 2 tells the encoder, at every position, "look like the latent 10 to 30
tokens later". The cheapest way to satisfy it is a latent that hardly
changes, such as one vector per document. The random-horizon joint run fits
this: document share 0.81, cos 0.92 between latents 10 tokens apart and 0.77
at 400 tokens. So the objective does not merely fail to reward content; it
actively rewards slowness.

Ticket 008 (ADR 0004) adds the opposite pull, a decoder reward for the block
end to summarize what it read. If route 2 is stronger, the per-document code
survives that change. This ticket removes route 2.

## Design

- **One switch.** `PredictorConfig.detach_residual: bool = False`. When true,
  the predictor builds `candidate = sg(z_t) + δ(z_t)`: the same value, but no
  gradient through the identity path. Old checkpoints have no field and load
  as `False`. It is a `predictor.*` override, so it works with
  `train --config` in the JEPA stage.
- **What still reaches the encoder.** δ, π and the trunk read z_t without
  detaching, so the move, π and diversity losses still train the encoder
  through the trunk: "be useful for predicting the future", not "be the
  future already". μ is detached already. VISReg and the decoder term are
  unchanged.
- **Diversity loses its identity path too.** All candidates share the same
  z_t, so its gradient through the sum is the sum over candidates; it is cut
  by the same detach. Note it, no extra change.
- **Forward values are unchanged.** Rollouts, generation and `measure` see the
  same candidates for the same weights. Only training differs.
- **The residual stays.** Removing it (candidate = δ(z_t)) would also change
  what the predictor has to learn at initialization. Out of scope.
- **P = 1.** The switch applies the same way, so the P = 1 baseline can be run
  with it for a fair comparison.

## What

1. **Config** (`jepa/config.py`): the field, its default and validation.
2. **Predictor** (`jepa/models/predictor.py`): `z_t.detach()` in the residual
   when the flag is set, with a comment that points to this ticket and the ADR.
3. **Preset** (`docs/experiments/`): `detach-residual.json`, on top of
   ticket 008's preset (`{"decoder_w": 0.3, "decoder_hint": "real",
   "predictor": {"detach_residual": true}}`), and its README row.
4. **Tests**: with the flag set, the move loss gives the encoder latents a
   gradient only through the trunk. Check it with the δ head's weights at
   zero (so δ does not depend on z_t) and the move term alone: with the flag
   the gradient on z_t is zero, without it nonzero. Existing tests pass unchanged with the default.
5. **Docs**: `docs/architecture.md`, the move-loss docstring in
   `jepa/models/losses.py`, GLOSSARY if a term is added.

## Gate

One run, 3 epochs, seed 1, against ticket 008's gate run (identical except
for the flag).

| Check | Pass |
|---|---|
| document share | lower than ticket 008's run, and below 0.60 |
| lag profile | cos at lag 10 and 400 lower than ticket 008's run |
| max rebound, slot-horizon MI | stay below 0.05 and near 0 (no shortcut returns) |
| effective rank | not lower than ticket 008's run |
| steady state μ vs copy | μ still beats copy |
| next-block probe, hindsight candidate over μ | not lower than ticket 008's run |

## Risks

- **The latent may become fast and noisy.** With route 2 gone, the only
  pressure toward a predictable latent is the trunk path. If μ and hindsight
  distances rise a lot while the probe does not improve, the latent is
  changing without carrying content. Compare against copy.
- **A slow latent is partly wanted.** Meaning should change more slowly than
  words. Route 2 rewards slowness of any kind, which is the problem, but the
  run may lose some useful smoothness. The lag profile shows how much.
- **It changes the core JEPA objective.** Earlier runs compare only through
  `measure`.
- **The recipe changes again after results.** The report states it, as for
  ADRs 0003 and 0004.

## Option

If compute allows, run the 2 × 2 instead of one run: hint mode (committed,
real) × residual (online, detached). It separates the effect of the decoder
reward from the effect of removing the pull toward slowness.
