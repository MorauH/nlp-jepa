# Predictor vision (long-term, not implemented)

Where the predictor is headed. See `docs/architecture.md` for the current code;
this file is planning, not a description of implemented behavior.

## Where we are

- The predictor maps the single latest latent `z_t` to `P` candidate futures
  `z_{t+k}`. The causal encoder already summarizes the prefix in `z_t`, so a
  trunk over past latents duplicated capacity and let the predictor do the
  encoder's job.
- A commit uses π, the win-probability distribution ("which candidate will be
  closest"), and generation samples from it. Judged by expected distance it
  can look worse than a constant index; that is expected, because under the
  normalized metric `E[d_i | z_t] = 2 − 2 ĉ_i·μ`. Committing by expected
  distance is minimized by the candidate nearest the mean future, which a
  single mean predictor already achieves. Multiple candidates only pay off
  through hindsight (decoder training) and through sampling distinct futures.
- μ, a mean-future head (`E[ŷ | z_t]`, detached input), is kept as an
  instrument: `|μ|` is the per-context uncertainty, μ is the in-model
  single-head baseline (hindsight gain over μ = what P candidates buy).

## Known open problem: dead slots

Typically only ~2 of the 8 slots are ever committed. Under the relaxed
winner-takes-all move loss, a slot that stops being the closest only gets the
tiny `smoothing / P` pull (toward the conditional mean, where the live heads
already are) and the diversity push away from them. It parks as a "shadow" and
does not recover. If new data adds a mode, the nearest live head stretches to
cover both modes (the k-means one-centre-for-two-clusters minimum) long before
a dead slot is recruited. More data or more slots alone will not create a third
useful head.

## Plan: head revival / growth

1. **Measure first** (`python -m jepa measure`). Hindsight gain over μ, π's
   likelihood gain over a constant distribution (is the closest candidate
   predictable from `z_t` at all?), and `|μ|`. If the hindsight gain is small,
   multiple heads are not justified at this horizon and target; consider a
   smoother target (block mean of `z_{t+1..t+k}` or a chunk embedding) first.
2. **Grow by splitting, not by usage.** Start with 2 active slots out of a
   preallocated `P_max` (active mask in the softmax, the temperature, the
   diversity term and the smoothing; no shape changes, so no recompiles or
   optimizer resizing).
   Track per-slot distortion `E[d | hindsight commits to slot i]`. Split head
   `i` when the targets assigned to it show two-cluster structure (2-means gain
   or a G-means style normality test along the main axis): copy its
   `MatrixExpansion` / `PerDimLinear` slices into a free slot, add noise, split
   its targets. Merge near-duplicates. Prior art: LBG splitting, DP-means,
   G-means. "Both heads are used" is not a split criterion: even usage does
   not imply a second mode exists.
3. **Fallbacks:** dead-slot restart (VQ-VAE codebook style) from the
   highest-distortion head; re-heat the assignment temperature when the data
   changes; recruit dead slots toward the worst-fit targets.
4. **μ as the growth trigger.** π's entropy is uncertainty among the existing
   candidates; `|μ|` is uncertainty in the data. A short `|μ|` (divergent
   futures) with a confident π or one dominant slot means the candidate set
   does not cover the spread: a head is stretched over several modes. With
   `expected_distances`, μ also gives a new slot a correct value from its
   first step. Possible later: gate generation on `|μ|` (commit greedily when
   the future is predictable, sample from π when it is open).

