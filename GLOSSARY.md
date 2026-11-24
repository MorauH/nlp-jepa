# nlp-jepa

A text model that predicts several possible futures in latent space and then commits to one. The research question: does committing keep each latent, and every chained prediction, a clear single future instead of a blurry average of all of them?

## Prediction

**Latent**:
The encoder's representation `z_t` of the text up to position `t`.
_Avoid_: hidden, embedding, state

**Horizon**:
The number of tokens `k` between a latent and the future latent the predictor aims at.
_Avoid_: pred_horizon (in prose), block size

**Candidate**:
One of the `P` future latents the predictor proposes for `z_{t+k}` from a single `z_t`.
_Avoid_: prediction, hypothesis, slot (except when talking about the predictor's fixed output position), waypoint

**π (win probability)**:
The predictor's distribution over candidates: how likely each one is to be the closest to the real future.
_Avoid_: scores, score head, selector

**μ (mean future)**:
The predictor's estimate of the average normalized future latent. It is the blurry average that commitment is compared against inside the model; feeding it back gives the **μ path**.
_Avoid_: mean head output, expected future, baseline (the baseline is the P = 1 model)

## Commitment

**Commit**:
To choose one candidate and continue from it as if it were the future.
_Avoid_: select, pick, win

**Committed future**:
The candidate chosen by a commit.
_Avoid_: waypoint, direction, z_future, selected, winner

**Policy**:
The rule a commit uses to choose: hindsight, π-argmax, π-sample or μ-safe.
_Avoid_: strategy, selection mode

**Hindsight**:
The policy that commits to the candidate closest to the real future. It needs the real future, so it is only available during training and measurement.
_Avoid_: oracle, best, winner

**μ-safe**:
The policy that commits to the candidate with the lowest expected distance under μ.
_Avoid_: safe, mean-selected

**Rollout**:
A chain of commits, each starting from the previous committed future, without going back through tokens.
_Avoid_: multi-step generation, unroll, trajectory

## Measurement

**Rollout error**:
The distance between a rollout's committed futures and the real latents at the same depths.
_Avoid_: drift, multi-step loss

**Manifold distance**:
The distance from a latent to its nearest neighbour in a bank of real encoder latents. A clear future is close to some real latent; a blurry average is not.
_Avoid_: realism, sharpness score

**Path validity**:
Synthetic world only: the fraction of rollouts whose steps 1..i all lie on a state that is a successor of the previous step's state.
_Avoid_: accuracy, correctness

**Branch entropy**:
Synthetic world only: the entropy, in bits, of how π-sample commits from one state split between its two successors (commits landing elsewhere are left out). An even split is 1 bit.
_Avoid_: diversity

**Mode coverage**:
Synthetic world only: the worst-case distance from a real successor to its nearest candidate. Small means every mode has a candidate.
_Avoid_: recall

## Decoding

**Hint**:
What the decoder is conditioned on for one block: the latent at the block start and the committed future at the block end.
_Avoid_: waypoint, plan (a plan is a whole decoded rollout), condition

**Decoded plan**:
A rollout (or the μ path) decoded into text block by block from a single encoding of the prompt: block i's hint is (path latent i−1, path latent i), and nothing is re-encoded.
_Avoid_: generation (which re-encodes every block), plan (for a single hint)

**Stage**:
One of the two training phases: the JEPA stage trains encoder and predictor (and the decoder too when it is joint); the decoder stage trains a decoder against a frozen JEPA.
_Avoid_: mode

**Joint JEPA stage**:
The JEPA stage with `decoder_w > 0` (ADR 0002): the decoder trains alongside the encoder and predictor, and its loss reaches the encoder only through the hint's block-start latent. The committed future is detached, so the predictor gets no decoder gradient. `decoder_w = 0` is the **two-stage** recipe.
_Avoid_: end-to-end (the predictor and the committed future are cut off), joint model (the model is the same; only the stage's objective changes)

## Experiments

**Baseline**:
The P = 1 model: the same JEPA trained with a single candidate (`docs/experiments/p1-baseline.json`), i.e. an ordinary single-prediction JEPA. It is the external comparison the claim is judged against.
_Avoid_: baseline for μ (μ is the in-model average), single-head baseline

**Variant**:
A run trained exactly like the main experiment except for the fields in one **preset**, a JSON override file in `docs/experiments/`. The baseline and the ablations are variants.
_Avoid_: override file, experiment config (say preset)
