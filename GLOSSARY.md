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
The predictor's estimate of the average normalized future latent. It is the blurry single-prediction baseline that commitment is compared against.
_Avoid_: mean head output, expected future

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

## Decoding

**Hint**:
What the decoder is conditioned on for one block: the latent at the block start and the committed future at the block end.
_Avoid_: waypoint, plan, condition

**Stage**:
One of the two training phases: the JEPA stage trains encoder and predictor; the decoder stage trains a decoder against a frozen JEPA.
_Avoid_: mode, joint training
