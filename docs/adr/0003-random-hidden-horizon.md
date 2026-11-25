# One random horizon per batch, used everywhere and hidden from the predictor

Proposed. Ticket 007 implements it. Builds on ADR 0002, whose gradient paths stay in force.

ADR 0002 let the decoder gradient reach the encoder to remove a shortcut: a latent whose period divides the horizon k = 20 is free to predict. The first joint run (`runs/joint-seed1`, `decoder_w = 0.3`) weakened the shortcut without removing it. The latent still repeats every 10 tokens (cosine 0.96 at lag 10, 0.65 at lag 5), and the periodic rebound is 0.31 against a limit of 0.05 (1.02 for the two-stage run). From position 80 on, which is 88% of positions, copying z_t is as close to z_{t+20} as the best of the eight candidates chosen in hindsight (0.032 for both), and μ is worse (0.038). The two live candidates are the same vector. At this weight, the decoder gradient does not beat a shortcut that the prediction loss gives away for free.

So the horizon k is now drawn at random, once per batch, uniformly from a range (10 to 30). The predictor stays a function of one latent and is not told k. Ticket 006 dismissed a variable k because a periodic feature is predictable from (z_t, k). Without k, z_t does not determine the phase of z_{t+k}. On the existing encoders, replacing the fixed k by a random one raises the distance of the best single prediction we tried from 0.032 to 0.265 (joint) and from 0.021 to 0.987 (two-stage).

There is only one horizon. The sampled k is the prediction target's offset, the hindsight target, the decoder block length, the end of the hint's block, the rollout stride and the length of a generated block. Keeping some of these at 20 would leave two meanings of "the future" in one model: the predictor would be trained on one distance and the decoder, rollouts and generation would assume another. It would also leave the decoder's hint reading z_t at multiples of 20 only, always at the same phase of the cycle. Inference draws one k per generation, plan or rollout, so a sequence has equal blocks, as in training. The decoder is not told k.

The random range is the default (`horizon_range = [10, 30]`, replacing `pred_horizon`). A fixed k is the range `[20, 20]`, kept as the ablation. This differs from `decoder_w`, whose default stays 0 until a sweep picks a weight: there the old behaviour is a recipe we may still want, here it is the cause of the shortcut. Old checkpoints store `pred_horizon` and no range, so that field is still read as a fixed range and they load as what they are.

Alternatives considered:

- **Sample k for the commitment loss only**, keeping blocks, hints and rollouts at 20. Simpler to compare with the old runs, but it has the two-horizons problem above.
- **Tell the predictor k.** It can then undo the phase, which is the case ticket 006 rejected.
- **A fixed k that is not a multiple of 10.** The encoder can pick a period that divides the new k (not tested).
- **Only raise `decoder_w`.** A sweep is still worth running and is not excluded by this decision. It competes with the prediction loss instead of taking the reward for the cycle away, and a large weight risks a plain language-model encoder (ADR 0002).
- **Penalize periodicity directly.** It would target the symptom we measure, and the encoder could move to a form the penalty does not cover.
- **One k per position or per document.** One per batch is simpler, keeps a single lag and block length in a batch, and is enough in expectation.

Accepted costs and open risks:

- **Winner-takes-all can hide the cycle.** Eight candidates covering the phases bring the hindsight distance under random k back to 0.043, against 0.032 today. Candidates would then mean "which k" rather than "which future". π cannot correct this, because its assignment target is detached. P = 1 has no such escape, so the main run and its baseline could end up with different encoders. Ticket 007 measures the mutual information between the winning slot and k and gates the full run on it. If the candidates do absorb the cycle, the next step is to score each candidate against two horizons at once. That is a separate decision.
- **A per-document constant is still free.** It does not depend on k. Document identity already explains 58% of the joint latent's variance (23% in the two-stage run), and only the decoder gradient works against it.
- **A candidate has no known distance.** It is one possible future some 10 to 30 tokens ahead. Candidate spread and |μ| mix uncertainty about the content with uncertainty about the distance, and blocks and rollouts use a drawn k the predictor did not aim at. A hindsight hint can reveal k to the decoder, beyond the `log2(P)` bits of ADR 0002; a π-argmax hint cannot. The report has to say all of this.
- **Losses are not comparable** with fixed-k runs. Compare the `measure` sections, which average over fixed evaluation horizons.
- **The recipe changed after seeing results.** The report fixes its hypotheses before the experiments. They keep their wording, and the report states that the horizon protocol changed after two single-seed runs and before the grid.
