# JEPA architecture

## The idea

A causal encoder maps tokens to latents. From the latest latent `z_t` the
predictor proposes several **candidate** futures for `z_{t+k}`, where the
**horizon** `k` is random and hidden from it (see [Horizon](#horizon)), a **π (win
probability)** distribution over them, and **μ (mean future)**, the blurry
average of the possible futures. A **commit** chooses exactly one candidate and
continues from it. A **rollout** chains commits without going back through
tokens. The paper asks whether committing keeps each latent, and every chained
latent, a clear single future instead of μ's average of all of them.

## Modules

- **Encoder** (`jepa/models/encoder.py`, `EncoderConfig`): a causal Transformer
  over token ids, returning one latent `z_t` per position. It owns
  `hidden_size`, `vocab_size` and `pad_token_id`; the other configs derive
  their dependent dimensions from it.
- **Horizon** (`jepa/horizon.py`): `draw_horizon` (training, generation, plans)
  and `cycle_horizon` (validation) choose k outside the model.
- **Predictor** (`jepa/models/predictor.py`, `PredictorConfig`): a pure
  function of one latent; it never receives k. It expands `z_t` into `P` slots, mixes them with a
  small non-causal Transformer, and returns candidate futures, π logits and μ.
  Candidates are RMS-normalized to the encoder-latent scale, so the loss
  compares normalized latents.
- **Commitment** (`jepa/models/commitment.py`): `Futures` holds candidates, π
  and μ. The module owns `distance` (the cosine distance `2 - 2 cos`, fp32,
  autocast off, clamped at 0) and `candidate_similarity` (the pairwise cosine
  matrix). A **policy** ranks candidates (hindsight, π-argmax, π-sample,
  μ-safe); `commit` returns one candidate and its index; `rollout` repeats
  that from the committed future. Averaging candidates is never a commit.
  `jepa/models/sampling.py` holds the top-k mask shared by π-sample and the
  token decoder.
- **Decoder** (`jepa/models/decoder.py`, `DecoderConfig`): an autoregressive
  token Transformer. Each position is conditioned on a **hint**: the latent at
  the block start and the committed future at the block end. The hint is added
  to the token embeddings and re-injected before every layer. The token
  embedding is tied to the output head and initialized at std 0.02, so a fresh
  decoder's loss starts near ln(vocab_size).
- **Losses** (`jepa/models/losses.py`): `VISReg` is the anti-collapse
  regularizer on the online encoder latents (scale, shape, center).
  `CommitmentLoss` trains the proposal: a relaxed winner-takes-all move loss,
  π's cross-entropy, μ's regression, and a near-duplicate diversity penalty.
  Both return a `LossOutput`, so the total is the weighted sum of the terms.
- **Stack** (`jepa/models/jepa.py`, `JepaConfig`): `JepaModel` owns encoder,
  predictor, decoder and both losses. `encode` is the single encoder entry
  point (right-padded batches let the causal encoder drop the padding mask);
  `jepa_loss` runs the encoder/predictor objective; `decoder_loss` runs the
  decoder against a frozen JEPA. Both take the batch's `horizon`.
  `horizon_targets` returns the detached latents `horizon` tokens ahead with
  their validity mask; `build_block_hints` aligns latents with decoder
  positions and returns a validity mask for the tail block.

## Horizon

There is one horizon k, and it is everything at once: the prediction target's
offset, the hindsight target, the decoder block length, the end of the hint's
block, the rollout stride and the number of tokens `generate` writes per block.
Training draws one k per batch, uniform over the inclusive
`JepaConfig.horizon_range` (default `[10, 30]`). Generation and decoded plans
draw one k per call or prompt, so a sequence has equal blocks, as in training.
The encoder, predictor and decoder are never told k; the decoder only sees
where its blocks fall, through the hints. The reason is in
[ADR 0003](adr/0003-random-hidden-horizon.md): with a fixed k, a latent whose
period divides k is free to predict.

- **Training** draws k from `(seed, global_step)` with a private generator, in
  both stages. `--resume` replays it without extra state and the global RNG is
  untouched.
- **Validation** cycles k through the range by batch index, so every epoch
  validates on the same k values.
- **Compile.** k is a 0-dim tensor, like `progress`. The predictor runs on every
  position; the target is gathered at t + k (clamped) and positions without a
  real token there are masked. No shape depends on k, so k never recompiles
  (100 steps and 30 validation batches on CUDA: 23 recompiles in the JEPA stage
  and 17 in the decoder stage, for `[10, 30]` as for `[20, 20]`).
- A batch must be longer than `hi`, whatever its k; shorter ones are skipped.
- A fixed horizon is the range `[k, k]` (`docs/experiments/fixed-horizon.json`
  is `[20, 20]`). Checkpoints that store the legacy `pred_horizon` and no
  `horizon_range` load as `[pred_horizon, pred_horizon]`; `pred_horizon` is
  never written or overridable.

A candidate therefore stands for "some `lo` to `hi` tokens ahead": candidate
spread and |μ| mix uncertainty about the content with uncertainty about the
distance. Winner-takes-all could even spend the candidates on "which k" (phase
copies); `measure` reports the slot-horizon mutual information for that.

## Two stages

Two stages share one trainer. The JEPA stage trains the encoder and predictor,
and with `JepaConfig.decoder_w > 0` the decoder too. The decoder stage trains a
decoder against a frozen JEPA. The reasons are in
[ADR 0002](adr/0002-joint-decoder-gradient.md), which supersedes the strictly
two-stage [ADR 0001](adr/0001-two-stage-training.md).

```bash
python -m jepa train                          # stage jepa -> runs/jepa
python -m jepa train --config docs/experiments/joint-decoder.json   # joint
python -m jepa train --config docs/experiments/fixed-horizon.json   # fixed k = 20
python -m jepa train --resume                 # continue it
python -m jepa train --stage decoder --from runs/jepa   # -> runs/jepa-decoder
python -m jepa train --stage decoder --resume --epochs 2
```

`--stage jepa` optimizes the encoder and predictor. With `decoder_w = 0` (the
default until the `decoder_w` sweep of ticket 006 picks a weight) the decoder
is never run. With `decoder_w > 0` the loss gains a `token` term,
`decoder_w × decoder CE` on block hints, and the decoder trains along with
the JEPA. The gradient paths are fixed:

- the hint `z_t` (the latent at the block start) keeps its graph, so the
  decoder loss trains the encoder: z_t is rewarded for what the decoder's
  window lacks and the next block needs;
- the committed future is committed on the JEPA loss's own `Futures` (no
  second predictor pass) and detached, so the predictor is trained only by
  the commitment loss and a hindsight choice, made with the real future, never
  becomes a gradient.

Training uses `DecoderConfig.train_policy` hints; validation uses π-argmax
hints, and validation perplexity comes from the `token` term, not the total.

`--stage decoder` loads a checkpoint from `--from`, freezes its encoder and
predictor (they run under `torch.no_grad()`), attaches a fresh decoder, and
trains only the decoder on committed futures. The decoder stage writes to
`<from>-decoder` (a sibling, `runs/jepa-decoder` by default), never over the
JEPA checkpoint: it rewrites its output dir with the full model, its history and
its trainer state. Pointing `--output-dir` at `--from` without `--resume` is
rejected. Each stage writes `history_<stage>.json` and
`trainer_state_<stage>.pt`. A checkpoint saved from either stage reloads as the
full stack.

### Resuming

Every save (mid-epoch and epoch end) also writes `trainer_state_<stage>.pt`:
optimizer state, LR-scheduler state, `global_step`, the epoch and the number of
batches consumed in it, the epoch's running loss totals, and the RNG states.
`--resume` loads it, continues `global_step` (so warmup/cosine progress and the
commitment temperature continue instead of restarting), restores the optimizer
and scheduler, recreates the interrupted epoch's deterministic bucket order and
skips the already-consumed batches, and appends to the existing history. A
missing trainer state on `--resume` is an error: there is no silent restart.
The schedule's `total_steps` is `len(loader) * --epochs` in both cases, so
resuming with a different `--epochs` re-plans the remaining schedule (the
trainer warns).

### Decoder hints

`decoder_loss` (and `jepa_loss`, for its `token` term) takes an explicit
commit `policy`: `None` uses
`DecoderConfig.train_policy` (the training condition); validation passes
`PiArgmax()` (the inference condition). The model's `train`/`eval` flag is not
involved.

Blocks have the batch's k tokens, so over training the hint reads z_t at many
positions rather than at multiples of one k. The last block of a sequence has
no real latent `horizon` tokens ahead, so it has no committed future;
`build_block_hints` marks it invalid (a block is valid when
`block_start + k <= seq - 1`) and both the
decoder loss and measurement mask those target positions out. No position is
paired with an earlier block's hint.

A raw block-end latent would leak the block's last token, which is a decoder
target. A hindsight commit is still a predictor output, but its choice among
`P` candidates was made with the real future, so it carries up to `log2(P)`
bits about that future (intended teacher forcing). It can also reveal k, on top
of those bits. A π-argmax hint carries neither. Validation and measurement report both, so the hindsight row is a
teacher-forced lower bound on the decoder cross-entropy, not the inference
number. Validation perplexity is `exp` of the token-weighted mean token loss
(weighted by the `tokens` diagnostic), not a mean of per-batch perplexities.

## Configuration

`jepa/config.py` holds the main experiment: model sizes, window, candidate
count, loss weights and annealing. Variants (baselines, ablations) are
override files in `docs/experiments/`, passed as `train --config PATH`. Each one is a JSON
object nested like `config.json` that holds only the fields that change, and
`apply_overrides` rejects unknown keys before anything trains. A fresh JEPA run
applies them to the defaults, and the decoder stage accepts only `decoder.*`
keys. With `--resume`, `--config` is an error because the checkpoint's config is the run's. `--seed N`
(default `SEED`) seeds model construction and the data order. It is saved in
`history_<stage>.json` and the trainer state, so `--resume` keeps it. On CPU
a seed reproduces the weights exactly. On CUDA, non-deterministic kernels
(Flash Attention backward) make same-seed runs differ slightly. Checkpoint
`config.json` nests the component configs:

```json
{
  "model_type": "jepa",
  "horizon_range": [10, 30],
  "decoder_w": 0.0,
  "encoder":    {"model_type": "jepa_encoder",    "...": "..."},
  "predictor":  {"model_type": "jepa_predictor",  "...": "..."},
  "decoder":    {"model_type": "jepa_decoder",    "...": "..."},
  "visreg":     {"model_type": "jepa_visreg",     "...": "..."},
  "commitment": {"model_type": "jepa_commitment", "...": "..."}
}
```

`JepaConfig` fills in or checks the dimensions shared with the encoder and
raises `ValueError` on a conflict. `DecoderConfig.loss_chunk_size` bounds the
logits tensor in the token loss.

Engineering choices that are not the experiment live as module-level constants
in `jepa/training/trainer.py`, the single source: bf16 autocast and
`torch.compile` (both CUDA-only), fused AdamW, TF32, the warmup+cosine
schedule (`WARMUP_FRACTION`, `LR_FLOOR_FRACTION`), gradient clipping, weight
decay, the default seed, batch padding, `LOG_EVERY` and `SAVE_EVERY_STEPS`.
`TrainConfig` carries only per-run values (stage, output dir, batch size,
epochs, learning rate, device, resume, workers, save interval, seed); the CLI sets
per-stage defaults and nothing else. Checkpoints are saved from the uncompiled
module.

## Data preparation

Train the 32k tokenizer once:

```bash
python -m jepa train-tokenizer
```

Then prepare roughly 500M tokens of 1024-token chunks:

```bash
python -m jepa prepare-data
```

The tokenizer step also accepts `--save-text data/fineweb_edu_500m_tokens.txt
--megabytes 2300` to keep the sampled text so preparation can reuse it:

```bash
python -m jepa prepare-data --text-file data/fineweb_edu_500m_tokens.txt
```

Preparation tokenizes with `data/tokenizer_32k`, splits documents into
non-overlapping chunks, drops chunks shorter than 64 tokens, and holds out
whole documents for validation. Keep `--max-length` (default 1024) equal to
`EncoderConfig.max_position_embeddings`. Batches are right-padded and the
encoder is causal, so no real query attends a pad key and the attention mask
can be dropped for fused SDPA; changing the padding side breaks that
invariant. `data/` and `runs/` are generated artifacts and are gitignored.

## Measurement

```bash
python -m jepa measure --model runs/jepa
python -m jepa measure --model runs/jepa --json runs/jepa/metrics.json
```

`measure` evaluates validation chunks against a bank of real train-split
latents.

Everything that compares a prediction with the future is scored at fixed
**evaluation horizons** and averaged over them: for `[10, 30]` they are 11, 14,
17, 20, 23, 26 and 29, whose residues modulo 10 all differ (steps of 5 would
always sit on the observed 10-token cycle). A fixed range `[k, k]` gives just
k. Latents, candidates and every rollout path except hindsight do not depend on
k, so each section computes them once and scores them at each; only the loss
section calls `jepa_loss` per k.

The JSON has one dict per section:

- `context`: checkpoint, device, example/token counts, `horizon_range`, the
  evaluation `horizons`, candidates.
- `representation`: per-dim std, mean norm, effective rank, temporal cosine
  (mean over the evaluation horizons), and VISReg terms — the no-collapse
  checks. Also the **lag profile** (`lag_cos`, mean cos(z_t, z_t+lag) for lags
  1..`hi` and 40, 100, 200, 400), the **max rebound** (over every lag L in
  2..`hi`: cos at L minus the lowest cos at a shorter lag, maximized; a run with
  a larger `hi` scans more lags, which can only raise it), the
  **fast share** (within-document energy left after a centred 10-step moving
  average) and the **document share** (between-document share of the variance,
  past the first 20 positions; each query chunk counts as a document). The
  rebound exists because the two-stage JEPA learned a latent with a period
  dividing k = 20: cos at 20 alone looked smooth (0.96) while cos at 5 was
  −0.15. Scanning every lag also catches a period that does not divide any one
  k. Verdict `no_periodic_shortcut`: max rebound < 0.05. The fast share has no
  verdict, since token-to-token change is legitimately fast. The document
  share has none either; a per-document constant is free to predict at any k.
- `loss`: every JEPA term (including `token` for a joint checkpoint) with its
  weight and weighted contribution, plus the
  π floor (`H(r)` and `KL(r||π)`).
- `heads`: candidate separation, hindsight shares and live slots, π's NLL
  gain over a constant distribution and its accuracy, μ's norm and distance
  calibration. **Slot-horizon mutual information** (`slot_horizon_mi`, with its
  bound `slot_horizon_mi_max`), between the hindsight slot and k over the
  positions every evaluation horizon can use: near its maximum the slots are
  phase copies, one per k. Near 0 no slot is tied to a distance from z_t, but a
  slot tied to an absolute phase wins by t + k and averaging over t hides it,
  so a low value does not exclude those. n/a for a fixed k or P = 1, and no
  verdict yet. Live slots alone prove nothing: phase copies would make all of
  them live. `steady_state` repeats slot usage and MI for positions 80 and up.
- `policies`: depth-1 distance to the real future for copy, hindsight,
  π-argmax, π-sample, μ-safe, μ and the lowest-distance fixed slot, as the mean
  over the evaluation horizons, with each k in `by_horizon`. `steady_state`
  has the same rows for positions 80 and up, because the document start
  dominates the all-position averages. The
  paper's main table is this one plus the rollout rows: hindsight gain over μ
  is what candidates buy; commit headroom is what a better commit rule could
  add.
- `rollouts`: **rollout error** and **manifold distance** per depth for
  hindsight, π-argmax, π-sample and μ, with the `real` row as calibration.
  Depth i is scored at t + i·k, with the same starts for every k (a start needs
  `depths × hi` tokens ahead). The
  claim is the depth ≥ 2 manifold row: committed rollouts stay closer to real
  latents than μ. `best_of_k[K][depth]` (minADE_K) exists because rollout
  error against the one realized continuation penalizes commitment by
  construction: μ minimizes the expected distance to whichever branch happens.
  Keeping the best of K π-sample rollouts, each scored as a whole rollout by
  its mean error over depths 1..i, asks instead whether the committed futures
  *include* the real one, so it should fall below μ as K grows (verdict:
  best-of-16 < μ at the deepest depth, with μ's error also averaged over
  depths 1..i).
- `decoder`: token-weighted cross-entropy over blocks of k tokens for each
  evaluation horizon, per hint condition (committed
  futures, real latent, shuffled, null, no_hints), hint use and specificity.
  `null` keeps the real z_t and zeroes the future; `no_hints` zeroes both.
  `z_t_use` (`no_hints − null`) and `future_use` (`null − pi_argmax`) split the
  two hint channels: with a jointly trained z_t, only `future_use` shows what
  the commitment adds. All rows use
  the same mask (the tail block with no committed future is excluded). The
  π-argmax row is the inference hint; the hindsight row is teacher-forced and
  its hint carries up to `log2(P)` bits about the target future (and may reveal
  k), so read it as
  a lower bound rather than a leak-free condition.
- `verdicts`: PASS/FAIL checks, each naming its metric and threshold.

## Generation

`jepa/inference/generate.py` loads a checkpoint, draws one `k` from its
`horizon_range`, re-encodes the context each block, commits a future sampled
from π, and generates `k` tokens with the decoder. Every block of one call has
the same k; `--seed` fixes it:

```bash
python -m jepa generate --prompt "When in Rome, " --steps 3 --temperature 0.8
```

`--commit-temperature` / `--commit-top-k` control the commit from π;
`--temperature` / `--top-k` control token sampling.

### Decoded plans

`--plan-depth N` decodes **latent plans** instead: the qualitative figure
that sets clear committed plans beside the blurry μ path.

```bash
python -m jepa generate --plan-depth 4 --baseline-model runs/p1 \
    --json runs/jepa/plans.json --html runs/jepa/plans.html
```

For each prompt the encoder runs **once**, and `z_t` is the prompt's last
latent. Each condition is a latent path of `N` latents:

- `pi_argmax`: `rollout(predictor, z_t, PiArgmax(), N)`;
- `pi_sample`: `--samples` rollouts with π-sample (seeded by `--seed`);
- `mu`: measurement's μ path (`mu_path` in `jepa/evaluation/measure.py`);
- `baseline`: with `--baseline-model`, that checkpoint's π-argmax rollout
  decoded by its own decoder (meant for the P = 1 baseline).

Block `i` is decoded with `decoder.generate` and the hint (start = path
latent `i − 1`, or `z_t` for `i = 1`; committed = path latent `i`). The token
context is the prompt plus every token decoded so far. Nothing is re-encoded,
so the text is a decoded plan rather than generation. Each prompt draws one k
from `(--seed, prompt index)`, shared by every condition, sample and the
baseline model so the columns line up (the baseline must have the same
`horizon_range`). Each record holds exactly `N × k` tokens, and an EOS does not
stop a plan.

Prompts are the first `--num-prompts` (default 8) validation chunks with at
least `--prompt-length + N·hi` tokens, taken in a `--seed`-ed random order, so
the choice does not depend on the drawn k. The `--prompt-length` prefix is the
prompt and the next `N·k` tokens are the reference continuation. `--prompt`
adds free prompts (repeatable) that have no reference. The JSON has one record
per prompt, condition and sample (`prompt_index`, `source`, `prompt`,
`horizon`, `reference` per block, `condition`, `sample`, `token_ids`,
`blocks`). `--html` writes one side-by-side table per prompt, with its k.

Caveats for the report:

- **A committed latent has no known distance.** Blocks decode to a drawn k, but
  the latent stands for "some k tokens ahead", so the text may not line up with
  the reference block by block.
- **The decoder never saw μ hints.** It is trained on committed candidates, so
  decoding the μ path is out of distribution. That is part of the point (a
  blurry latent is not a valid plan), but it confounds the comparison. The
  fair comparison is the `baseline` row: the P = 1 model's own decoder,
  trained on its own single (blurry) prediction.
- **Token sampling adds variance.** Plans decode greedily by default
  (`--temperature` defaults to 0 in this mode), so the differences come from
  the plan. Show sampled decodes separately.
- **Pick samples honestly.** Prompts are seeded and fixed. Don't cherry-pick,
  and if you show a subset, say how it was chosen.
