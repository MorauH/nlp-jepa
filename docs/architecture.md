# JEPA architecture

## The idea

A causal encoder maps tokens to latents. From the latest latent `z_t` the
predictor proposes several **candidate** futures for `z_{t+k}`, a **π (win
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
- **Predictor** (`jepa/models/predictor.py`, `PredictorConfig`): a pure
  function of one latent. It expands `z_t` into `P` slots, mixes them with a
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
  decoder against a frozen JEPA; `build_block_hints` aligns latents with
  decoder positions and returns a validity mask for the tail block.

## Two stages

The JEPA is trained first, then a decoder is trained against the frozen JEPA.
The reasons are in [ADR 0001](adr/0001-two-stage-training.md).

```bash
python -m jepa train                          # stage jepa -> runs/jepa
python -m jepa train --resume                 # continue it
python -m jepa train --stage decoder --from runs/jepa   # -> runs/jepa-decoder
python -m jepa train --stage decoder --resume --epochs 2
```

`--stage jepa` optimizes the encoder and predictor; the decoder is never run.
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

`decoder_loss` takes an explicit commit `policy`: `None` uses
`DecoderConfig.train_policy` (the training condition); validation passes
`PiArgmax()` (the inference condition). The model's `train`/`eval` flag is not
involved.

The last block of a sequence has no real latent `horizon` tokens ahead, so it
has no committed future; `build_block_hints` marks it invalid and both the
decoder loss and measurement mask those target positions out. No position is
paired with an earlier block's hint.

A raw block-end latent would leak the block's last token, which is a decoder
target. A hindsight commit is still a predictor output, but its choice among
`P` candidates was made with the real future, so it carries up to `log2(P)`
bits about that future (intended teacher forcing). A π-argmax hint carries
none. Validation and measurement report both, so the hindsight row is a
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
  "pred_horizon": 20,
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
latents. The JSON has one dict per section:

- `context`: checkpoint, device, example/token counts, horizon, candidates.
- `representation`: per-dim std, mean norm, effective rank, temporal cosine,
  and VISReg terms — the no-collapse checks.
- `loss`: every JEPA term with its weight and weighted contribution, plus the
  π floor (`H(r)` and `KL(r||π)`).
- `heads`: candidate separation, hindsight shares and live slots, π's NLL
  gain over a constant distribution and its accuracy, μ's norm and distance
  calibration.
- `policies`: depth-1 distance to the real future for copy, hindsight,
  π-argmax, π-sample, μ-safe, μ and the lowest-distance fixed slot. The
  paper's main table is this one plus the rollout rows: hindsight gain over μ
  is what candidates buy; commit headroom is what a better commit rule could
  add.
- `rollouts`: **rollout error** and **manifold distance** per depth for
  hindsight, π-argmax, π-sample and μ, with the `real` row as calibration. The
  claim is the depth ≥ 2 manifold row: committed rollouts stay closer to real
  latents than μ. `best_of_k[K][depth]` (minADE_K) exists because rollout
  error against the one realized continuation penalizes commitment by
  construction: μ minimizes the expected distance to whichever branch happens.
  Keeping the best of K π-sample rollouts, each scored as a whole rollout by
  its mean error over depths 1..i, asks instead whether the committed futures
  *include* the real one, so it should fall below μ as K grows (verdict:
  best-of-16 < μ at the deepest depth, with μ's error also averaged over
  depths 1..i).
- `decoder`: token-weighted cross-entropy per hint condition (committed
  futures, real latent, shuffled, null), hint use and specificity. All rows use
  the same mask (the tail block with no committed future is excluded). The
  π-argmax row is the inference hint; the hindsight row is teacher-forced and
  its hint carries up to `log2(P)` bits about the target future, so read it as
  a lower bound rather than a leak-free condition.
- `verdicts`: PASS/FAIL checks, each naming its metric and threshold.

## Generation

`jepa/inference/generate.py` loads a checkpoint, re-encodes the context each
block, commits a future sampled from π, and generates `k` tokens with the
decoder:

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
so the text is a decoded plan rather than generation. Each record holds
exactly `N × k` tokens, and an EOS does not stop a plan.

Prompts are the first `--num-prompts` (default 8) validation chunks with at
least `--prompt-length + N·k` tokens, taken in a `--seed`-ed random order. The
`--prompt-length` prefix is the prompt and the next `N·k` tokens are the
reference continuation. `--prompt` adds free prompts (repeatable) that have no
reference. The JSON has one record per prompt, condition and sample
(`prompt_index`, `source`, `prompt`, `reference` per block, `condition`,
`sample`, `token_ids`, `blocks`). `--html` writes one side-by-side table per
prompt.

Caveats for the report:

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
