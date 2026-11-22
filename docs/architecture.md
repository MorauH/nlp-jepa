# JEPA architecture

## The idea

A causal encoder maps tokens to latents. From the latest latent `z_t` the
predictor proposes several **candidate** futures for `z_{t+k}`, a **π (win
probability)** distribution over them, and **μ (mean future)**, the blurry
single-prediction baseline. A **commit** chooses exactly one candidate and
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
  compares directions.
- **Commitment** (`jepa/models/commitment.py`): `Futures` holds candidates, π
  and μ, and owns the candidate-target distance `2 - 2 cos`. A **policy**
  ranks candidates (hindsight, π-argmax, π-sample, μ-safe); `commit` returns
  one candidate and its index; `rollout` repeats that from the committed
  future. Averaging candidates is never a commit.
- **Decoder** (`jepa/models/decoder.py`, `DecoderConfig`): an autoregressive
  token Transformer. Each position is conditioned on a **hint**: the latent at
  the block start and the committed future at the block end. The hint is added
  to the token embeddings and re-injected before every layer.
- **Losses** (`jepa/models/losses.py`): `VISReg` is the anti-collapse
  regularizer on the online encoder latents (scale, shape, center).
  `CommitmentLoss` trains the proposal: a relaxed winner-takes-all move loss,
  π's cross-entropy, μ's regression, and a near-duplicate diversity penalty.
  Both return a `LossOutput`, so the total is the weighted sum of the terms.
- **Stack** (`jepa/models/jepa.py`, `JepaConfig`): `JepaModel` owns encoder,
  predictor, decoder and both losses. `jepa_loss` runs the encoder/predictor
  objective; `decoder_loss` runs the decoder against a frozen JEPA;
  `build_block_hints` aligns latents with decoder positions.

## Two stages

The JEPA is trained first, then a decoder is trained against the frozen JEPA.
The reasons are in [ADR 0001](adr/0001-two-stage-training.md).

```bash
python -m jepa train                          # stage jepa
python -m jepa train --resume                 # continue it
python -m jepa train --stage decoder --from runs/jepa
python -m jepa train --stage decoder --resume --epochs 2
```

`--stage jepa` optimizes the encoder and predictor; the decoder is never run.
`--stage decoder` loads a checkpoint from `--from`, freezes its encoder and
predictor (they run under `torch.no_grad()`), attaches a fresh decoder, and
trains only the decoder on committed futures. By default both stages read and
write `runs/jepa`; each stage writes `history_<stage>.json`. A checkpoint
saved from either stage reloads as the full stack.

## Configuration

`jepa/config.py` holds the experiment: model sizes, window, candidate count,
loss weights and annealing. Those defaults are the run; the training scripts
do not override them. Checkpoint `config.json` nests the component configs:

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
in `jepa/training/train.py` and `jepa/training/trainer.py`: bf16 autocast on
CUDA, fused AdamW, TF32, `torch.compile` of the stage loss, the warmup+cosine
schedule, batch padding, `LOG_EVERY`, `SAVE_EVERY_STEPS`, and the stage batch
size / epoch / learning-rate defaults. Checkpoints are saved from the
uncompiled module.

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
- `heads`: candidate separations, win shares and live slots, π's NLL gain over
  a constant distribution and its accuracy, μ's norm and calibration.
- `policies`: depth-1 distance to the real future for copy, hindsight,
  π-argmax, π-sample, μ-safe, μ-direction and the best fixed slot. The paper's
  main table is this one plus the rollout rows: oracle gain over μ-direction
  is what candidates buy; selection headroom is what a better selector could
  add.
- `rollouts`: **rollout error** and **manifold distance** per depth for
  hindsight, π-argmax, π-sample and μ, with the `real` row as calibration. The
  claim is the depth ≥ 2 manifold row: committed rollouts stay closer to real
  latents than μ.
- `decoder`: token-weighted cross-entropy per hint condition (committed
  futures, real latent, shuffled, null), hint use and specificity.
- `verdicts`: PASS/FAIL checks, each naming its metric and threshold.

## Generation

`jepa/inference/generate.py` loads a checkpoint, re-encodes the context each
block, commits a future sampled from π, and generates `k` tokens with the
decoder:

```bash
python -m jepa generate --prompt "When in Rome, " --steps 3 --temperature 0.8
```

`--direction-temperature` / `--direction-top-k` control the commit from π;
`--temperature` / `--top-k` control token sampling.
