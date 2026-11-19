# JEPA architecture

All configuration lives in [`jepa/config.py`](../jepa/config.py). Each trainable piece has
its own self-contained HuggingFace config, so any piece can be built, saved and
reloaded alone. `JepaConfig` composes them for the full stack.

| Config            | Model            | Input                         | Output                        |
| ----------------- | ---------------- | ----------------------------- | ----------------------------- |
| `EncoderConfig`   | `EncodeTransformer` | token ids                 | latent sequence `(B, T, H)`   |
| `PredictorConfig` | `PredictorModel`    | `z_t`                     | `P` candidate `z_{t+k}` + scores |
| `DecoderConfig`   | `DecoderModel`      | token context, latent waypoints | next-token logits        |
| `VISRegConfig`    | `VISRegLoss`        | candidates, scores, targets, encoder latents | scalar JEPA loss |
| `JepaConfig`      | `JepaModel` / `JepaDecodeModel` | composed stack | loss / tokens     |

## Dimension ownership

The encoder owns the model width and vocabulary. `JepaConfig` fills in or
checks the dependent fields:

```
predictor.target_dim   == encoder.hidden_size
predictor.pad_token_id == encoder.pad_token_id
decoder.hidden_size    == encoder.hidden_size
decoder.vocab_size     == encoder.vocab_size
decoder.pad_token_id   == encoder.pad_token_id
```

If you pass an explicit conflicting value, construction fails with
`ValueError` instead of producing a stack whose pieces do not fit.

## Run a component on its own

```python
from jepa import EncoderConfig, EncodeTransformer
from jepa import PredictorConfig, PredictorModel
from jepa import DecoderConfig, DecoderModel

# Encoder
enc_cfg = EncoderConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,
    num_attention_heads=8,
    num_hidden_layers=4,
)
encoder = EncodeTransformer(enc_cfg)
encoder.save_pretrained("checkpoints/encoder")

# ... later / elsewhere
encoder = EncodeTransformer.from_pretrained("checkpoints/encoder")
latents = encoder(tokens, attention_mask=mask)

# Predictor: derive the width/vocab fields from the encoder, override the rest
pred_cfg = PredictorConfig.from_encoder(enc_cfg, num_hidden_layers=2)
predictor = PredictorModel(pred_cfg)
predictor.save_pretrained("checkpoints/predictor")

# Decoder: causal token transformer with a 96-token sliding window
dec_cfg = DecoderConfig.from_encoder(
    enc_cfg, num_hidden_layers=2, sliding_window=96
)
decoder = DecoderModel(dec_cfg)
decoder.save_pretrained("checkpoints/decoder")
```

Each config is a normal `PretrainedConfig`, so `save_pretrained` /
`from_pretrained` work independently.

## Full stack

```python
from jepa import JepaConfig, JepaDecodeModel, VISRegConfig

# Decoder is created from the encoder config by default.
config = JepaConfig.from_encoder(
    enc_cfg,
    pred_horizon=5,
    visreg=VISRegConfig(num_slices=256, move_w=1.0),
)
model = JepaDecodeModel(config)

# One JSON file describes the whole stack, including nested configs.
model.save_pretrained("runs/jepa")
model = JepaDecodeModel.from_pretrained("runs/jepa")
```

To train the self-supervised JEPA objective without a decoder:

```python
from jepa import JepaConfig, JepaModel

config = JepaConfig.from_encoder(enc_cfg, with_decoder=False)
model = JepaModel(config)
```

Or inject an already-trained encoder into the JEPA stack:

```python
encoder = EncodeTransformer.from_pretrained("checkpoints/encoder")
model = JepaModel(JepaConfig.from_encoder(encoder.config, with_decoder=False), encoder=encoder)
```

## Data

`jepa/training/train.py` trains on a prepared FineWeb-Edu subset. First train
the 32k tokenizer (once):

```bash
python -m jepa train-tokenizer
```

This streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`), trains a 32k
GPT-2-style byte-level BPE with the Rust `tokenizers` library on 500 MB of
English text (~1.5 min), and writes `data/tokenizer_32k/`. It reserves
`<pad>` (id 0), `<bos>` and `<eos>`. 500 MB is enough to learn the vocabulary;
`--save-text` can additionally store the sampled documents so the next step
does not stream FineWeb-Edu again. A text sample large enough for a 500M-token
dataset is about 2.3 GB:

```bash
python -m jepa train-tokenizer \
    --save-text data/fineweb_edu_500m_tokens.txt \
    --megabytes 2300
```

To extend an existing sample later without retraining:

```bash
python -m jepa train-tokenizer --text-only --save-text data/fineweb_edu_500m_tokens.txt
```

Then prepare roughly 500M tokens. The default streams FineWeb-Edu directly:

```bash
python -m jepa prepare-data
```

or reuse a saved text sample:

```bash
python -m jepa prepare-data --text-file data/fineweb_edu_500m_tokens.txt
```

This tokenizes with `data/tokenizer_32k` (4.6 chars/token, so 500M tokens is
~2.3 GB of text), splits long documents into non-overlapping 1024-token
chunks, drops chunks shorter than 64 tokens, and writes train/validation
splits to `data/fineweb_edu_500m_32k` (ignored by git). No BOS/EOS tokens are
added. Whole documents are held out for validation, so validation does not
contain continuations of training chunks. The target token count and chunk
length are configurable:

```bash
python -m jepa prepare-data --target-tokens 500_000_000 --max-length 1024
```

Keep `--max-length` equal to `MODEL_MAX_LENGTH` in `jepa/training/train.py`.

`load_tokenizer` (`jepa/data/tokenizer.py`) loads the HuggingFace tokenizer
directory produced by `jepa.data.train_tokenizer`.

`jepa/training/train.py` trains on the prepared chunks in
`data/fineweb_edu_500m_32k`; pass `--fineweb-dir PATH` to use another
directory. The collate function accepts pre-tokenized `input_ids` examples,
so the FineWeb chunks are not tokenized again on every epoch.

FineWeb chunks are long. The default is `--batch-size 32`; lower it if memory
runs out on a 24 GB L4 or 48 GB L40S.

## Training

`jepa/training/trainer.py` contains `JepaTrainer`, a minimal training loop. It computes
both objectives in one backward pass:

```
loss = jepa_loss_weight * jepa["loss"] + decoder_loss_weight * decoder_out["loss"]
```

The decoder is fed detached hints and trained as an autoregressive token
transformer with a sliding window. Its loss only updates the decoder; the JEPA
loss updates the encoder and predictor. The decoder sees the last
`DECODER_SLIDING_WINDOW` tokens and a latent hint for the current `k`-token
block: the real latent at the block start and the predictor's selected
waypoint at the block end. At inference the waypoint is sampled from the
predictor's scores.

The learning rate warms up linearly over the first 1% of optimizer steps, then
decays with a cosine schedule to 10% of the base LR. The current LR is shown in
the progress bar.

The training command is `python -m jepa train` (or `scripts/slurm_train.sh` on
SLURM):

```bash
python -m jepa train
```

It writes the full checkpoint to `runs/jepa/`, plus standalone component
checkpoints in `runs/jepa/encoder`, `runs/jepa/predictor`, and `runs/jepa/decoder`.

### Performance knobs

`jepa/training/train.py` exposes the main throughput switches:

| Setting | Default | Effect |
| --- | --- | --- |
| `USE_AMP` | `True` | Runs the model forward under `torch.autocast`. On an L4/Ada GPU, `AMP_DTYPE="auto"` selects **bf16** (no gradient scaler needed). Falls back to fp16 with a scaler, and is disabled on CPU/MPS. |
| `COMPILE_MODEL` | `True` | Wraps the model in `torch.compile(..., dynamic=True)`. Compilation is amortized over a full run. If the backend fails, `suppress_errors` falls back to eager instead of aborting. |
| `COMPILE_MODE` | `"default"` | Passed to `torch.compile`; `"max-autotune"` can be faster but has a much longer warm-up. |
| `DECODER_HIDDEN_LAYERS` | `2` | Depth of the autoregressive token decoder. |
| `DECODER_SLIDING_WINDOW` | `96` | Number of past tokens visible to each decoder position. Lower it for cheaper generation; the latent hints carry longer-range semantics. |
| `DECODER_LOSS_CHUNK_SIZE` | `256` | Computes the token loss in position chunks so the full `(B, S, vocab)` logits tensor is never materialized. Mathematically identical; set to `None` to disable. |
| `PAD_TO_MULTIPLE_OF` | `64` | Right-pads each batch to a tensor-core-friendly length and limits the number of compiled shapes. |
| `LENGTH_BUCKET_WINDOW` | `100` | Training batches are built by sorting within windows of 100 batches (`jepa/training/data.py`), so each batch holds similarly long chunks. Cuts padding from ~1.5x to ~1.04x on the FineWeb chunks. |
| `LOG_EVERY` | `10` | Accumulates losses on the GPU and synchronizes only every N steps instead of calling `.item()` every step. |
| `RIGHT_PADDED` | `True` | The tokenizer pads on the right. A causal encoder then does not need the padding mask in attention, enabling the fused causal SDPA (Flash Attention) kernel. Set `False` for a left-padding tokenizer. |

On WSL2/NixOS, run the trainer from the dev shell:

```bash
nix develop --command python -m jepa train
```

The `flake.nix` dev shell adds `openssl` to `PATH` (Inductor's C++ codegen
needs it) and sets `TRITON_LIBCUDA_PATH` (Triton cannot discover `libcuda`
through `/sbin/ldconfig` on NixOS). Without these, `torch.compile` falls back
to eager with a warning. On a normal Linux cluster node neither workaround is
needed.

Under the hood the trainer also:

- builds the attention bias once per forward and shares it across all layers
  (previously it was rebuilt, including a full `(T, T)` causal tensor, in every
  layer);
- uses fused AdamW on CUDA;
- enables TF32 for the remaining fp32 matmuls (the VISReg loss);
- keeps VISReg's statistics and sorting in fp32 even under bf16 autocast;
- saves checkpoints from the original module, so `torch.compile` does not add
  `_orig_mod.` prefixes to the state dict.

The decoder is now a small autoregressive transformer over token positions, so
the `lm_head` projection and cross-entropy over `(B, S, vocab)` dominate.
`DECODER_LOSS_CHUNK_SIZE` reduces the memory cost; lowering
`DECODER_SLIDING_WINDOW` or the decoder depth reduces compute.

## Generation

`jepa/inference/generate.py` loads a trained checkpoint and asks for a prompt and a
number of prediction steps. Each step re-encodes the current context, samples a
latent waypoint from the predictor, and generates `k` tokens autoregressively:

```bash
python -m jepa generate
# or non-interactively
python -m jepa generate --prompt "When in Rome, " --steps 3 --temperature 0.8
```

The decoder sees the last `DECODER_SLIDING_WINDOW` tokens and the latent
waypoint. Grammar comes from the token context; the waypoint supplies the
semantic direction. Generated tokens are fed back into the encoder between
blocks, and the context is capped at the encoder's `max_position_embeddings`.
Use `--direction-temperature` / `--direction-top-k` to control waypoint
sampling and `--temperature` / `--top-k` to control token sampling.

## Saved layout

`JepaDecodeModel.save_pretrained("runs/jepa")` produces one `config.json`:

```json
{
  "model_type": "jepa",
  "pred_horizon": 5,
  "encoder":   { "model_type": "jepa_encoder",   "...": "..." },
  "predictor": { "model_type": "jepa_predictor", "...": "..." },
  "decoder":   { "model_type": "jepa_decoder",   "...": "..." },
  "visreg":    { "model_type": "jepa_visreg",    "...": "..." }
}
```

No hyperparameter is duplicated at the top level: the composite config only
stores the links (`pred_horizon`) and the nested component configs.
