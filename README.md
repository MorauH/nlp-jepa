# JEPA language model

A joint-embedding predictive architecture (JEPA) for text. A causal encoder
maps token sequences to latents, a predictor proposes future latents, and an
autoregressive decoder turns latent waypoints into tokens. Training combines
the VISReg JEPA objective with a token-level decoder loss.

## Layout

```
data/                         generated datasets and tokenizer
docs/
  architecture.md             model, data, training, and generation details
  verification.md             collapse / latent-usage test notes
jepa/
  __main__.py                 `python -m jepa <command>`
  cli.py                      command dispatcher
  config.py                   component configs
  paths.py                    project paths
  runtime.py                  device and AMP helpers
  models/
    blocks.py                 shared Transformer blocks
    encoder.py                causal token encoder
    predictor.py              latent predictor
    decoder.py                autoregressive token decoder
    losses.py                 VISReg objective
    jepa.py                   JepaModel (encoder + predictor)
    stack.py                  JepaDecodeModel (full stack)
  data/
    tokenizer.py              tokenizer loading
    prepare.py                FineWeb-Edu -> token chunks
    train_tokenizer.py        byte-level BPE
  training/
    data.py                   collation and length bucketing
    optim.py                  parameter groups
    trainer.py                JepaTrainer
    train.py                  end-to-end training
    train_decoder.py          decoder-only fine-tuning
  inference/
    generate.py               sample text from a checkpoint
  evaluation/
    verify.py                 collapse / latent-usage checks
runs/                         checkpoints (ignored)
scripts/
  slurm_train.sh              SLURM launcher
```

## Environment

The project uses [uv](https://docs.astral.sh/uv/). On NixOS/WSL2, enter the dev
shell:

```bash
nix develop
uv sync
```

`flake.nix` provides Python 3.12, uv, and the CUDA/Triton paths needed by
`torch.compile` on WSL2. You can also run a command directly:

```bash
nix develop --command python -m jepa train
```

## Data

Train the 32k tokenizer once

```bash
python -m jepa train-tokenizer
```

Then prepare 500M tokens of training chunks. The default streams FineWeb-Edu
directly:

```bash
python -m jepa prepare-data
```

To reuse the sampled text instead, save enough for the target size (about
2.3 GB for 500M tokens):

```bash
python -m jepa train-tokenizer \
    --save-text data/fineweb_edu_500m_tokens.txt \
    --megabytes 2300
python -m jepa prepare-data --text-file data/fineweb_edu_500m_tokens.txt
```

See [docs/architecture.md](docs/architecture.md) for the full data pipeline
and options.

## Training

```bash
python -m jepa train                  # prepared FineWeb-Edu chunks
python -m jepa train --resume         # continue from runs/jepa
```

For decoder-only fine-tuning against a frozen encoder and predictor (same
trainer, AMP, compile and checkpoint machinery as `train`, minus the JEPA loss):

```bash
python -m jepa train-decoder    # reads and writes runs/jepa
```

Two-stage training (decoder trained only once the hints are stable):

```bash
python -m jepa train --jepa-only   # encoder + predictor; the decoder is never run
python -m jepa train-decoder       # fresh decoder on the frozen encoder/predictor
python -m jepa train-decoder --resume --epochs 2   # keep training that decoder
```

It replaces the decoder in `runs/jepa` (encoder/predictor stay unchanged) and
logs to `runs/jepa/history_decoder.json`, leaving the joint `history.json`.

## Generation

```bash
python -m jepa generate
python -m jepa generate --prompt "When in Rome, " --steps 3 --temperature 0.8
```

## Evaluation

```bash
python -m jepa verify --model runs/jepa
```

Run `python -m jepa --help` for the command list. Detailed configuration,
training knobs, and the saved checkpoint layout are in
[docs/architecture.md](docs/architecture.md).
