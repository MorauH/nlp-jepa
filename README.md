# JEPA language model

A joint-embedding predictive architecture (JEPA) for text. A causal encoder
maps token sequences to latents, a predictor proposes candidate future latents,
and the model commits to exactly one of them. An autoregressive decoder turns
the committed future into tokens. The paper asks whether committing keeps each
latent, and every chained rollout, a clear single future instead of a blurry
average. Terms are defined in [GLOSSARY.md](GLOSSARY.md).

## Environment

The project uses [uv](https://docs.astral.sh/uv/). On NixOS/WSL2, enter the
dev shell:

```bash
nix develop
uv sync
```

You can also run a command directly:

```bash
nix develop --command python -m jepa train
```

## Data

Train the 32k tokenizer once:

```bash
python -m jepa train-tokenizer
```

Then prepare 500M tokens of 1024-token training chunks. The default streams
FineWeb-Edu directly:

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

See [docs/architecture.md](docs/architecture.md) for the pipeline and options.

## Training

Training has two stages ([ADR 0001](docs/adr/0001-two-stage-training.md)).
First train the encoder and predictor:

```bash
python -m jepa train                  # prepared FineWeb-Edu chunks
python -m jepa train --resume         # continue from runs/jepa
```

Then train a decoder against the frozen JEPA:

```bash
python -m jepa train --stage decoder --from runs/jepa
python -m jepa train --stage decoder --resume --epochs 2
```

Each stage writes the full checkpoint and its own `history_<stage>.json` to
its output dir (`runs/jepa`, or the sibling `runs/jepa-decoder` for the decoder
stage). The decoder stage keeps the frozen encoder and predictor and replaces
only the decoder.

## Generation

```bash
python -m jepa generate --prompt "When in Rome, " --steps 3 --temperature 0.8
```

`--temperature` / `--top-k` control token sampling; `--commit-temperature` /
`--commit-top-k` control the commit from π.

## Measurement

```bash
python -m jepa measure --model runs/jepa
python -m jepa measure --model runs/jepa --json runs/jepa/metrics.json
```

Run `python -m jepa --help` for the command list. Configuration, checkpoint
layout and the meaning of each measurement section are in
[docs/architecture.md](docs/architecture.md).
