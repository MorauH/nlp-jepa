# AGENTS.md

JEPA text model: causal encoder → latent predictor → commit → autoregressive
decoder. Python 3.12, uv, PyTorch + HF `PretrainedConfig`/`PreTrainedModel`.
Vocabulary: `GLOSSARY.md`. Design: `docs/architecture.md`.

## Commands
- Env: `nix develop` (NixOS/WSL2), then `uv sync`. Run one-offs as `nix develop --command <cmd>`.
- Tests: `uv run python -m pytest -q`. Single test: `uv run python -m pytest -q tests/test_smoke.py::test_full_stack_forward`.
  - Use `python -m pytest`, not bare `pytest`/`uv run pytest`: `[tool.uv] package = false`, so `jepa` is only importable when the CWD is on `sys.path`. Bare pytest fails with `No module named 'jepa'`.
- No lint, formatter, typecheck, or CI config exists. Don't invent one.
- CLI entrypoint: `python -m jepa <train|generate|prepare-data|train-tokenizer|measure>` (`jepa/cli.py`). Each subcommand owns its argparse (`python -m jepa train --help`).
- Training: `python -m jepa train` (stage jepa), `python -m jepa train --resume`, `python -m jepa train --stage decoder --from runs/jepa` (fresh decoder on a frozen JEPA, writes `runs/jepa-decoder`), `python -m jepa train --stage decoder --resume --epochs 2`, `python -m jepa train --config docs/experiments/p1-baseline.json --seed 1 --output-dir runs/p1-seed1`.
- Quick end-to-end check: `python -m jepa train --subset` (needs the prepared data and tokenizer in `data/`).
- Measurement: `python -m jepa measure --model runs/jepa --json runs/jepa/metrics.json`.
- Decoded plans (qualitative figure): `python -m jepa generate --plan-depth 4 --baseline-model runs/p1 --json runs/jepa/plans.json --html runs/jepa/plans.html`.

## Layout / boundaries
- `jepa/config.py`: every component config (`EncoderConfig`, `PredictorConfig`, `DecoderConfig`, `VISRegConfig`, `CommitmentLossConfig`, `JepaConfig`) and `apply_overrides`.
- `docs/experiments/`: override presets for baselines and ablations, with a README mapping each to its report question.
- `jepa/models/`: `encoder.py`, `predictor.py`, `decoder.py`, `commitment.py` (`distance` — the only `2 - 2 cos` — `Futures`, policies, `commit`, `rollout`), `sampling.py` (`top_k_mask`), `losses.py` (`VISReg`, `CommitmentLoss`), `jepa.py` (`JepaModel`, `build_block_hints`).
- `jepa/training/`: `data.py` (dataset loading + collation + length bucketing), `trainer.py` (`JepaTrainer`/`TrainConfig` and every engineering constant), `train.py` (CLI, per-stage defaults, model building/loading, paths).
- `jepa/evaluation/measure.py`: the paper's numbers. `jepa/evaluation/synthetic.py`: the two-successor Markov chain and its metrics. `jepa/inference/generate.py`: sampling and decoded latent plans (`--plan-depth`).
- `jepa/paths.py`: `data/` and `runs/` are resolved from the package location, not the CWD.
- `data/` (tokenizer + prepared chunks) and `runs/` (checkpoints) are gitignored generated artifacts. Don't hand-edit them. Regenerating them streams FineWeb-Edu from the network and takes a long time.
- `docs/architecture.md` describes the implemented model; `docs/vision.md` is planning, not implemented behavior; `docs/adr/` records decisions; `docs/tickets/` holds open work items (not implemented).
- `docs/report/report.html`: the draft research report. Section 5.1 is filled in; other results are placeholders (marked `TODO`) until the experiments in `docs/tickets/` are run.
- `docs/report/synthetic.py`: the synthetic experiment (section 5.1). `python docs/report/synthetic.py` (CPU, ~15 s) writes `docs/report/synthetic.json` and `synthetic.svg` (Figure 1) itself; don't redirect stdout (the `nix develop` banner goes there). The world, training loop and synthetic-only metrics live in `jepa/evaluation/synthetic.py`, shared with `tests/test_research_claim.py`; the μ path is `measure.mu_path`.

## Gotchas
- Config defaults are the main experiment; variants are override files in `docs/experiments/` (`train --config PATH`, applied by `apply_overrides` in `jepa/config.py`). Don't override model size, window, candidate count, loss weights or annealing in the scripts. Unknown override keys raise `KeyError` before data loads. The decoder stage accepts only `decoder.*` overrides. `--config` and `--seed` are errors with `--resume`, which keeps the checkpoint's config and the seed saved in `trainer_state_<stage>.pt`. On CUDA, same-seed runs differ slightly (non-deterministic Flash Attention backward). On CPU they are identical. Checkpoint `config.json` nests `encoder`/`predictor`/`decoder`/`visreg`/`commitment` with no top-level duplicates.
- Engineering constants (AMP, `torch.compile`, padding, logging, save interval, LR schedule, weight decay, default seed `SEED`) are module-level in `jepa/training/trainer.py`, not CLI flags or `TrainConfig` fields. `TrainConfig` holds only per-run values (including `seed`). AMP and `torch.compile` apply on CUDA only, so CPU runs and tests stay eager fp32. Flags: data paths, `--tokenizer`, `--batch-size`, `--epochs`, `--stage`, `--from`, `--output-dir`, `--resume`, `--config`, `--seed`, `--lr`, `--device`, `--subset`.
- The encoder owns `hidden_size`, `vocab_size`, and `pad_token_id`. Build the other configs with `PredictorConfig.from_encoder` / `DecoderConfig.from_encoder` / `JepaConfig.from_encoder`. Conflicting explicit values raise `ValueError`.
- The encoder is causal and batches are **right-padded**, so attention skips the padding mask and uses fused SDPA. Changing the padding side breaks this.
- Two stages share one trainer. The JEPA stage (`stage="jepa"`) never runs the decoder and logs `jepa_*` keys; the checkpoint keeps a loadable decoder. The decoder stage (`stage="decoder"`) freezes encoder/predictor, runs them under `torch.no_grad()`, trains only the decoder, and logs `decoder_*` + perplexity. `decoder_loss(labels, attention_mask, policy=None)` takes the commit rule as an argument: `None` means `DecoderConfig.train_policy` (training), `PiArgmax()` is the inference condition (validation). The model's train/eval flag no longer chooses the policy; the trainer only sets the stage modules' modes. Hints need no detach because the encoder and predictor run under `torch.no_grad()`.
- Decoder stage output: `--output-dir` defaults to a sibling `<from>-decoder` (e.g. `runs/jepa-decoder`), and `--resume` loads the model/history/state from the output dir. `--output-dir == --from` without `--resume` is rejected (it would overwrite the JEPA checkpoint).
- Each stage writes `history_<stage>.json` and, on every save, `trainer_state_<stage>.pt` (optimizer, scheduler, `global_step`, epoch, batches consumed, running totals, RNG). `--resume` continues the schedule, restores Adam/scheduler state, and replays the epoch's deterministic bucket order to skip consumed batches; a missing state file on `--resume` is a hard error.
- `build_block_hints` returns `(z_t, committed, valid)`: the last block of a sequence has no latent `horizon` tokens ahead, so `valid` is False there and both `decoder_loss` and `measure` mask those target tokens out. Hints: the raw block-end latent leaks the block's last token; a hindsight commit is a predictor output but its choice among `P` candidates used the real future, so it carries up to `log2(P)` bits (intended teacher forcing); π-argmax carries none. Validation and measurement report both.
- `JepaModel.encode` is the single encoder path (public). Batches are right-padded and the encoder is causal, so it drops the padding mask; losses keep the mask only to validate positions.
- Checkpoints are saved from the uncompiled module so state dicts don't get `_orig_mod.` prefixes. Keep this when touching saving code.
- `--max-length` in `prepare-data` must match `EncoderConfig.max_position_embeddings` (default 1024).
- `DecoderConfig.loss_chunk_size` bounds the decoder logits tensor; measurement's `--loss-chunk-size` overrides it for one run.
- On NixOS, `torch.compile` needs the dev shell (`openssl` on PATH, `TRITON_LIBCUDA_PATH`). Outside it, `torch.compile` quietly falls back to eager mode with only a warning.
- `tests/test_research_claim.py` is the main test of the idea (candidates on modes, μ blur, committed rollout on the manifold, hindsight rollout error). `tests/test_measure.py` covers the measurement dict. Keep both green.
- `scripts/slurm_train.sh` is the cluster launcher (L40S). It forwards its args to `python -m jepa train`.
