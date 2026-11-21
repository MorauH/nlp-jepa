# AGENTS.md

JEPA text model (causal encoder → latent predictor → autoregressive decoder). Python 3.12, uv, PyTorch + HF `PretrainedConfig`/`PreTrainedModel`. Details: `README.md`, `docs/architecture.md`.

## Commands
- Env: `nix develop` (NixOS/WSL2), then `uv sync`. Run one-offs as `nix develop --command <cmd>`.
- Tests: `uv run python -m pytest -q`. Single test: `uv run python -m pytest -q tests/test_smoke.py::test_full_stack_forward`.
  - Use `python -m pytest`, not bare `pytest`/`uv run pytest`: `[tool.uv] package = false`, so `jepa` is only importable when the CWD is on `sys.path`. Bare pytest fails with `No module named 'jepa'`.
- No lint, formatter, typecheck, or CI config exists. Don't invent one.
- CLI entrypoint: `python -m jepa <train|train-decoder|generate|prepare-data|train-tokenizer|verify>` (`jepa/cli.py`). Each subcommand owns its argparse (`python -m jepa train --help`).
- Quick end-to-end training check: `python -m jepa train --subset` (needs the prepared data and tokenizer in `data/`).

## Layout / boundaries
- `jepa/config.py`: all component configs. `jepa/models/`: encoder, predictor, decoder, VISReg loss, `JepaModel` (no decoder), `JepaDecodeModel` (full stack).
- `jepa/paths.py`: `data/` and `runs/` are resolved from the package location, not the CWD.
- `data/` (tokenizer + prepared chunks) and `runs/` (checkpoints) are gitignored generated artifacts. Don't hand-edit them. Regenerating them streams FineWeb-Edu from the network and takes a long time.
- `docs/verification.md` contains temporary planning notes, not a spec of implemented behavior.

## Gotchas
- Training hyperparameters (model size, AMP, `torch.compile`, decoder window, loss weights) are **module-level constants** in `jepa/training/train.py`, not CLI flags. Only data paths, batch size, epochs, `--resume`, `--jepa-only`, and `--subset` are flags; `train-decoder` adds `--model-dir`, `--output-dir`, `--lr`, `--device`, and its own `--resume`.
- Both entry points share one trainer (`JepaTrainer`/`TrainConfig`) and the constants in `train.py`; `jepa/training/train_decoder.py` only defines its decoder-only CLI defaults (`DECODER_BATCH_SIZE`, `DECODER_LEARNING_RATE`, `DECODER_EPOCHS`).
- `JepaDecodeModel.freeze_jepa()` (used by `train-decoder`) freezes the encoder/predictor and makes `forward` skip the JEPA loss (they run under `torch.no_grad()`); its `train()` re-asserts eval mode for them. The frozen path selects the hindsight waypoint in fp32 so it matches the VISReg loss's winner under bf16 autocast. The mirror image, `skip_decoder()` (`train --jepa-only`), makes `forward` return `decoder_out=None`; the trainer drops whichever loss term is missing.
- `MODEL_MAX_LENGTH` in `train.py` must match `prepare-data --max-length` (default 1024).
- The encoder owns `hidden_size`, `vocab_size`, and `pad_token_id`. Build the other configs with `PredictorConfig.from_encoder` / `DecoderConfig.from_encoder` / `JepaConfig.from_encoder`. If you pass conflicting explicit values, construction raises `ValueError`.
- The checkpoint `config.json` nests the component configs (`encoder`/`predictor`/`decoder`/`visreg`) with no top-level duplicates. Keep it that way when adding config fields.
- `RIGHT_PADDED=True` assumes a right-padding tokenizer. Under that assumption, causal attention skips the padding mask so the code can use fused SDPA. Changing the padding side breaks this.
- The decoder gets **detached** latent hints, so the decoder loss must not update the encoder or predictor.
- Checkpoints are saved from the uncompiled module so state dicts don't get `_orig_mod.` prefixes. Keep this when touching saving code.
- On NixOS, `torch.compile` needs the dev shell (`openssl` on PATH, `TRITON_LIBCUDA_PATH`). Outside it, `torch.compile` quietly falls back to eager mode with only a warning.
- `scripts/slurm_train.sh` is the cluster launcher (L40S). It forwards its args to `python -m jepa train`.
