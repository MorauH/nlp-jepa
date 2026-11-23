# 001: Initialize the decoder so its first loss is near uniform

Status: done

## Why

A freshly initialized decoder has a token loss of about **143 nats** on random
tokens. A uniform guess over the 32k vocabulary is ln(32000) ≈ **10.4**. The
first part of every decoder run is spent shrinking the logits instead of
learning, and the loss curves in the report start at a misleading height.

This predates the refactor: `main` gives the same 143.2 on the same inputs.

Cause (`jepa/models/decoder.py`):

- `self.embedding = nn.Embedding(...)` keeps PyTorch's default init, N(0, 1)
  per entry.
- `lm_head.weight` is tied to that table.
- `_init_weights` is a deliberate no-op (it protects the zero-initialized
  `hint_proj`), so HF's usual small init never runs.
- The final hidden state leaves an RMSNorm with RMS ≈ 1, so its norm is about
  √448 ≈ 21. Each logit is a dot product with an N(0, 1) row, giving a logit
  std of about 21 and a very peaked, essentially random softmax.

## What

Initialize the tied token table at a small scale in `DecoderModel.__init__`,
after `post_init()` and the tying, for example `nn.init.normal_(std=0.02)`
(the GPT-2 convention) or `std = hidden_size ** -0.5`. Keep `hint_proj` at zero.

Check the encoder's embedding as well. It is not tied to an output head, and
VISReg normalizes the latent scale, so it may be fine. Record the decision in
the docstring either way.

## Acceptance criteria

- A test builds a decoder with a realistic vocabulary (e.g. 32000, hidden 64,
  so it stays fast) and checks that its `token_loss` mean on random tokens is
  within about 1 nat of ln(vocab_size).
- `hint_proj` weights are still exactly zero after construction (the existing
  test still passes).
- The real-data smoke run (`train --stage decoder --subset`, output under
  `/tmp`) starts near 10.4 instead of about 120.
- `docs/architecture.md` mentions the initialization in one sentence.
