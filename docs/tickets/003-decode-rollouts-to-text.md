# 003: Decode rollouts into text (clear vs blurry, side by side)

Status: done

## Why

The report's most persuasive figure is qualitative. Take one prompt, plan
several blocks ahead in latent space, and decode each plan into text:

- a **committed rollout** (π-sample, a few samples, plus π-argmax);
- the **μ path** (the blurry mean);
- the **P = 1 baseline's rollout** (once ticket 002 exists);
- the **real continuation**, for reference.

If committing works, the committed plans should read like specific, coherent
continuations (different from each other), while the μ plan drifts into
generic text. Today `generate` re-encodes the generated tokens after every
block, so it never decodes a pure latent plan, and `measure` decodes only
one step ahead.

## What

A rollout mode for generation, e.g.
`python -m jepa generate --plan-depth 4 --compare`, or a `samples` section in
`measure`. Prefer whichever keeps the code smaller. For each prompt:

1. Encode the prompt and take `z_t` at the last position.
2. Plan in latent space with `rollout(model.predictor, z_t, policy, depth)` for
   each committed policy. The μ path is the measurement baseline from
   `jepa/evaluation/measure.py`; reuse it rather than re-implementing it.
3. Decode block by block with `model.decoder.generate`. For block i the hint
   is (start = path latent i−1, with `z_t` for i = 1; committed = path
   latent i). The token context is the prompt plus all tokens decoded so far,
   and nothing is re-encoded. That is what makes it a decoded plan.
4. Output one record per prompt and condition: policy, sample index, the
   decoded text per block, and the real continuation. Write it as JSON, and
   optionally as a small HTML page laid out side by side for the report.

Prompts: a fixed, seeded set of validation chunks (their continuation is the
reference), plus optional `--prompt` strings.

## Caveats to state in the report (and in the docstring)

- **The decoder never saw μ hints.** It is trained on committed candidates, so
  decoding the μ path is out of distribution for it. That is part of the
  point (a blurry latent is not a valid plan), but it confounds the
  comparison. The fair comparison is with the **P = 1 model's own decoder**,
  which was trained on its own single (blurry) prediction. So this ticket is
  most useful after ticket 002.
- **Token sampling adds variance.** Decode greedily (`temperature=0`) for the
  main figure so the differences come from the plan, and show sampled decodes
  separately.
- **Pick samples honestly.** The prompts are seeded and fixed; don't
  cherry-pick. If showing a subset, say how it was chosen.

## Acceptance criteria

- A test with the tiny config produces, for one prompt, a decoded record for
  every condition with `depth × k` tokens each, and is deterministic under a
  fixed seed with greedy decoding.
- The decoded plan uses no re-encoding. A test can check that the encoder is
  called exactly once per prompt (e.g. by counting calls through a forward hook
  on the public module, without monkeypatching privates).
- `docs/architecture.md` documents the command and the caveats above.
