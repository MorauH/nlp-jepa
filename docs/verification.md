# JEPA decoder verification tests

Temporary notes. Goal: verify that the autoregressive decoder actually uses the
latent hints, rather than ignoring them and acting as a plain token LM.

Current setup:

- Training hints: `z_t` is the real encoder latent at the block start;
  `z_future` is the frozen predictor's selected waypoint at the block end.
- The decoder sees the last `sliding_window = 96` tokens.
- At inference, `z_future` is sampled from the predictor's scores.
- The hint is added to every token embedding; it is not a separate prefix token.

Note: older checkpoints were trained with the raw block-end latent, which
contains the last target token. The per-offset test is still useful for
verifying that the new predictor-waypoint hint is semantic rather than a leak.

---

## 1. Teacher-forced hint ablation (do this first)

On a held-out batch:

1. **Correct hints**: `z_t`, `z_future` from the real latents.
2. **Shuffled hints**: permute `z_future` across the batch (or roll it far
   along the sequence).
3. **Null hints**: zeros or a learned null.

Compare decoder cross-entropy.

- If `correct ≈ shuffled ≈ null`: the decoder ignores the hint.
- If `correct < shuffled`: the decoder uses it.

Pseudo-code:

```python
latents = encoder(labels, attention_mask)          # frozen encoder
z_t, z_future = build_block_hints(latents, k)

loss_correct = decoder(z_t, z_future, labels, attention_mask)["loss"]

perm = torch.randperm(labels.shape[0], device=labels.device)
loss_shuffled = decoder(z_t, z_future[perm], labels, attention_mask)["loss"]

loss_null = decoder(
    torch.zeros_like(z_t),
    torch.zeros_like(z_future),
    labels,
    attention_mask,
)["loss"]
```

Report the CE gap, not just the raw losses.

---

## 2. Per-block-offset analysis (leakage check)

Repeat the ablation, but split the loss by token position inside each
`k`-token block:

```text
offset 1: target t+1
offset 2: target t+2
...
offset k: target t+k
```

Interpretation:

- Gain spread across offsets 1..k → semantic guidance.
- Gain concentrated at offset k → the decoder is mostly leaking the last
  token from `z_future`.
- Gain zero everywhere → the hint is ignored.

Implementation sketch:

```python
input_ids = labels[:, :-1]
targets = labels[:, 1:]
hidden = decoder._run(input_ids, hint, mask)
logits = decoder.lm_head(hidden)
per_token = F.cross_entropy(
    logits.reshape(-1, logits.shape[-1]),
    targets.reshape(-1),
    reduction="none",
).view(labels.shape[0], -1)

positions = torch.arange(per_token.shape[1], device=per_token.device)
offset = (positions % k) + 1
for o in range(1, k + 1):
    print(o, per_token[:, offset == o].mean())
```

`_run` is currently private; this is fine for a diagnostic script.

---

## 3. Counterfactual generation

Same prompt/context, two hints:

- correct waypoint,
- shuffled/distractor waypoint from another sequence or continuation.

Generate greedily (`temperature=0`).

Measure:

- token sequences differ or not,
- first divergence position,
- KL / total variation between next-token distributions at each step.

If outputs are identical, the hint has no effect.

If outputs differ, use a semantic judge:

- encode the generated block with the frozen encoder and compare its latent to
  the correct `z_future` vs the distractor `z_future`;
- or use an external sentence-embedding model.

The correct hint should produce a generated latent closer to its own waypoint.

---

## 4. Predicted-waypoint test (real inference condition)

Training uses true latents; inference uses the predictor's waypoint. Test:

- predicted waypoint,
- shuffled predicted waypoint,
- true latent hint.

Compare teacher-forced loss and greedy generation.

- If predicted waypoint does not beat shuffled predicted → decoder ignores the
  predictor's guidance at inference.
- If true latent works but predicted does not → the predictor is the
  bottleneck, not the decoder.

Pseudo-code:

```python
candidates, scores = predictor(z_t.unsqueeze(1), return_all_timesteps=False)
pred = candidates[0, scores.argmax(-1)].unsqueeze(0)
```

---

## 5. No-hint baseline (gold standard)

Train a decoder with the hint removed or zeroed for the same number of steps
and data.

Compare validation CE and generation.

- If the hint-conditioned decoder is not better, the latents are decorative.
- If it is better, the hint is genuinely useful.

This costs another training run but is the strongest evidence.

---

## 6. Optional: hint dropout / classifier-free guidance

During training, randomly drop the hint (zero it or replace it with a learned
null) with probability `p`.

At inference:

- compare conditioned vs unconditioned generation,
- optionally use guidance:
  `logits = uncond + scale * (cond - uncond)`.

This makes the hint's effect measurable and prevents the decoder from quietly
ignoring it.

---

## Metrics

- CE with correct vs shuffled vs null hints.
- Per-offset CE (1..k).
- Greedy sequence divergence and first divergence step.
- KL / total variation between next-token distributions.
- Semantic similarity via the frozen encoder or an external embedding model.
- Win rate over a prompt set: fraction of prompts where the correct hint is
  closer to the true continuation than a distractor hint.
- Gradient norm of the loss w.r.t. the hint, and loss increase under hint
  perturbation.

---

## Success criteria

- Correct hints reduce CE meaningfully.
- Improvement is spread across block offsets, not just the last token.
- Greedy generations change semantically with different hints.
- Predicted waypoint beats shuffled predicted waypoint.
- No-hint baseline is worse.
- Generated-block latent is closer to the correct hint than to a distractor.

## Failure criteria

- CE gap between correct and shuffled hints is ~0.
- Outputs are identical with different hints.
- Only the last token of each block improves (leakage, not semantics).
- Predicted waypoint is no better than shuffled.
- No-hint baseline matches the hint-conditioned model.
