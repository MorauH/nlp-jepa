# Autoregressive token decoder conditioned on latent hints.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from ..config import DecoderConfig
from .blocks import DecoderLayer, RMSNorm, RotaryEmbedding
from .sampling import top_k_mask


def build_sliding_window_bias(seq_len: int, window: int, device, dtype):
    """Additive causal sliding-window mask for a token sequence."""
    positions = torch.arange(seq_len, device=device)
    causal = positions[:, None] >= positions[None, :]
    within = (positions[:, None] - positions[None, :]) < window
    keep = causal & within
    bias = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    bias.masked_fill_(~keep, torch.finfo(dtype).min)
    return bias


class DecoderModel(PreTrainedModel):
    """Causal token transformer conditioned on latent hints.

    The decoder is trained with standard next-token cross-entropy. It sees the
    last ``sliding_window`` tokens and a latent hint for the current block:
    ``z_t`` (the latent at the block start) and ``committed`` (the committed
    future at the block end). The hint is added to every token embedding in
    the block, so the token transformer keeps the grammar while the latent
    supplies the semantics. It is also re-injected (through a zero-initialized
    per-layer projection) before every layer, and the token embedding is tied
    to the output projection.
    """

    config_class = DecoderConfig

    def __init__(self, config: DecoderConfig):
        super().__init__(config)

        config.validate()
        self.pad_token_id = config.pad_token_id
        self.sliding_window = config.sliding_window

        self.rotary_emb = RotaryEmbedding(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.latent_proj_t = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.latent_proj_f = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.decoders = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Per-layer hint injection.  Zero-initialized so a fresh model is
        # exactly equivalent to input-only injection and the projections only
        # pick up a signal when the hint is useful.
        self.hint_proj = nn.ModuleList(
            [
                nn.Linear(config.hidden_size, config.hidden_size, bias=False)
                for _ in range(config.num_hidden_layers)
            ]
        )
        for projection in self.hint_proj:
            nn.init.zeros_(projection.weight)

        # Input/output embedding tying.  Declare the mapping so HF's save/load
        # deduplicates the shared tensor instead of failing on it.
        self._tied_weights_keys = {"lm_head.weight": "embedding.weight"}

        self.post_init()

        # ``post_init`` already ties via ``config.tie_word_embeddings``, but
        # assign explicitly so sharing does not depend on HF internals.
        self.lm_head.weight = self.embedding.weight
        # Small init on the tied head so the first loss is near ln(vocab_size).
        nn.init.normal_(self.embedding.weight, std=0.02)

    def _init_weights(self, module):
        # The decoder has no custom-initialized head that must survive
        # post_init; keep the constructor initialization.  This also keeps the
        # zero-initialized per-layer hint projections at zero.
        return

    def _hint(self, z_t, committed):
        """Project the block-start latent and committed future to decoder space."""
        return self.latent_proj_t(z_t) + self.latent_proj_f(committed)

    def _run(self, input_ids, hint, attention_mask=None):
        hidden_states = self.embedding(input_ids) + hint
        rope_rotations = self.rotary_emb(hidden_states)

        bias = build_sliding_window_bias(
            hidden_states.shape[1],
            self.sliding_window,
            hidden_states.device,
            hidden_states.dtype,
        ).view(1, 1, hidden_states.shape[1], hidden_states.shape[1])
        if attention_mask is not None:
            keep = attention_mask.to(hidden_states.dtype)
            bias = bias + (1.0 - keep)[:, None, None, :] * torch.finfo(
                hidden_states.dtype
            ).min

        for index, decoder in enumerate(self.decoders):
            # Re-inject the hint right before each layer so it does not have to
            # survive the whole residual stream.  Shared by the training
            # forward and generate() because both go through here.
            hidden_states = hidden_states + self.hint_proj[index](hint)
            hidden_states = decoder(
                hidden_states,
                rope_rotations,
                attn_bias=bias,
                is_causal=False,
            )
        return self.norm(hidden_states)

    def token_loss(
        self,
        z_t,
        committed,
        labels,
        attention_mask=None,
        chunk_size: int | None = None,
        loss_mask=None,
    ):
        """Summed next-token cross-entropy and token count for one hint.

        This is the single implementation of the decoder's token-level loss.
        :meth:`JepaModel.decoder_loss`, measurement, and the tests of the
        decoder all go through it.

        Args:
            z_t: (B, S, H) latent at the start of each token's block.
            committed: (B, S, H) hint at the end of each token's block.
            labels: (B, S) token ids.
            attention_mask: optional (B, S) padding mask.
            chunk_size: optional number of positions per ``lm_head`` chunk, so
                the full (B, S, vocab) logits tensor is never held.
            loss_mask: optional (B, S - 1) bool aligned with the hint
                positions.  False positions do not contribute to the loss or
                the count, which is how a block with no committed future is
                excluded.

        Returns:
            ``(total, count)``: the summed loss over counted target tokens and
            the number of those tokens.  Their ratio is the token-weighted
            mean loss.
        """
        input_ids = labels[:, :-1]
        targets = labels[:, 1:]
        steps = input_ids.shape[1]

        hint = self._hint(z_t[:, :steps], committed[:, :steps])
        mask = attention_mask[:, :-1] if attention_mask is not None else None
        hidden_states = self._run(input_ids, hint, mask)

        # Cross-entropy is fp32 even under autocast; accumulate there so a
        # chunked loss has the same precision as the single-pass one.
        total = torch.zeros((), device=hidden_states.device, dtype=torch.float32)
        count = torch.zeros((), device=hidden_states.device, dtype=torch.long)
        chunk = steps if chunk_size is None else max(1, int(chunk_size))
        for start in range(0, steps, chunk):
            end = min(start + chunk, steps)
            logits = self.lm_head(hidden_states[:, start:end, :])
            chunk_targets = targets[:, start:end]
            keep = chunk_targets != self.pad_token_id
            if loss_mask is not None:
                keep = keep & loss_mask[:, start:end]
            values = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                chunk_targets.reshape(-1),
                ignore_index=self.pad_token_id,
                reduction="none",
            ).reshape(chunk_targets.shape)
            total = total + values.masked_fill(~keep, 0.0).sum()
            count = count + keep.sum()
        return total, count

    def forward(
        self,
        z_t,
        committed,
        labels,
        attention_mask=None,
        loss_chunk_size: int | None = None,
    ):
        """Teacher-forced next-token loss.

        Args:
            z_t: (B, S, H) latent at the start of each token's block.
            committed: (B, S, H) committed future at the end of each token's
                block.
            labels: (B, S) token ids.
            loss_chunk_size: optional number of positions per ``lm_head`` chunk.
            attention_mask: optional (B, S) padding mask.
        """
        total, count = self.token_loss(
            z_t,
            committed,
            labels,
            attention_mask=attention_mask,
            chunk_size=loss_chunk_size,
        )
        return {"loss": total / count.clamp_min(1)}

    @torch.no_grad()
    def generate(
        self,
        z_t,
        committed,
        context_ids,
        k: int,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
    ):
        """Generate ``k`` tokens autoregressively from the latent hints.

        Args:
            z_t: (B, H) latent at the start of the block.
            committed: (B, H) committed future for the block.
            context_ids: (B, C) tokens already known/generated.
            k: number of tokens to generate.
            temperature: token sampling temperature; 0 means greedy.
            top_k: optional top-k cutoff.
        Returns:
            (B, k) generated token ids.
        """
        generated = []
        for _ in range(k):
            window = context_ids[:, -self.sliding_window :]
            hint = self._hint(
                z_t.unsqueeze(1), committed.unsqueeze(1)
            ).expand(-1, window.shape[1], -1)
            hidden_states = self._run(window, hint)
            logits = self.lm_head(hidden_states[:, -1, :])

            if temperature <= 0:
                next_ids = logits.argmax(dim=-1)
            else:
                logits = logits / temperature
                logits = top_k_mask(logits, top_k)
                probabilities = torch.softmax(logits, dim=-1)
                next_ids = torch.multinomial(
                    probabilities, num_samples=1
                ).squeeze(-1)

            generated.append(next_ids)
            context_ids = torch.cat([context_ids, next_ids[:, None]], dim=1)

        return torch.stack(generated, dim=1)
