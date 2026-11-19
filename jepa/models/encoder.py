# Encodes tokens to hidden state, where hidden[-1] = latent_t

from __future__ import annotations

from torch import nn
from transformers import PreTrainedModel

from ..config import EncoderConfig
from .blocks import (
    DecoderLayer,
    RMSNorm,
    RotaryEmbedding,
    build_attention_bias,
)


class EncodeTransformer(PreTrainedModel):
    """A language model based on the Transformer architecture, without un-embedding.

    Configured by :class:`jepa.config.EncoderConfig` and usable on its own::

        encoder = EncodeTransformer(EncoderConfig(vocab_size=len(tokenizer)))
        latents = encoder(tokens, attention_mask=mask)
    """

    config_class = EncoderConfig

    def __init__(self, config: EncoderConfig):
        super().__init__(config)

        config.validate()
        self.pad_token_id = config.pad_token_id

        self.rotary_emb = RotaryEmbedding(config)

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        self.decoders = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # The final norm is scale-free so the VISReg shape regularizer cannot
        # shrink a learnable per-dimension gain and collapse the latent scale.
        self.norm = RMSNorm(config, elementwise_affine=False)

        self.post_init()

    def forward(self, labels, attention_mask=None):
        """The forward pass of the Transformer encoder.

        Args:
            labels: The input tensor (2D), a batch of integer-encoded texts.
            attention_mask: Optional (2D) mask, 1 = real token, 0 = padding.
        Returns:
            Hidden states of shape (batch, sequence, hidden_size).
        """
        # labels : (batch, sequence)
        rope_rotations = self.rotary_emb(labels)

        hidden_states = self.embedding(labels)  # (B, N, hidden_size)

        # Build the padding/causal bias once and share it across every layer.
        # With right-aligned padding the caller can pass attention_mask=None,
        # which lets SDPA use its fused causal (Flash Attention) kernel.
        attn_bias, is_causal = build_attention_bias(
            attention_mask, self.config.is_causal, hidden_states.dtype
        )

        for decoder in self.decoders:
            hidden_states = decoder(
                hidden_states,
                rope_rotations,
                attn_bias=attn_bias,
                is_causal=is_causal,
            )

        return self.norm(hidden_states)
