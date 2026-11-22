# Blocks for building transformers

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')

        self.branch_silu = nn.Sequential(
            nn.Linear(
                in_features = config.hidden_size,
                out_features = config.intermediate_size,
                bias = False
            ),
            nn.SiLU()
        )
        self.branch_lin = nn.Linear(
            in_features = config.hidden_size,
            out_features = config.intermediate_size,
            bias = False
        )

        self.linear = nn.Linear(
            in_features = config.intermediate_size,
            out_features = config.hidden_size,
            bias = False
        )

    def forward(self, hidden_states):
        out_silu = self.branch_silu(hidden_states)
        out_lin = self.branch_lin(hidden_states)
        
        combine = out_silu * out_lin

        return self.linear(combine)


class RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config, elementwise_affine: bool = True):
        super().__init__()
        self.rms_norm = nn.RMSNorm(
            normalized_shape=(config.hidden_size,),
            eps=config.rms_norm_eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, hidden_states):
        # Norms keep fp32 parameters.  Under bf16 autocast, run the norm in
        # fp32 (and return the activation dtype) so the fused kernel is used
        # and the statistics stay precise.
        out_dtype = hidden_states.dtype
        return self.rms_norm(hidden_states.float()).to(out_dtype)


def build_attention_bias(attention_mask, causal, dtype):
    """Turn a (B, T) keep-mask into one reusable SDPA additive bias.

    Returns ``(bias, is_causal)``.  When there is no padding (or the caller has
    already dropped the mask because padding is right-aligned) this returns
    ``(None, causal)``, which lets SDPA use the fused causal kernel.

    The bias is built once per forward and shared by every layer.  Building it
    inside each layer allocated a (B, 1, 1, T) tensor and a (T, T) causal tensor
    per layer, per step.
    """
    if attention_mask is None:
        return None, causal

    keep = attention_mask[:, None, None, :].to(dtype=dtype)
    bias = (1.0 - keep) * torch.finfo(dtype).min

    if causal:
        # SDPA does not accept an explicit mask together with is_causal=True,
        # so fold the causal constraint into the additive bias.
        seq = attention_mask.shape[1]
        causal_bias = torch.full(
            (seq, seq),
            torch.finfo(dtype).min,
            device=attention_mask.device,
            dtype=dtype,
        ).triu(1)
        bias = bias + causal_bias
        causal = False

    return bias, causal


class Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""

    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.is_causal = config.is_causal

        self.w_q_norm = nn.Sequential(
            nn.Linear(
                in_features = config.hidden_size,
                out_features = config.hidden_size,
                bias = False
            ),
            RMSNorm(config)
        )
        self.w_k_norm = nn.Sequential(
            nn.Linear(
                in_features = config.hidden_size,
                out_features = config.hidden_size,
                bias = False
            ),
            RMSNorm(config)
        )
        self.w_v = nn.Linear(
            in_features = config.hidden_size,
            out_features = config.hidden_size,
            bias = False
        )
        self.w_o = nn.Linear(
            in_features = config.hidden_size,
            out_features = config.hidden_size,
            bias = False
        )


    def forward(
        self,
        hidden_states,
        rope_rotations=None,
        attention_mask=None,
        attn_bias=None,
        is_causal=None,
    ):
        batch_size, seq_size, hidden_size = hidden_states.shape

        query_norm = self.w_q_norm(hidden_states)   # (batch, seq, hidden_size)
        key_norm = self.w_k_norm(hidden_states)     # (batch, seq, hidden_size)
        value = self.w_v(hidden_states)             # (batch, seq, hidden_size)

        q = query_norm.view(batch_size, seq_size, self.num_heads, self.head_dim).transpose(1, 2) # (batch, num_heads, seq, head_dim)
        k = key_norm.view(batch_size, seq_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = value.view(batch_size, seq_size, self.num_heads, self.head_dim).transpose(1, 2)
       
        if rope_rotations is not None:
            q, k = apply_rotary_pos_emb(q=q, k=k, rope_rotations=rope_rotations)

        if is_causal is None:
            is_causal = bool(self.is_causal)

        # Callers that share one bias across layers pass ``attn_bias`` directly.
        # The ``attention_mask`` path is kept for standalone use.
        if attn_bias is None and attention_mask is not None:
            attn_bias, is_causal = build_attention_bias(
                attention_mask, is_causal, q.dtype
            )
        elif attn_bias is not None and attn_bias.dtype != q.dtype:
            # Shared bias may have been built before autocast changed q to
            # bf16; SDPA requires the mask and query dtypes to agree.
            attn_bias = attn_bias.to(q.dtype)

        attn_out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            is_causal=is_causal
        ) # (batch, num_heads, seq, head_dim)

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_size, hidden_size) # (batch, seq, hidden_size)

        return self.w_o(attn_out)


class DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()

        self.attn = Attention(config)
        self.attn_norm = RMSNorm(config)
        
        self.mlp = MLP(config)
        self.mlp_norm = RMSNorm(config)


    def forward(
        self,
        hidden_states,
        rope_rotations=None,
        attention_mask=None,
        attn_bias=None,
        is_causal=None,
    ):

        attn_norm = self.attn_norm(
            self.attn(
                hidden_states,
                rope_rotations,
                attention_mask=attention_mask,
                attn_bias=attn_bias,
                is_causal=is_causal,
            )
        )
        hidden_states = hidden_states + attn_norm

        mlp_norm = self.mlp_norm(self.mlp(hidden_states))
        hidden_states = hidden_states + mlp_norm

        return hidden_states


#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])
    # RoPE is computed in fp32, but the projection itself can run in the
    # autocast dtype.  Cast explicitly so bf16/fp16 queries do not silently
    # get promoted to fp32 here.
    cos = cos.to(q.dtype).unsqueeze(unsqueeze_dim)
    sin = sin.to(q.dtype).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        self.rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        self.dim = int(head_dim * partial_rotary_factor)

    @torch.no_grad()
    def forward(self, x):
        # Computed on the input device so loading a model with
        # ``from_pretrained`` (which may initialize on the meta device) works.
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64, device=x.device).to(
                    dtype=torch.float
                )
                / self.dim
            )
        )

        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = (
            inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin

