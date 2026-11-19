# Predicts latent_t+k from latent_t

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedModel

from ..config import PredictorConfig
from .blocks import DecoderLayer, RMSNorm, RotaryEmbedding


class MatrixExpansion(nn.Module):
    """Expansion layer. From (batch, in_size) to (batch, out_dims, out_size)."""

    def __init__(self, in_size: int, out_size: int, out_dims: int):
        super().__init__()

        self.in_size = in_size
        self.output_size = out_size
        self.output_dims = out_dims

        self.weights = nn.Parameter(torch.randn(out_dims, out_size, in_size))
        self.bias = nn.Parameter(torch.zeros(out_dims, out_size))

        nn.init.kaiming_uniform_(self.weights, nonlinearity="linear")

    def forward(self, hidden_vector):
        """
        hidden_vector: Tensor of shape (batch, hidden_size)
        Returns: Tensor of shape (batch, out_dims, output_size)
        """
        # bh: batch, hidden_size
        # xoh: out_dims matrices, out_size, hidden_size
        # bxo: batch, out_dims, out_size
        out = torch.einsum("bh, xoh -> bxo", hidden_vector, self.weights)

        if self.bias is not None:
            # Keep the activation in the autocast dtype: adding an fp32 bias to
            # a bf16 matmul result would promote the whole residual stream.
            out = out + self.bias.to(out.dtype)

        return out


class PerDimLinear(nn.Module):
    """Applies a different linear layer to each of the `dims` positions.
    Input: (batch, dims, in_size)
    Output: (batch, dims, out_size)
    """
    def __init__(self, dims: int, in_size: int, out_size: int):
        super().__init__()

        self.dims = dims
        self.in_size = in_size
        self.output_size = out_size

        # Weight shape matches the existing dims: (dims, out_size, in_size)
        self.weights = nn.Parameter(torch.empty(dims, out_size, in_size))
        self.bias = nn.Parameter(torch.zeros(dims, out_size))

        nn.init.kaiming_uniform_(self.weights, nonlinearity="linear")

    def forward(self, hidden_states):
        """
        hidden_states: Tensor of shape (batch, dims, in_size)
        Returns: Tensor of shape (batch, dims, output_size)
        """
        # b = batch, d = dims, h = in_size, o = output_size
        # x: (b, d, h), weights: (d, o, h) -> out: (b, d, o)
        out = torch.einsum("bdh, doh -> bdo", hidden_states, self.weights)

        if self.bias is not None:
            # Keep the activation in the autocast dtype (preserves bf16/fp16 safety)
            out = out + self.bias.to(out.dtype)

        return out


class PredictorTransformer(nn.Module):
    """A Transformer stack used by the predictor."""

    def __init__(self, config: PredictorConfig):
        super().__init__()

        self.decoders = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Keep the predictor output at a fixed scale; a learnable final gain
        # would otherwise be driven to zero by the VISReg shape term.
        self.norm = RMSNorm(config, elementwise_affine=False)

    def forward(self, hidden_states, attn_bias=None):
        """The forward pass of the predictor transformer.

        Args:
            hidden_states: (batch, tokens, hidden_size): the expansion slots,
                           optionally followed by context tokens.
            attn_bias: optional additive SDPA bias, broadcastable to
                       (batch, heads, tokens, tokens).
        Returns:
            Hidden states of the same shape.

        The expansion slots are not temporal positions, so no RoPE is applied
        and attention is non-causal by default (``PredictorConfig.is_causal``).
        Context tokens carry a learned offset embedding instead.
        """
        for decoder in self.decoders:
            hidden_states = decoder(
                hidden_states, rope_rotations=None, attn_bias=attn_bias
            )

        return self.norm(hidden_states)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # A small multi-layer perceptron to score each token in the sequence
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        """
        x shape: (batch, sequence, hidden_size)
        Returns: (batch, hidden_size)
        """
        scores = self.attention_net(x)  # (batch, sequence, 1)

        # Normalize scores across the sequence dimension to sum to 1
        weights = F.softmax(scores, dim=1)  # (batch, sequence, 1)

        # Compute the weighted sum (context vector) across the sequence
        return torch.sum(x * weights, dim=1)  # (batch, hidden_size)


class PredictorModel(PreTrainedModel):
    """Predict input_hidden[i, t+k, :] based on input_hidden[i, t, :].
    Does p predictions, along with a probability assigned to each.

    With ``PredictorConfig.trunk_layers > 0`` a causal transformer first maps
    the latent sequence to one hidden state per position (temporal RoPE, fused
    causal attention); the candidate head then expands that hidden state
    instead of the raw latent.  Candidates are still ``z_t + delta_i``, so the
    residual lives in latent space.

    Configured by :class:`jepa.config.PredictorConfig` and usable on its own::

        predictor = PredictorModel(PredictorConfig.from_encoder(enc_config))
        predictions, scores = predictor(z_t, return_all_timesteps=False)
    """

    config_class = PredictorConfig

    def __init__(self, config: PredictorConfig):
        super().__init__(config)

        config.validate()

        # Shared trunk over the latent sequence (``trunk_layers == 0`` keeps
        # the legacy position-independent predictor).
        self.trunk_layers = config.trunk_layers
        if self.trunk_layers > 0:
            self.trunk_in = nn.Linear(config.target_dim, config.hidden_size)
            self.trunk_rotary = RotaryEmbedding(config)
            self.trunk = nn.ModuleList(
                [DecoderLayer(config) for _ in range(self.trunk_layers)]
            )
            self.trunk_norm = RMSNorm(config)

        self.hidden_expansion = MatrixExpansion(
            in_size=(
                config.hidden_size if self.trunk_layers > 0 else config.target_dim
            ),
            out_size=config.hidden_size,
            out_dims=config.prediction_count,
        )

        self.transformer = PredictorTransformer(config=config)

        self.delta_head = PerDimLinear(
            dims=config.prediction_count,
            in_size=config.hidden_size,
            out_size=config.target_dim
        )

        self.score_head = PerDimLinear(
            dims=config.prediction_count,
            in_size=config.hidden_size,
            out_size=1
        )

        # Optional context tokens: the last ``context_window`` latents, so the
        # candidates and the score head are not limited to the single z_t.
        self.context_window = config.context_window
        if self.context_window > 0:
            self.context_proj = nn.Linear(config.target_dim, config.hidden_size)
            # Learned offset embedding (oldest .. z_t); the slots use none.
            self.context_offset = nn.Parameter(
                torch.randn(self.context_window, config.hidden_size) * 0.02
            )

        self.post_init()

    def _context_index(self, batch_size, seq_len, device):
        """Flat (batch*seq*W,) indices of each position's context window.

        Row ``(b, t)`` of the gathered windows holds the projected latents
        ``(b, t-W+1 .. t)``, clamped to 0 at the start of the sequence.  Flat
        indices let us gather straight from the (batch*seq, hidden) projections
        instead of projecting W shifted copies of every latent.
        """
        W = self.context_window
        positions = torch.arange(seq_len, device=device)[:, None]
        offsets = torch.arange(W, device=device)[None, :] - (W - 1)
        relative = (positions + offsets).clamp_min(0)  # (seq, W)
        batch_offset = (
            torch.arange(batch_size, device=device) * seq_len
        )[:, None, None]
        return (relative[None] + batch_offset).reshape(-1)

    def _context_windows(self, input, return_all_timesteps: bool, projected=None):
        """Return ``(windows, present)`` for the context tokens.

        ``windows`` is (N, W, target_dim) with the oldest latent first and
        z_t last; ``present`` is (N, W) bool, False where the window reaches
        before the start of the sequence (zero-padded and masked out).
        Assumes no left padding inside ``input`` (right padding is fine:
        a valid position only looks back at earlier, valid positions).

        When ``projected`` is given (only for ``return_all_timesteps``), it is
        ``context_proj`` applied to every input position, shaped
        (batch*seq, hidden).  The windows are then gathered from it, which is
        W times less projection work than unfolding first.
        """
        W = self.context_window
        batch_size, seq_len, target_dim = input.shape
        if not return_all_timesteps:
            input = input[:, -W:, :]
            seq_len = input.shape[1]

        positions = torch.arange(seq_len + W - 1, device=input.device)
        present_all = positions >= W - 1  # (seq + W - 1,)

        if return_all_timesteps:
            present = present_all.unfold(0, W, 1)  # (seq, W)
            present = present.unsqueeze(0).expand(batch_size, -1, -1)
            present = present.reshape(batch_size * seq_len, W)
            if projected is not None:
                index = self._context_index(batch_size, seq_len, input.device)
                windows = projected.index_select(0, index).view(
                    batch_size, seq_len, W, -1
                )
                # Positions before the sequence start are masked out of the
                # attention bias; zero them so no stale latent is read.
                windows = windows.masked_fill(
                    ~present.view(batch_size, seq_len, W, 1), 0.0
                )
                return windows.reshape(batch_size * seq_len, W, -1), present
            padded = F.pad(input, (0, 0, W - 1, 0))  # (B, seq + W - 1, D)
            # unfold -> (B, seq, D, W); window j of position t is t - W + 1 + j
            windows = padded.unfold(1, W, 1).permute(0, 1, 3, 2)
            windows = windows.reshape(batch_size * seq_len, W, target_dim)
        else:
            padded = F.pad(input, (0, 0, W - 1, 0))
            windows = padded[:, -W:, :]
            present = present_all[-W:].unsqueeze(0).expand(batch_size, -1)
        return windows, present

    def _run_trunk(self, input):
        """Causal transformer over the latent sequence.

        Args:
            input: (batch, seq, target_dim) encoder latents.
        Returns:
            (batch, seq, hidden_size): one hidden state per position, each a
            causal summary of the latents up to and including that position.

        ``PredictorConfig.is_causal`` is False because the candidate slots are
        not temporal positions, so causality is requested per call.  No
        attention mask is needed: with right-aligned padding a valid position
        never attends to a pad, which keeps the fused causal SDPA kernel.
        """
        hidden_states = self.trunk_in(input)
        rope_rotations = self.trunk_rotary(hidden_states)
        for decoder in self.trunk:
            hidden_states = decoder(
                hidden_states,
                rope_rotations,
                attn_bias=None,
                is_causal=True,
            )
        return self.trunk_norm(hidden_states)

    def forward(self, input, return_all_timesteps: bool = True):
        """Predict k steps into the future for each position in ``input``.

        Args:
            input: (batch, seq, target_dim)
            return_all_timesteps: If True, return predictions for every input
                                  position. If False, predict only from the
                                  final step.  With ``context_window > 0`` the
                                  previous latents of ``input`` are used as
                                  context, so pass the whole latent sequence,
                                  not only z_t.
        Returns:
            ``(predictions, scores)`` where ``predictions`` has shape
            (batch, seq, prediction_count, target_dim) and ``scores`` has
            shape (batch, seq, prediction_count) when
            ``return_all_timesteps`` is True, or
            (batch, prediction_count, target_dim) and
            (batch, prediction_count) otherwise.
        """
        assert input.ndim == 3, (
            f"Expected 3D tensor (batch, seq, target_dim), got shape {input.shape}"
        )

        batch_size, seq_len, target_dim = input.shape
        prediction_count = self.config.prediction_count

        if return_all_timesteps:
            input_flat = input.reshape(batch_size * seq_len, target_dim)
        else:
            input_flat = input[:, -1, :]
            seq_len = 1

        if self.trunk_layers > 0:
            # Run the trunk over the whole sequence even when only the final
            # position is wanted: the last hidden state is the causal summary
            # of the prefix, and callers pass the full latent sequence anyway.
            trunk_hidden = self._run_trunk(input)
            if return_all_timesteps:
                head_input = trunk_hidden.reshape(batch_size * seq_len, -1)
            else:
                head_input = trunk_hidden[:, -1, :]
        else:
            head_input = input_flat

        hidden_states = self.hidden_expansion(head_input)  # (B*seq, prediction_count, hidden_size)

        if self.context_window > 0:
            # Project once per position and gather the shifted windows from the
            # projected latents.  Unfolding first would run context_proj on W
            # copies of every latent (W times the work and W activation
            # buffers); a Linear commutes with the gather, so this is exact.
            if return_all_timesteps:
                windows, present = self._context_windows(
                    input,
                    return_all_timesteps,
                    projected=self.context_proj(input_flat),
                )
                context = windows
            else:
                windows, present = self._context_windows(
                    input, return_all_timesteps
                )
                context = self.context_proj(windows)
            context = context + self.context_offset.to(context.dtype)
            hidden_states = torch.cat(
                [hidden_states, context.to(hidden_states.dtype)], dim=1
            )  # (B*seq, prediction_count + W, hidden_size)

            # Mask context keys that fall before the sequence start.  Slot keys
            # are always present, so no query row is fully masked.
            keep = torch.cat(
                [
                    torch.ones(
                        present.shape[0],
                        prediction_count,
                        dtype=torch.bool,
                        device=present.device,
                    ),
                    present,
                ],
                dim=1,
            )
            attn_bias = torch.zeros(
                keep.shape, dtype=hidden_states.dtype, device=keep.device
            ).masked_fill(~keep, torch.finfo(hidden_states.dtype).min)
            attn_bias = attn_bias[:, None, None, :]  # (N, 1, 1, slots + W)

            hidden_states = self.transformer(hidden_states, attn_bias=attn_bias)
            hidden_states = hidden_states[:, :prediction_count, :]
        else:
            hidden_states = self.transformer(hidden_states)

        pred_deltas = self.delta_head(hidden_states)  # (B*seq, prediction_count, target_dim)
        scores = self.score_head(hidden_states)  # (B*seq, prediction_count, 1)

        predictions = input_flat.unsqueeze(1) + pred_deltas  # (B*seq, prediction_count, target_dim)

        if return_all_timesteps:
            return (
                predictions.view(batch_size, seq_len, prediction_count, target_dim),
                scores.view(batch_size, seq_len, prediction_count),
            )

        return predictions, scores.squeeze(-1)
