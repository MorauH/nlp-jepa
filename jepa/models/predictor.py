# Predicts latent_t+k from latent_t

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedModel

from ..config import PredictorConfig
from .blocks import DecoderLayer, RMSNorm


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

    def forward(self, hidden_states):
        """The forward pass of the predictor transformer.

        Args:
            hidden_states: (batch, prediction_count, hidden_size): the
                           expansion slots of one latent each.
        Returns:
            Hidden states of the same shape.

        The expansion slots are not temporal positions, so no RoPE is applied
        and attention is non-causal by default (``PredictorConfig.is_causal``).
        """
        for decoder in self.decoders:
            hidden_states = decoder(hidden_states, rope_rotations=None)

        return self.norm(hidden_states)


class PredictorModel(PreTrainedModel):
    """Predict ``z_{t+k}`` from the single latest latent ``z_t``.

    The causal encoder already summarizes the whole prefix in ``z_t``, so the
    predictor deliberately sees nothing else: ``z_t`` is the state.  Each
    position is predicted independently, which is what makes
    ``return_all_timesteps=True`` (training) and ``False`` (inference) agree.

    ``z_t`` is expanded into ``prediction_count`` slots, mixed by a small
    non-causal transformer across the slots, and turned into

    * candidates ``z_t + delta_i`` (residual in latent space), and
    * a value per candidate: the predicted distance ``d̂_i`` between that
      candidate and the real future (same distance as the loss).

    The values are returned as ``scores = -d̂`` so that "higher is better"
    holds everywhere: ``scores.argmax`` selects the candidate with the
    smallest expected distance, and a softmax over ``scores / T`` samples
    with a temperature ``T`` in distance units.

    Configured by :class:`jepa.config.PredictorConfig` and usable on its own::

        predictor = PredictorModel(PredictorConfig.from_encoder(enc_config))
        predictions, scores = predictor(latents, return_all_timesteps=False)
    """

    config_class = PredictorConfig

    def __init__(self, config: PredictorConfig):
        super().__init__(config)

        config.validate()

        self.hidden_expansion = MatrixExpansion(
            in_size=config.target_dim,
            out_size=config.hidden_size,
            out_dims=config.prediction_count,
        )

        self.transformer = PredictorTransformer(config=config)

        self.delta_head = PerDimLinear(
            dims=config.prediction_count,
            in_size=config.hidden_size,
            out_size=config.target_dim
        )

        # Predicted candidate-target distance (the "value" of each candidate).
        self.value_head = PerDimLinear(
            dims=config.prediction_count,
            in_size=config.hidden_size,
            out_size=1
        )

        self.post_init()

    def forward(self, input, return_all_timesteps: bool = True):
        """Predict k steps into the future from each latent in ``input``.

        Args:
            input: (batch, seq, target_dim) encoder latents.
            return_all_timesteps: If True, predict from every position
                                  independently.  If False, predict only from
                                  the final latent (the others are ignored).
        Returns:
            ``(predictions, scores)`` where ``predictions`` has shape
            (batch, seq, prediction_count, target_dim) and ``scores`` has
            shape (batch, seq, prediction_count) when
            ``return_all_timesteps`` is True, or
            (batch, prediction_count, target_dim) and
            (batch, prediction_count) otherwise.  ``scores`` is the negated
            predicted distance of each candidate to the real future.
        """
        assert input.ndim == 3, (
            f"Expected 3D tensor (batch, seq, target_dim), got shape {input.shape}"
        )

        batch_size, seq_len, target_dim = input.shape
        prediction_count = self.config.prediction_count

        if return_all_timesteps:
            z_t = input.reshape(batch_size * seq_len, target_dim)
        else:
            z_t = input[:, -1, :]
            seq_len = 1

        hidden_states = self.hidden_expansion(z_t)  # (N, prediction_count, hidden_size)
        hidden_states = self.transformer(hidden_states)

        pred_deltas = self.delta_head(hidden_states)  # (N, prediction_count, target_dim)
        values = self.value_head(hidden_states).squeeze(-1)  # (N, prediction_count)
        scores = -values

        predictions = z_t.unsqueeze(1) + pred_deltas  # (N, prediction_count, target_dim)

        if self.config.normalize_candidates:
            # Fix every candidate to the encoder-latent scale (per-dim RMS 1,
            # i.e. norm ~sqrt(target_dim)).  The loss compares directions only,
            # so without this a candidate's norm is free; unused slots drifted
            # to norms ~50x z_t, where the gradient through the loss's
            # normalization (~1/norm) vanishes and they could never recover.
            # No learnable gain: it would reintroduce the free scale.  Runs in
            # fp32 for precise statistics, like the other norms.
            predictions = F.rms_norm(
                predictions.float(),
                (target_dim,),
                eps=self.config.rms_norm_eps,
            ).to(predictions.dtype)

        if return_all_timesteps:
            return (
                predictions.view(batch_size, seq_len, prediction_count, target_dim),
                scores.view(batch_size, seq_len, prediction_count),
            )

        return predictions, scores
