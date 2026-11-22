# Predicts latent_t+k from latent_t

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from transformers import PreTrainedModel

from ..config import PredictorConfig
from .blocks import DecoderLayer, RMSNorm


class PredictorOutput(NamedTuple):
    """What :class:`PredictorModel` returns (unpacks as a 3-tuple).

    Shapes are for ``return_all_timesteps=True``; with ``False`` the ``seq``
    dimension is dropped.

    candidates: (batch, seq, P, D) candidate future latents.
    scores:     (batch, seq, P) win-probability logits (pi): how likely each
                candidate is the closest one to the real future.  Sample from
                these to commit to one future.
    mean:       (batch, seq, D) mu, the predicted mean of the *normalized*
                future latent.  Its norm measures how spread the real futures
                are (~1 = predictable, short = divergent); see
                :func:`expected_distances`.
    """

    candidates: torch.Tensor
    scores: torch.Tensor
    mean: torch.Tensor


def expected_distances(candidates, mean):
    """Expected candidate-target distance under the loss's normalized metric.

    For unit vectors ``d = 2 - 2 c·y``, and the candidate is fixed given
    ``z_t``, so ``E[d_i | z_t] = 2 - 2 ĉ_i · E[ŷ | z_t] = 2 - 2 ĉ_i · mu``.

    candidates: (..., P, D), mean: (..., D) -> (..., P)
    """
    directions = F.normalize(candidates.float(), dim=-1)
    return 2.0 - 2.0 * torch.einsum("...pd,...d->...p", directions, mean.float())


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
    * win-probability logits ``scores`` (pi): which candidate is likely the
      closest to the real future.  Generation samples from these.

    Separately, the mean head maps ``z_t`` to ``mu ≈ E[ŷ | z_t]``, the mean
    normalized future.  It is an auxiliary instrument (uncertainty, a
    single-head baseline, a future head-growth signal), not the selector.
    Its input is detached, so it never changes the encoder or candidates.

    Configured by :class:`jepa.config.PredictorConfig` and usable on its own::

        predictor = PredictorModel(PredictorConfig.from_encoder(enc_config))
        candidates, scores, mean = predictor(latents, return_all_timesteps=False)
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

        # Win-probability logit of each candidate (pi).
        self.score_head = PerDimLinear(
            dims=config.prediction_count,
            in_size=config.hidden_size,
            out_size=1
        )

        # mu: predicted mean of the normalized future latent, from z_t.
        self.mean_head = nn.Sequential(
            nn.Linear(config.target_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.target_dim),
        )

        self.post_init()

        # Start mu at 0 ("no knowledge", norm 0) instead of a random vector.
        # After post_init, which would otherwise re-initialize the Linear.
        nn.init.zeros_(self.mean_head[-1].weight)
        nn.init.zeros_(self.mean_head[-1].bias)

    def forward(self, input, return_all_timesteps: bool = True):
        """Predict k steps into the future from each latent in ``input``.

        Args:
            input: (batch, seq, target_dim) encoder latents.
            return_all_timesteps: If True, predict from every position
                                  independently.  If False, predict only from
                                  the final latent (the others are ignored).
        Returns:
            :class:`PredictorOutput` ``(candidates, scores, mean)`` with shapes
            (batch, seq, P, D), (batch, seq, P), (batch, seq, D) when
            ``return_all_timesteps`` is True, or (batch, P, D), (batch, P),
            (batch, D) otherwise.
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
        scores = self.score_head(hidden_states).squeeze(-1)  # (N, prediction_count)
        # Detached: mu is a measuring instrument and must not shape z_t.
        mean = self.mean_head(z_t.detach())  # (N, target_dim)

        predictions = z_t.unsqueeze(1) + pred_deltas  # (N, prediction_count, target_dim)

        # Fix every candidate to the encoder-latent scale (per-dim RMS 1, i.e.
        # norm ~sqrt(target_dim)).  The loss compares directions only, so
        # without this a candidate's norm is free; unused slots drifted to norms
        # ~50x z_t, where the gradient through the loss's normalization
        # (~1/norm) vanishes and they could never recover.  No learnable gain:
        # it would reintroduce the free scale.  Runs in fp32 for precise
        # statistics, like the other norms.
        predictions = F.rms_norm(
            predictions.float(),
            (target_dim,),
            eps=self.config.rms_norm_eps,
        ).to(predictions.dtype)

        if return_all_timesteps:
            return PredictorOutput(
                predictions.view(batch_size, seq_len, prediction_count, target_dim),
                scores.view(batch_size, seq_len, prediction_count),
                mean.view(batch_size, seq_len, target_dim),
            )

        return PredictorOutput(predictions, scores, mean)
