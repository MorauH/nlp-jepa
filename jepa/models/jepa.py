"""JEPA encoder + predictor and the self-supervised objective."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel

from ..config import JepaConfig
from .encoder import EncodeTransformer
from .losses import VISRegLoss
from .predictor import PredictorModel


class JepaModel(PreTrainedModel):
    """JEPA language model: predicts future latent states from current ones.

    The model is configured by :class:`jepa.config.JepaConfig` and can be used
    without a decoder, either standalone or with a pretrained encoder::

        model = JepaModel(JepaConfig.from_encoder(enc_cfg, with_decoder=False))
        encoder = EncodeTransformer.from_pretrained("runs/jepa/encoder")
        model = JepaModel(config, encoder=encoder)
    """

    config_class = JepaConfig

    def __init__(
        self,
        config: JepaConfig,
        encoder: EncodeTransformer | None = None,
        predictor: PredictorModel | None = None,
    ):
        super().__init__(config)

        self.encoder = (
            encoder if encoder is not None else EncodeTransformer(config.encoder)
        )
        self.predictor = (
            predictor if predictor is not None else PredictorModel(config.predictor)
        )

        self.criterion = VISRegLoss(config.visreg)
        self.pred_horizon = config.pred_horizon

        self.post_init()

    def _init_weights(self, module):
        # Sub-modules initialize themselves (and may have been loaded from a
        # checkpoint); re-initializing them here would discard that.
        #
        # Exception: the loss's non-persistent temperature buffer is not in
        # the checkpoint, so with meta-device loading it comes back as
        # uninitialized memory.  transformers calls this hook for exactly such
        # modules, so restore the buffer's default value here.
        if isinstance(module, VISRegLoss):
            module.reset_annealing()

    def _encode(self, labels, attention_mask, right_padded: bool):
        """Encode tokens, dropping the padding mask when that is safe.

        Dropping the mask is only valid for right-aligned padding *and* a
        causal encoder: then no real query can see a pad key.
        """
        skip_mask = right_padded and self.encoder.config.is_causal
        return self.encoder(
            labels, attention_mask=None if skip_mask else attention_mask
        )

    def forward(
        self,
        labels,
        attention_mask=None,
        training: bool = True,
        right_padded: bool = False,
    ):
        """Run the JEPA objective.

        Args:
            labels: (batch, seq) token ids.
            attention_mask: optional (batch, seq), 1 = real token.
            training: If True, return the VISReg loss plus every latent/prediction.
                      If False, predict only from the final position.
            right_padded: If True, padding is right-aligned.  A causal encoder
                      can then ignore the padding mask in attention (pads are
                      never attended by real queries), which enables the fused
                      causal SDPA kernel.
        Returns:
            dict with:
              training: ``loss``, ``latents``, ``preds``, ``scores``,
                        ``z_future``, ``valid``, ``stats``
              inference: ``latents``, ``z_t``, ``preds``, ``scores``,
                         ``z_future``
        """
        enc_hidden = self._encode(labels, attention_mask, right_padded)

        k = self.pred_horizon
        seq_len = enc_hidden.shape[1]

        if training:
            if seq_len <= k:
                raise ValueError(
                    f"sequence length ({seq_len}) must exceed pred_horizon ({k})"
                )

            X = enc_hidden[:, :-k, :]
            Y = enc_hidden[:, k:, :].detach()

            if attention_mask is not None:
                valid = attention_mask[:, :-k].bool() & attention_mask[:, k:].bool()
            else:
                valid = None

            # From each hidden[i, t, :] predict hidden[i, t+k, :]
            preds, scores = self.predictor(X, return_all_timesteps=True)
            loss, stats = self.criterion(
                preds,
                Y,
                mask=valid,
                scores=scores,
                reg_source=enc_hidden,
                reg_mask=attention_mask,
                return_stats=True,
            )

            # Commit to the closest candidate for the (detached) decoder
            # target.  At inference the score head makes this choice instead.
            win = stats["win"]  # (B, T)
            z_future = torch.gather(
                preds,
                2,
                win[..., None, None].expand(-1, -1, 1, preds.shape[-1]),
            ).squeeze(2)

            return {
                "loss": loss,
                "latents": enc_hidden,
                "preds": preds,
                "scores": scores,
                "z_future": z_future,
                "valid": valid,
                "stats": stats,
            }

        assert attention_mask is None or attention_mask[:, -1].all(), (
            "Inference expects no padding at the end of the sequence"
        )

        preds, scores = self.predictor(enc_hidden, return_all_timesteps=False)

        # Greedy commitment by default.  Callers that want stochastic
        # directions can sample from ``scores`` themselves.
        best = scores.argmax(dim=-1)  # (B,)
        z_future = preds[
            torch.arange(preds.shape[0], device=preds.device), best
        ]

        return {
            "latents": enc_hidden,
            "z_t": enc_hidden[:, -1, :],
            "preds": preds,
            "scores": scores,
            "z_future": z_future,
        }
