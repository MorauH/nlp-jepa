"""Centralized configuration objects for the JEPA stack.

Every component is independently runnable
-----------------------------------------
Each trainable component owns a self-contained
:class:`~transformers.PretrainedConfig` subclass, so it can be constructed,
saved and reloaded without the rest of the stack::

    EncoderConfig   -> EncodeTransformer   tokens    -> latent sequence
    PredictorConfig -> PredictorModel      z_t       -> z_{t+k}
    DecoderConfig   -> DecoderModel        token context, latent hints -> next-token logits

The full model is described by :class:`JepaConfig`, which composes those
configs (plus the VISReg regularizer and the commitment loss) and links the
dimensions that must agree.

Dimension ownership
-------------------
The encoder owns the embedding width and the vocabulary.  When a
``JepaConfig`` is built, the dependent fields are either filled in or checked::

    predictor.target_dim == encoder.hidden_size
    decoder.hidden_size  == encoder.hidden_size
    decoder.vocab_size   == encoder.vocab_size
    decoder.pad_token_id == encoder.pad_token_id

Passing an explicitly conflicting value raises ``ValueError`` instead of
silently building a model whose pieces do not fit together.  The defaults in
this module are the actual experiment; the training scripts do not override
them.

Standalone components
---------------------
``PredictorConfig.from_encoder(...)`` and ``DecoderConfig.from_encoder(...)``
link the dimensions that must agree with an encoder config, while leaving every
other default in place::

    pred_cfg = PredictorConfig.from_encoder(enc_cfg)
    dec_cfg = DecoderConfig.from_encoder(enc_cfg)
"""

from __future__ import annotations

from transformers import PretrainedConfig

# Keys written by ``PretrainedConfig.to_dict`` that should not be fed back into
# a nested config constructor.
_SERIALIZATION_METADATA = frozenset(
    {"_name_or_path", "architectures", "model_type", "transformers_version"}
)


class TransformerConfig(PretrainedConfig):
    """Hyperparameters shared by the RoPE Transformer blocks.

    This is the common base for the encoder and the predictor.  It is not
    instantiated on its own.
    """

    model_type = "jepa_transformer"

    def __init__(
        self,
        hidden_size: int = 448,
        intermediate_size: int | None = None,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        rope_theta: float = 500_000.0,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1024,
        rms_norm_eps: float = 1e-6,
        is_causal: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        # SwiGLU convention: the hidden MLP width is twice the model width.
        self.intermediate_size = (
            2 * hidden_size if intermediate_size is None else intermediate_size
        )
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.is_causal = is_causal

    def validate(self) -> "TransformerConfig":
        """Fail fast on values that ``DecoderLayer`` cannot consume."""
        if self.hidden_size is None or self.hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        if self.num_attention_heads is None or self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be a positive integer")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_hidden_layers is None or self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")
        if self.hidden_act != "silu":
            raise ValueError(
                f"hidden_act={self.hidden_act!r} is unsupported; the MLP is SwiGLU (silu)"
            )
        return self


class EncoderConfig(TransformerConfig):
    """Configuration for :class:`jepa.encoder.EncodeTransformer`."""

    model_type = "jepa_encoder"

    def __init__(
        self,
        vocab_size: int = 512,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def validate(self) -> "EncoderConfig":
        super().validate()
        if self.vocab_size is None or self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if self.pad_token_id is None:
            raise ValueError("pad_token_id must be set")
        return self


class PredictorConfig(TransformerConfig):
    """Configuration for :class:`jepa.predictor.PredictorModel`.

    ``target_dim`` is the width of the encoder latent that the predictor maps
    forward in time.  Under :class:`JepaConfig` it is always linked to the
    encoder's ``hidden_size``.

    ``prediction_count`` is the number of candidate future latents the
    predictor proposes from one input latent.  Each candidate gets a
    win-probability logit, so one future can be sampled later (like sampling a
    token from logits).

    The predictor only sees the latest latent ``z_t``: the causal encoder
    already summarizes the prefix there.  It consumes no tokens, so it has no
    ``pad_token_id``, and it applies no RoPE.

    Every candidate is RMS-normalized to the encoder latent scale.  The
    multi-hypothesis loss compares normalized latents, so the candidate norm is
    otherwise unconstrained, and slots that drift to large norms stop receiving
    gradient.
    """

    model_type = "jepa_predictor"

    def __init__(
        self,
        target_dim: int | None = None,
        prediction_count: int = 8,
        num_hidden_layers: int = 2,
        is_causal: bool = False,
        **kwargs,
    ):
        super().__init__(
            num_hidden_layers=num_hidden_layers, is_causal=is_causal, **kwargs
        )
        self.target_dim = target_dim
        # Number of candidate predictions created from one latent vector
        # before the predictor transformer runs.  These are expansion slots,
        # not time steps, which is why the predictor defaults to non-causal
        # attention.
        self.prediction_count = prediction_count

    def validate(self) -> "PredictorConfig":
        super().validate()
        if self.target_dim is None or self.target_dim <= 0:
            raise ValueError("target_dim must be set (use from_encoder(...))")
        if self.prediction_count is None or self.prediction_count < 1:
            raise ValueError("prediction_count must be at least 1")
        return self

    @classmethod
    def from_encoder(
        cls, encoder: EncoderConfig, **overrides
    ) -> "PredictorConfig":
        """Build a predictor config that fits an encoder config.

        Only the width is linked (``target_dim`` and the transformer's
        ``hidden_size``); everything else keeps this class's defaults.  Any
        keyword passed in ``overrides`` replaces the linked value.
        """
        encoder = JepaConfig._coerce(EncoderConfig, encoder, required=True)
        values = dict(
            target_dim=encoder.hidden_size,
            hidden_size=encoder.hidden_size,
        )
        values.update(overrides)
        return cls(**values)


class DecoderConfig(TransformerConfig):
    """Configuration for the autoregressive :class:`jepa.decoder.DecoderModel`.

    The decoder is a causal token transformer conditioned on latent hints.  It
    sees the last ``sliding_window`` tokens and generates tokens one at a time.

    The token embedding table is shared with the output projection
    (``tie_word_embeddings``), and the latent hint is re-injected into the
    residual stream before every transformer layer.

    ``train_policy`` is the policy the decoder's hint commits by:
    ``"hindsight"`` uses the candidate closest to the real future,
    ``"pi_argmax"`` the most likely candidate under pi (the inference
    condition).  Evaluation always uses ``"pi_argmax"``.  The names are
    registered in :data:`jepa.models.commitment.POLICIES`.

    ``loss_chunk_size`` bounds the ``(batch, positions, vocab)`` logits tensor
    held at once by the token loss; ``None`` computes it in one pass.
    """

    model_type = "jepa_decoder"

    # The subset of ``commitment.POLICIES`` that makes sense for training the
    # decoder (sampling policies would make the target noisy).
    TRAIN_POLICIES = ("hindsight", "pi_argmax")

    def __init__(
        self,
        vocab_size: int | None = None,
        pad_token_id: int | None = None,
        sliding_window: int = 32,
        num_hidden_layers: int = 4,
        train_policy: str = "hindsight",
        loss_chunk_size: int | None = 256,
        **kwargs,
    ):
        kwargs.setdefault("is_causal", True)
        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(num_hidden_layers=num_hidden_layers, **kwargs)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        # Maximum number of past tokens visible to each decoder position.
        # Grammar is local; the latent hints carry the longer-range semantics.
        self.sliding_window = sliding_window
        self.train_policy = train_policy
        self.loss_chunk_size = loss_chunk_size

    def validate(self) -> "DecoderConfig":
        super().validate()
        if self.vocab_size is None or self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if self.pad_token_id is None:
            raise ValueError("pad_token_id must be set")
        if self.sliding_window is None or self.sliding_window < 1:
            raise ValueError("sliding_window must be at least 1")
        if self.train_policy not in self.TRAIN_POLICIES:
            raise ValueError(
                f"train_policy must be one of {self.TRAIN_POLICIES}; "
                f"got {self.train_policy!r}"
            )
        if self.loss_chunk_size is not None and self.loss_chunk_size < 1:
            raise ValueError("loss_chunk_size must be None or at least 1")
        return self

    @classmethod
    def from_encoder(cls, encoder: EncoderConfig, **overrides) -> "DecoderConfig":
        """Build a decoder config that consumes an encoder's latents.

        Only the token/width dimensions are linked; depth, window and every
        other shape keep this class's defaults.
        """
        encoder = JepaConfig._coerce(EncoderConfig, encoder, required=True)
        values = dict(
            hidden_size=encoder.hidden_size,
            vocab_size=encoder.vocab_size,
            pad_token_id=encoder.pad_token_id,
        )
        values.update(overrides)
        return cls(**values)


class VISRegConfig(PretrainedConfig):
    """Weights of the VISReg anti-collapse regularizer.

    The scale/shape/center terms act on the online encoder latents, so they
    shape the representation the predictor consumes rather than the predictor
    output.
    """

    model_type = "jepa_visreg"

    def __init__(
        self,
        num_slices: int = 256,
        scale_w: float = 1.0,
        shape_w: float = 1.0,
        center_w: float = 1.0,
        eps: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_slices = num_slices
        self.scale_w = scale_w
        self.shape_w = shape_w
        self.center_w = center_w
        self.eps = eps

    def validate(self) -> "VISRegConfig":
        if self.num_slices is None or self.num_slices < 1:
            raise ValueError("num_slices must be at least 1")
        if self.eps is None or self.eps <= 0:
            raise ValueError("eps must be positive")
        for name in ("scale_w", "shape_w", "center_w"):
            value = getattr(self, name)
            if value is None or value < 0:
                raise ValueError(f"{name} must be non-negative")
        return self


class CommitmentLossConfig(PretrainedConfig):
    """Weights and annealing of the commitment loss.

    The objective trains the predictor's proposal: a relaxed winner-takes-all
    move loss pulls candidates toward their targets, a cross-entropy trains
    the win-probability distribution pi, a regression trains the mean-future
    head mu, and an optional penalty keeps candidates apart.
    """

    model_type = "jepa_commitment"

    def __init__(
        self,
        move_w: float = 1.0,
        pi_w: float = 1.0,
        mu_w: float = 1.0,
        diversity_w: float = 0.5,
        diversity_margin: float = 0.5,
        temperature: float = 0.1,
        temperature_start: float = 1.0,
        anneal_fraction: float = 0.3,
        smoothing: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Weight of the relaxed winner-takes-all move loss.
        self.move_w = move_w
        # Weight of the pi (win probability) cross-entropy.
        self.pi_w = pi_w
        # Weight of the mu (mean future) regression.  mu's input is detached,
        # so this only scales the mu head's own learning rate.
        self.mu_w = mu_w
        # Weight of the pairwise candidate-diversity penalty.
        self.diversity_w = diversity_w
        # Only candidate pairs with cosine above this margin are penalized
        # (near-duplicates).
        self.diversity_margin = diversity_margin
        # Soft-assignment temperature, relative to the mean candidate distance.
        # With annealing this is the final value.
        self.temperature = temperature
        # Annealed winner-takes-all: the temperature multiplier starts here and
        # decays geometrically to ``temperature`` over the first
        # ``anneal_fraction`` of training (driven by the trainer through the
        # ``progress`` argument).  A high start shares every target across all
        # slots so each slot finds a region before they specialize.
        self.temperature_start = temperature_start
        self.anneal_fraction = anneal_fraction
        # Uniform smoothing of the soft assignment (relaxed WTA).
        self.smoothing = smoothing

    def validate(self) -> "CommitmentLossConfig":
        for name in ("move_w", "pi_w", "mu_w", "diversity_w"):
            value = getattr(self, name)
            if value is None or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.temperature is None or self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.smoothing is None or not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("smoothing must be in [0, 1]")
        if self.temperature_start is None or self.temperature_start <= 0:
            raise ValueError("temperature_start must be positive")
        if self.anneal_fraction is None or not 0.0 < self.anneal_fraction <= 1.0:
            raise ValueError("anneal_fraction must be in (0, 1]")
        if self.diversity_margin is None or not -1.0 <= self.diversity_margin < 1.0:
            raise ValueError("diversity_margin must be in [-1, 1)")
        return self


class JepaConfig(PretrainedConfig):
    """Full configuration for the JEPA stack.

    The stack always has a decoder.  A missing one is built from the encoder,
    so every checkpoint can be generated from and trained in either stage.
    """

    model_type = "jepa"

    def __init__(
        self,
        encoder: EncoderConfig | dict | None = None,
        predictor: PredictorConfig | dict | None = None,
        decoder: DecoderConfig | dict | None = None,
        visreg: VISRegConfig | dict | None = None,
        commitment: CommitmentLossConfig | dict | None = None,
        pred_horizon: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Composite config: always serialize the full nested structure rather
        # than diffing against a default instance.
        self.is_composition = True

        if encoder is None:
            encoder = EncoderConfig()
        self.encoder = self._coerce(EncoderConfig, encoder, required=True)
        self.predictor = self._coerce(PredictorConfig, predictor)
        self.decoder = self._coerce(DecoderConfig, decoder)
        self.visreg = self._coerce(VISRegConfig, visreg)
        if self.visreg is None:
            self.visreg = VISRegConfig()
        self.commitment = self._coerce(CommitmentLossConfig, commitment)
        if self.commitment is None:
            self.commitment = CommitmentLossConfig()
        self.pred_horizon = pred_horizon

        # Fill in derived component configs before validating them.
        if self.predictor is None:
            self.predictor = PredictorConfig.from_encoder(self.encoder)
        if self.decoder is None:
            self.decoder = DecoderConfig.from_encoder(self.encoder)
        self.link()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce(config_cls, value, *, required: bool = False):
        """Convert a config instance / serialized dict into ``config_cls``."""
        if value is None:
            if required:
                raise ValueError(
                    f"JepaConfig requires an {config_cls.__name__}; "
                    "nothing was provided."
                )
            return None
        if isinstance(value, config_cls):
            return value
        if isinstance(value, PretrainedConfig):
            raise TypeError(
                f"Expected {config_cls.__name__}; got {type(value).__name__}"
            )
        if isinstance(value, dict):
            value = {
                key: item
                for key, item in value.items()
                if key not in _SERIALIZATION_METADATA
            }
            return config_cls(**value)
        raise TypeError(
            f"Expected {config_cls.__name__}, a dict, or None; "
            f"got {type(value).__name__}"
        )

    @classmethod
    def from_encoder(
        cls,
        encoder: EncoderConfig,
        *,
        predictor: PredictorConfig | None = None,
        decoder: DecoderConfig | None = None,
        pred_horizon: int = 20,
        visreg: VISRegConfig | None = None,
        commitment: CommitmentLossConfig | None = None,
        **kwargs,
    ) -> "JepaConfig":
        """Derive a full JEPA config from an encoder config.

        Missing predictor/decoder configs are built from the encoder.
        """
        encoder = cls._coerce(EncoderConfig, encoder, required=True)
        if predictor is None:
            predictor = PredictorConfig.from_encoder(encoder)
        if decoder is None:
            decoder = DecoderConfig.from_encoder(encoder)
        return cls(
            encoder=encoder,
            predictor=predictor,
            decoder=decoder,
            visreg=visreg,
            commitment=commitment,
            pred_horizon=pred_horizon,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def link(self) -> "JepaConfig":
        """Fill in or check every dimension shared between components."""
        self.encoder.validate()

        self._set_or_check(
            self.predictor,
            "target_dim",
            self.encoder.hidden_size,
            "predictor.target_dim",
        )
        self.predictor.validate()

        self._set_or_check(
            self.decoder,
            "hidden_size",
            self.encoder.hidden_size,
            "decoder.hidden_size",
        )
        self._set_or_check(
            self.decoder,
            "vocab_size",
            self.encoder.vocab_size,
            "decoder.vocab_size",
        )
        self._set_or_check(
            self.decoder,
            "pad_token_id",
            self.encoder.pad_token_id,
            "decoder.pad_token_id",
        )
        self.decoder.validate()

        if not isinstance(self.pred_horizon, int) or self.pred_horizon < 1:
            raise ValueError(
                f"pred_horizon must be a positive integer; got {self.pred_horizon!r}"
            )

        self.visreg.validate()
        self.commitment.validate()
        return self

    @staticmethod
    def _set_or_check(config, attribute: str, expected, path: str) -> None:
        actual = getattr(config, attribute, None)
        if actual is None:
            setattr(config, attribute, expected)
        elif actual != expected:
            raise ValueError(
                f"{path}={actual} is inconsistent with the encoder "
                f"(expected {expected}). Remove the override or make the "
                "encoder match."
            )
