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
configs (plus the VISReg objective) and links the dimensions that must agree.

Dimension ownership
-------------------
The encoder owns the embedding width and the vocabulary.  When a
``JepaConfig`` is built, the dependent fields are either filled in or checked::

    predictor.target_dim   == encoder.hidden_size
    predictor.pad_token_id == encoder.pad_token_id
    decoder.hidden_size    == encoder.hidden_size
    decoder.vocab_size     == encoder.vocab_size
    decoder.pad_token_id   == encoder.pad_token_id

Passing an explicitly conflicting value raises ``ValueError`` instead of
silently building a model whose pieces do not fit together.

Standalone components
---------------------
``PredictorConfig.from_encoder(...)`` and ``DecoderConfig.from_encoder(...)``
derive a valid config from an encoder config, while still allowing overrides::

    pred_cfg = PredictorConfig.from_encoder(enc_cfg, num_hidden_layers=2)
    dec_cfg = DecoderConfig.from_encoder(enc_cfg, num_hidden_layers=2, sliding_window=96)
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
        hidden_size: int = 256,
        intermediate_size: int | None = None,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 4,
        rope_theta: float = 500_000.0,
        hidden_act: str = "silu",
        max_position_embeddings: int = 256,
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
    predictor proposes from one input latent.  Each candidate gets a predicted
    distance to the real future (its value), so the best candidate can be
    selected, or one sampled, later.

    The predictor only sees the latest latent ``z_t``: the causal encoder
    already summarizes the prefix there.  The former ``trunk_layers`` and
    ``context_window`` options (extra access to past latents) were removed;
    loading a config that enabled them raises ``ValueError``.

    ``normalize_candidates`` RMS-normalizes every candidate to the encoder
    latent scale.  The multi-hypothesis loss compares directions only, so the
    candidate norm is otherwise unconstrained, and slots that drift to large
    norms stop receiving gradient.
    """

    model_type = "jepa_predictor"

    def __init__(
        self,
        target_dim: int | None = None,
        prediction_count: int = 16,
        normalize_candidates: bool = False,
        pad_token_id: int | None = None,
        is_causal: bool = False,
        **kwargs,
    ):
        # Removed options.  Old configs store them as 0 when unused; drop
        # those so they are not re-saved.  Anything else is a checkpoint whose
        # predictor read past latents and cannot be loaded into this one.
        for removed in ("trunk_layers", "context_window"):
            if kwargs.pop(removed, 0):
                raise ValueError(
                    f"PredictorConfig.{removed} was removed: the predictor now "
                    "only reads z_t. This checkpoint's predictor cannot be loaded."
                )
        super().__init__(is_causal=is_causal, **kwargs)
        self.target_dim = target_dim
        # Number of candidate predictions created from one latent vector
        # before the predictor transformer runs.  These are expansion slots,
        # not time steps, which is why the predictor defaults to non-causal
        # attention.
        self.prediction_count = prediction_count
        # RMS-normalize each candidate (z_t + delta) to the encoder-latent
        # scale.  False keeps the legacy, unbounded candidate norm.
        self.normalize_candidates = normalize_candidates
        self.pad_token_id = pad_token_id

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

        Any keyword passed in ``overrides`` replaces the derived value.
        """
        encoder = JepaConfig._coerce(EncoderConfig, encoder, required=True)
        values = dict(
            target_dim=encoder.hidden_size,
            hidden_size=encoder.hidden_size,
            intermediate_size=encoder.intermediate_size,
            num_attention_heads=encoder.num_attention_heads,
            num_hidden_layers=encoder.num_hidden_layers,
            rope_theta=encoder.rope_theta,
            hidden_act=encoder.hidden_act,
            max_position_embeddings=encoder.max_position_embeddings,
            rms_norm_eps=encoder.rms_norm_eps,
            is_causal=False,
            pad_token_id=encoder.pad_token_id,
        )
        values.update(overrides)
        return cls(**values)


class DecoderConfig(TransformerConfig):
    """Configuration for the autoregressive :class:`jepa.decoder.DecoderModel`.

    The decoder is a causal token transformer conditioned on latent hints.  It
    sees the last ``sliding_window`` tokens and generates tokens one at a time.

    ``tie_embeddings`` shares the token embedding table with the output
    projection.  ``hint_every_layer`` re-injects the latent hint into the
    residual stream before every transformer layer instead of only adding it
    to the token embeddings.
    """

    model_type = "jepa_decoder"

    def __init__(
        self,
        vocab_size: int | None = None,
        pad_token_id: int | None = None,
        sliding_window: int = 96,
        tie_embeddings: bool = False,
        hint_every_layer: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("is_causal", True)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        # Maximum number of past tokens visible to each decoder position.
        # Grammar is local; the latent hints carry the longer-range semantics.
        self.sliding_window = sliding_window
        # Share ``embedding.weight`` with ``lm_head.weight`` (one table, and
        # both gradient paths accumulate into it).  Off by default so old
        # checkpoints keep their separate tables.
        self.tie_embeddings = tie_embeddings
        # ``PreTrainedModel``'s tying machinery (``post_init``,
        # ``from_pretrained``, ``save_pretrained``) reads
        # ``config.tie_word_embeddings``, so mirror our flag into it.  The flag
        # above stays the source of truth; this only keeps HF in sync.
        self.tie_word_embeddings = tie_embeddings
        # Add a zero-initialized per-layer projection of the hint before each
        # layer (see DecoderModel._run).
        self.hint_every_layer = hint_every_layer

    def validate(self) -> "DecoderConfig":
        super().validate()
        if self.vocab_size is None or self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if self.pad_token_id is None:
            raise ValueError("pad_token_id must be set")
        if self.sliding_window is None or self.sliding_window < 1:
            raise ValueError("sliding_window must be at least 1")
        return self

    @classmethod
    def from_encoder(cls, encoder: EncoderConfig, **overrides) -> "DecoderConfig":
        """Build a decoder config that consumes an encoder's latents."""
        encoder = JepaConfig._coerce(EncoderConfig, encoder, required=True)
        values = dict(
            hidden_size=encoder.hidden_size,
            intermediate_size=encoder.intermediate_size,
            num_attention_heads=encoder.num_attention_heads,
            num_hidden_layers=2,
            rope_theta=encoder.rope_theta,
            hidden_act=encoder.hidden_act,
            max_position_embeddings=encoder.max_position_embeddings,
            rms_norm_eps=encoder.rms_norm_eps,
            is_causal=True,
            vocab_size=encoder.vocab_size,
            pad_token_id=encoder.pad_token_id,
        )
        values.update(overrides)
        return cls(**values)


class VISRegConfig(PretrainedConfig):
    """Weights of the VISReg + multi-hypothesis objective.

    The single-prediction invariance term is replaced by a relaxed
    winner-takes-all move loss plus a regression that trains the predictor's
    value head to predict each candidate's distance to the target.  The
    scale/shape/center regularizers act on the online encoder latents.
    """

    model_type = "jepa_visreg"

    def __init__(
        self,
        num_slices: int = 256,
        scale_w: float = 1.0,
        shape_w: float = 1.0,
        center_w: float = 1.0,
        move_w: float = 1.0,
        score_w: float = 0.1,
        diversity_w: float = 0.01,
        diversity_margin: float = 0.0,
        mh_temperature: float = 0.1,
        mh_temperature_start: float | None = None,
        mh_anneal_fraction: float = 0.3,
        mh_eps: float = 0.05,
        normalize_distance: bool = True,
        eps: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_slices = num_slices
        self.scale_w = scale_w
        self.shape_w = shape_w
        self.center_w = center_w
        self.move_w = move_w
        # Weight of the value-head distance regression (MSE).
        self.score_w = score_w
        # Weight of the pairwise candidate-diversity penalty.
        self.diversity_w = diversity_w
        # Only candidate pairs with cosine above this margin are penalized
        # (near-duplicates).  0.0 = legacy: any positive cosine is penalized,
        # which lets unused slots "hide" by pointing orthogonal to everything.
        self.diversity_margin = diversity_margin
        # Soft-assignment temperature, relative to the mean candidate distance.
        # With annealing this is the final value.
        self.mh_temperature = mh_temperature
        # Annealed winner-takes-all: when set, the temperature multiplier
        # starts here and decays geometrically to ``mh_temperature`` over the
        # first ``mh_anneal_fraction`` of training (driven by the trainer via
        # ``VISRegLoss.set_anneal_progress``).  A high start shares every
        # target across all slots so each slot finds a region before they
        # specialize.  None = no annealing (legacy constant temperature).
        self.mh_temperature_start = mh_temperature_start
        self.mh_anneal_fraction = mh_anneal_fraction
        # Uniform smoothing of the soft assignment (relaxed WTA).
        self.mh_eps = mh_eps
        # Compare unit-normalized latents instead of raw L2 distance.
        self.normalize_distance = normalize_distance
        self.eps = eps

    def validate(self) -> "VISRegConfig":
        if self.num_slices is None or self.num_slices < 1:
            raise ValueError("num_slices must be at least 1")
        if self.eps is None or self.eps <= 0:
            raise ValueError("eps must be positive")
        for name in (
            "scale_w",
            "shape_w",
            "center_w",
            "move_w",
            "score_w",
            "diversity_w",
        ):
            value = getattr(self, name)
            if value is None or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.mh_temperature is None or self.mh_temperature <= 0:
            raise ValueError("mh_temperature must be positive")
        if self.mh_eps is None or not 0.0 <= self.mh_eps <= 1.0:
            raise ValueError("mh_eps must be in [0, 1]")
        if self.mh_temperature_start is not None and self.mh_temperature_start <= 0:
            raise ValueError("mh_temperature_start must be positive (or None)")
        if self.mh_anneal_fraction is None or not 0.0 < self.mh_anneal_fraction <= 1.0:
            raise ValueError("mh_anneal_fraction must be in (0, 1]")
        if self.diversity_margin is None or not -1.0 <= self.diversity_margin < 1.0:
            raise ValueError("diversity_margin must be in [-1, 1)")
        return self


class JepaConfig(PretrainedConfig):
    """Full configuration for the JEPA stack.

    ``decoder`` is optional: omit it (or pass ``with_decoder=False`` to
    :meth:`from_encoder`) when training the self-supervised JEPA objective on
    its own.
    """

    model_type = "jepa"

    def __init__(
        self,
        encoder: EncoderConfig | dict | None = None,
        predictor: PredictorConfig | dict | None = None,
        decoder: DecoderConfig | dict | None = None,
        visreg: VISRegConfig | dict | None = None,
        pred_horizon: int = 5,
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
        self.pred_horizon = pred_horizon

        # Fill in derived predictor values before validating it.
        if self.predictor is None:
            self.predictor = PredictorConfig.from_encoder(self.encoder)
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
        with_decoder: bool = True,
        pred_horizon: int = 5,
        visreg: VISRegConfig | None = None,
        **kwargs,
    ) -> "JepaConfig":
        """Derive a full JEPA config from an encoder config.

        Missing predictor/decoder configs are built from the encoder.  Set
        ``with_decoder=False`` to train the JEPA objective without a decoder.
        """
        encoder = cls._coerce(EncoderConfig, encoder, required=True)
        if predictor is None:
            predictor = PredictorConfig.from_encoder(encoder)
        if decoder is None and with_decoder:
            decoder = DecoderConfig.from_encoder(encoder)
        return cls(
            encoder=encoder,
            predictor=predictor,
            decoder=decoder,
            visreg=visreg,
            pred_horizon=pred_horizon,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def link(self) -> "JepaConfig":
        """Fill in or verify every dimension shared between components."""
        self.encoder.validate()

        self._set_or_check(
            self.predictor,
            "target_dim",
            self.encoder.hidden_size,
            "predictor.target_dim",
        )
        self._set_or_check(
            self.predictor,
            "pad_token_id",
            self.encoder.pad_token_id,
            "predictor.pad_token_id",
        )
        self.predictor.validate()

        if self.decoder is not None:
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
