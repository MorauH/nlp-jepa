"""JEPA language model: encoder + predictor + optional decoder.

Component models can be used on their own::

    from jepa import EncoderConfig, EncodeTransformer
    encoder = EncodeTransformer(EncoderConfig(vocab_size=8192))

    from jepa import PredictorConfig, PredictorModel
    predictor = PredictorModel(PredictorConfig.from_encoder(encoder.config))

    from jepa import DecoderConfig, DecoderModel
    decoder = DecoderModel(DecoderConfig.from_encoder(encoder.config))

The full stack takes a single composed config::

    from jepa import JepaConfig, JepaDecodeModel
    config = JepaConfig.from_encoder(encoder.config)
    model = JepaDecodeModel(config)
"""

from .config import (
    CommitmentLossConfig,
    DecoderConfig,
    EncoderConfig,
    JepaConfig,
    PredictorConfig,
    TransformerConfig,
    VISRegConfig,
)
from .models.decoder import DecoderModel
from .models.encoder import EncodeTransformer
from .models.jepa import JepaModel
from .models.losses import CommitmentLoss, LossOutput, VISReg
from .models.predictor import PredictorModel
from .models.stack import JepaDecodeModel

__all__ = [
    # Configs
    "TransformerConfig",
    "EncoderConfig",
    "PredictorConfig",
    "DecoderConfig",
    "VISRegConfig",
    "CommitmentLossConfig",
    "JepaConfig",
    # Models
    "EncodeTransformer",
    "PredictorModel",
    "DecoderModel",
    "JepaModel",
    "JepaDecodeModel",
    # Loss
    "VISReg",
    "CommitmentLoss",
    "LossOutput",
]
