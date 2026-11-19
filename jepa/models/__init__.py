"""Model components and the composed JEPA stack."""

from .blocks import (
    MLP,
    Attention,
    DecoderLayer,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    build_attention_bias,
    rotate_half,
)
from .decoder import DecoderModel
from .encoder import EncodeTransformer
from .jepa import JepaModel
from .losses import VISRegLoss
from .predictor import PredictorModel
from .stack import JepaDecodeModel, build_block_hints, select_waypoints

__all__ = [
    # Shared Transformer blocks
    "MLP",
    "Attention",
    "DecoderLayer",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_pos_emb",
    "build_attention_bias",
    "rotate_half",
    # Models
    "EncodeTransformer",
    "PredictorModel",
    "DecoderModel",
    "JepaModel",
    "JepaDecodeModel",
    # Loss / helpers
    "VISRegLoss",
    "build_block_hints",
    "select_waypoints",
]
