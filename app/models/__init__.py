"""Production model components.

Keep model imports lazy so data-only utilities can run without importing
PyTorch. The production predictor still loads each model module explicitly.
"""

__all__ = ["StationAttentionHF", "Normalizer", "time_feature_frame"]


def __getattr__(name: str):
    if name == "StationAttentionHF":
        from .station_attention import StationAttentionHF

        return StationAttentionHF
    if name == "Normalizer":
        from .normalizer import Normalizer

        return Normalizer
    if name == "time_feature_frame":
        from .utils import time_feature_frame

        return time_feature_frame
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
