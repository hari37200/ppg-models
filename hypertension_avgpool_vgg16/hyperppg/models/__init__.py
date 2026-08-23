"""Model zoo: the paper's four image CNNs plus the improved 1-D hybrids."""

from hyperppg.models.paper import PAPER_MODELS, build_model, avgpool_vgg16
from hyperppg.models.hybrid import (
    PPGEncoder1D,
    PPGHybridClassifier,
    MaskedPPGAutoencoder,
)

__all__ = [
    "PAPER_MODELS",
    "build_model",
    "avgpool_vgg16",
    "PPGEncoder1D",
    "PPGHybridClassifier",
    "MaskedPPGAutoencoder",
]
