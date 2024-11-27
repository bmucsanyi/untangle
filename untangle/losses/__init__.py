"""Implementations of supported loss functions."""

from .bma_cross_entropy_loss import BMACrossEntropyLoss
from .correctness_prediction_loss import CorrectnessPredictionLoss
from .duq_loss import DUQLoss
from .edl_loss import EDLLoss
from .fbar_cross_entropy_loss import FBarCrossEntropyLoss
from .loss_prediction_loss import LossPredictionLoss
from .regularized_uce_loss import RegularizedUCELoss

__all__ = [
    "BMACrossEntropyLoss",
    "CorrectnessPredictionLoss",
    "DUQLoss",
    "EDLLoss",
    "FBarCrossEntropyLoss",
    "LossPredictionLoss",
    "RegularizedUCELoss",
]
