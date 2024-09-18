"""Regularized predictive NLL loss for multiclass classification."""

import torch.nn.functional as F
from torch import nn

from untangle.losses.normcdf_nll_loss import NormCDFNLLLoss
from untangle.losses.sigmoid_nll_loss import SigmoidNLLLoss
from untangle.utils.predictive import PREDICTIVE_DICT, normcdf


class UnnormalizedPredictiveNLLLoss(nn.Module):
    """Regularized predictive NLL loss."""

    def __init__(self, predictive):
        super().__init__()

        if not predictive.startswith("probit", "logit"):
            msg = "Invalid predictive provided"
            raise ValueError(msg)

        self._predictive = PREDICTIVE_DICT[predictive]
        self._activation = normcdf if predictive.startswith("probit") else F.sigmoid
        self._loss = (
            NormCDFNLLLoss() if predictive.startswith("probit") else SigmoidNLLLoss()
        )

    def forward(self, logits, targets):
        preds = self._predictive(logits, return_unnormalized=True)
        loss = self._loss(preds, targets, apply_activation=False)

        return loss
