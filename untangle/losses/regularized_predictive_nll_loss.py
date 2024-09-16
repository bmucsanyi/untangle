"""Regularized predictive NLL loss for multiclass classification."""

import torch
import torch.nn.functional as F
from torch import nn

from untangle.utils.predictive import PREDICTIVE_DICT, normcdf


class RegularizedPredictiveNLLLoss(nn.Module):
    """Regularized predictive NLL loss."""

    def __init__(self, predictive, regularization_factor):
        super().__init__()

        if not predictive.startswith("probit", "logit"):
            msg = "Invalid predictive provided"
            raise ValueError(msg)

        self._predictive = PREDICTIVE_DICT[predictive]
        self._activation = normcdf if predictive.startswith("probit") else F.sigmoid
        self._regularization_factor = regularization_factor

    def forward(self, logits, targets):
        preds = self._predictive(logits)
        loss = (
            -preds[torch.arange(0, preds.shape[-1]), targets]
            .log()
            .clamp(torch.finfo(preds.dtype).min)
        ).mean() + self._regularization_factor * self._activation(logits).sum(
            dim=-1
        ).sub(1).square().mean()

        return loss
