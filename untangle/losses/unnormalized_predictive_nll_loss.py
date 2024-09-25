"""Unnormalized predictive NLL loss for multiclass classification."""

from torch import nn

from untangle.losses.normcdf_nll_loss import NormCDFNLLLoss
from untangle.losses.sigmoid_nll_loss import SigmoidNLLLoss
from untangle.utils.predictive import PREDICTIVE_DICT


class UnnormalizedPredictiveNLLLoss(nn.Module):
    """Unnormalized predictive NLL loss."""

    def __init__(self, predictive):
        super().__init__()

        if not predictive.startswith(("probit", "logit")) or predictive.endswith("mc"):
            msg = "Invalid predictive provided"
            raise ValueError(msg)

        self._predictive = PREDICTIVE_DICT[predictive]
        self._loss = (
            NormCDFNLLLoss() if predictive.startswith("probit") else SigmoidNLLLoss()
        )

    def forward(self, logits, targets):
        preds = self._predictive(*logits, return_logits=True)
        loss = self._loss(preds, targets)

        return loss
