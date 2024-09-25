"""Softmax predictive NLL loss for multiclass classification."""

from torch import nn

from untangle.utils.predictive import PREDICTIVE_DICT


class SoftmaxPredictiveNLLLoss(nn.Module):
    """Softmax predictive NLL loss."""

    def __init__(self, predictive):
        super().__init__()

        if not predictive.startswith("softmax") or predictive.endswith("mc"):
            msg = "Invalid predictive provided"
            raise ValueError(msg)

        self._predictive = PREDICTIVE_DICT[predictive]
        self._loss = nn.CrossEntropyLoss

    def forward(self, logits, targets):
        preds = self._predictive(*logits, return_logits=True)
        loss = self._loss(preds, targets)

        return loss
