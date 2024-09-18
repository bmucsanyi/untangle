"""Sigmoid + NLL loss for multiclass classification."""

import torch
import torch.nn.functional as F
from torch import nn


class SigmoidNLLLoss(nn.Module):
    """Sigmoid + NLL loss."""

    def __init__(self):
        super().__init__()

        self._softplus = nn.Softplus()

    def forward(self, logits, targets, *, apply_activation=True):
        targets = F.one_hot(targets, num_classes=logits.shape[-1])

        if apply_activation:
            # Compute sigmoid BCE loss
            loss = self._softplus(logits) - targets * logits
        else:
            preds = logits

            # Compute loss
            loss = torch.where(targets == 1, -torch.log(preds), -torch.log(1 - preds))

        # Sum along the class dimension
        loss = loss.sum(dim=1)

        # Mean over the batch
        return loss.mean()
