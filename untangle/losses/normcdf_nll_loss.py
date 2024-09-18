"""NormCDF + NLL loss for multiclass classification."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.special import ndtr


class NormCDFNLLLoss(nn.Module):
    """NormCDF + NLL loss."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(logits, targets, *, apply_activation=True):
        # Compute sigmoid BCE loss
        targets = F.one_hot(targets, num_classes=logits.shape[-1])

        # Compute CDF of standard normal
        cdf = ndtr(logits) if apply_activation else logits

        # Compute loss
        loss = torch.where(targets == 1, -torch.log(cdf), -torch.log(1 - cdf))

        # Sum along the class dimension
        loss = loss.sum(dim=1)

        # Mean over the batch
        return loss.mean()
