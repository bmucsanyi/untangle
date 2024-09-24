"""NormCDF + NLL loss for multiclass classification."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.special import log_ndtr


class NormCDFNLLLoss(nn.Module):
    """NormCDF + NLL loss."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(logits, targets):
        # Compute sigmoid BCE loss
        targets = F.one_hot(targets, num_classes=logits.shape[-1])

        # Compute loss
        loss = torch.where(
            targets == 1,
            -log_ndtr(logits.double()).float(),
            -log_ndtr(-logits.double()).float(),
        )

        # Sum along the class dimension
        loss = loss.mean(dim=1)

        # Mean over the batch
        return loss.mean()
