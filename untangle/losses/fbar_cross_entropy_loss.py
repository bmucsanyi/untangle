"""Cross-entropy loss combined with the calculation of f_bar."""

import torch.nn.functional as F
from torch import nn


class FBarCrossEntropyLoss(nn.Module):
    """Implements a Cross-Entropy loss combined with f_bar calculation.

    This loss function applies log softmax to the input logits, computes the mean
    across the sample dimension, and then calculates the cross-entropy loss.
    """

    def __init__(self):
        super().__init__()

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        logits = F.log_softmax(logits, dim=-1).mean(dim=1)  # [B, C]

        return self.ce_loss(logits, targets)
