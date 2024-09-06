"""Cross-entropy loss combined with Bayesian Model Averaging."""

import torch.nn.functional as F
from torch import nn


class BMACrossEntropyLoss(nn.Module):
    """Implements Cross-entropy loss combined with Bayesian Model Averaging.

    This class extends nn.Module to provide a loss function that combines
    cross-entropy with Bayesian Model Averaging (BMA). It's designed to work
    with logits from multiple models or samples.

    The loss is computed by first applying softmax to the input logits,
    averaging the probabilities, and then computing the negative log-likelihood
    loss.

    Example:
        >>> loss_fn = BMACrossEntropyLoss()
        >>> logits = torch.randn(10, 5, 3)  # [batch_size, num_models, num_classes]
        >>> targets = torch.randint(0, 3, (10,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(self):
        super().__init__()

        self.ce_loss = nn.NLLLoss()
        self.eps = 1e-10

    def forward(self, logits, targets):
        log_probs = F.softmax(logits, dim=-1).mean(dim=1).add(self.eps).log()  # [B, C]

        return self.ce_loss(log_probs, targets)
