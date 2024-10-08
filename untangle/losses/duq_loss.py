"""DUQ loss."""

import torch
from torch import Tensor, nn


class DUQLoss(nn.Module):
    """Implements the Deep Uncertainty Quantification (DUQ) loss.

    This class wraps the Binary Cross Entropy (BCE) loss and disables automatic mixed
    precision during forward computation to ensure numerical stability.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.loss = nn.BCELoss()

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Computes the DUQ loss.

        Args:
            prediction: The predicted probabilities.
            target: The target probabilities.

        Returns:
            The computed DUQ loss value.
        """
        with torch.autocast("cuda", enabled=False):
            return self.loss(prediction, target)
