"""Sigmoid + NLL loss for multiclass classification."""

import torch
import torch.nn.functional as F
from torch import nn


class SigmoidNLLLoss(nn.Module):
    """Sigmoid + NLL loss."""

    def __init__(self):
        super().__init__()

        self._softplus = nn.Softplus()

    def forward(self, logits, targets):
        targets = F.one_hot(targets, num_classes=logits.shape[-1])

        # Compute sigmoid BCE loss
        loss = self._softplus(logits) - targets * logits

        # Sum along the class dimension
        loss = loss.mean(dim=1)

        # Mean over the batch
        return loss.mean()


class SigmoidNLLLoss2(nn.Module):
    """Timm implementation."""

    def __init__(
        self,
        smoothing=0.0,
        target_threshold: float | None = None,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
        pos_weight: torch.Tensor | float | None = None,
        *,
        sum_classes: bool = False,
    ):
        super().__init__()
        if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
            pos_weight = torch.tensor(pos_weight)
        self.smoothing = smoothing
        self.target_threshold = target_threshold
        self.reduction = "none" if sum_classes else reduction
        self.sum_classes = sum_classes
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        if target.shape != x.shape:
            num_classes = x.shape[-1]
            off_value = self.smoothing / num_classes
            on_value = 1.0 - self.smoothing + off_value
            target = target.long().view(-1, 1)
            target = torch.full(
                (batch_size, num_classes), off_value, device=x.device, dtype=x.dtype
            ).scatter_(1, target, on_value)

        if self.target_threshold is not None:
            # Make target 0, or 1 if threshold set
            target = target.gt(self.target_threshold).to(dtype=target.dtype)

        loss = F.binary_cross_entropy_with_logits(
            x,
            target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )
        if self.sum_classes:
            loss = loss.sum(-1).mean()

        return loss
