"""Correctness prediction loss."""

import torch
from torch import Tensor, nn


class CorrectnessPredictionLoss(nn.Module):
    """A custom loss function for correctness prediction in classification tasks.

    This loss combines a task loss (cross-entropy) with an uncertainty loss (binary
    cross-entropy) for predicting the correctness of the model's classification. It
    supports both top-1 and top-5 accuracy metrics.

    Args:
        lambda_uncertainty_loss (float): The weight factor for the uncertainty loss
            component.
        use_top5_correctness (bool): If True, uses top-5 correctness.
            Otherwise, uses top-1.
    """

    def __init__(
        self,
        lambda_uncertainty_loss,
        use_top5_correctness,
    ):
        super().__init__()
        self.task_loss = nn.CrossEntropyLoss(reduction="none")
        self.uncertainty_loss = nn.BCEWithLogitsLoss()
        self.lambda_uncertainty_loss = lambda_uncertainty_loss
        self.use_top5_correctness = use_top5_correctness

    def forward(
        self,
        prediction_tuple: tuple,
        target: Tensor,
    ) -> Tensor:
        prediction, correctness_prediction = prediction_tuple

        task_loss_per_sample = self.task_loss(prediction, target)

        if self.use_top5_correctness:
            _, prediction_argmax_top5 = torch.topk(prediction, 5, dim=1)
            expanded_gt_hard_labels = target.unsqueeze(dim=1).expand_as(
                prediction_argmax_top5
            )
            correctness = (
                prediction_argmax_top5.eq(expanded_gt_hard_labels).max(dim=1)[0].float()
            )
        else:
            correctness = prediction.argmax(dim=-1).eq(target).float()

        uncertainty_loss = self.uncertainty_loss(correctness_prediction, correctness)
        task_loss = task_loss_per_sample.mean()

        return task_loss + self.lambda_uncertainty_loss * uncertainty_loss
