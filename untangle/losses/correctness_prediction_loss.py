"""Correctness prediction loss."""

import torch
from torch import Tensor, nn


class CorrectnessPredictionLoss(nn.Module):
    """Custom loss function for correctness prediction in classification tasks.

    This loss combines a task loss (cross-entropy) with an uncertainty loss (binary
    cross-entropy) for predicting the correctness of the model's classification. It
    supports both top-1 and top-5 accuracy metrics.

    Args:
        lambda_uncertainty_loss: The weight factor for the uncertainty loss component.
        detach_uncertainty_target: If True, detaches the correctness target from the
            computation graph.
        use_top5_correctness: If True, uses top-5 correctness. Otherwise, uses top-1.
    """

    def __init__(
        self,
        lambda_uncertainty_loss: float,
        detach_uncertainty_target: bool,
        use_top5_correctness: bool,
    ) -> None:
        super().__init__()
        self.task_loss = nn.CrossEntropyLoss(reduction="none")
        self.uncertainty_loss = nn.BCEWithLogitsLoss()
        self.lambda_uncertainty_loss = lambda_uncertainty_loss
        self.detach_uncertainty_target = detach_uncertainty_target
        self.use_top5_correctness = use_top5_correctness

    def forward(
        self,
        prediction_tuple: tuple[Tensor, Tensor],
        target: Tensor,
    ) -> Tensor:
        """Computes the combined task and uncertainty loss.

        Args:
            prediction_tuple: A tuple containing the main prediction and the
                correctness prediction tensors.
            target: The ground truth labels.

        Returns:
            The computed loss value.
        """
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

        if self.detach_uncertainty_target:
            correctness_target = correctness.detach()
        else:
            correctness_target = correctness

        uncertainty_loss = self.uncertainty_loss(
            correctness_prediction, correctness_target
        )
        task_loss = task_loss_per_sample.mean()

        return task_loss + self.lambda_uncertainty_loss * uncertainty_loss
