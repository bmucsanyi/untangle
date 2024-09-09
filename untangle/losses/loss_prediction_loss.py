"""Loss prediction loss."""

from torch import Tensor, nn


class LossPredictionLoss(nn.Module):
    """Combines the task loss with loss prediction for uncertainty estimation.

    Args:
        lambda_uncertainty_loss (float): Weight for the uncertainty loss component.
        detach_task_loss (bool): If True, detaches task loss when used as target for
            loss prediction.
    """

    def __init__(
        self,
        lambda_uncertainty_loss,
        detach_task_loss,
    ):
        super().__init__()

        self.task_loss = nn.CrossEntropyLoss(reduction="none")
        self.uncertainty_loss = nn.MSELoss()
        self.lambda_uncertainty_loss = lambda_uncertainty_loss
        self.detach_task_loss = detach_task_loss

    def forward(
        self,
        prediction_tuple: tuple,
        target: Tensor,
    ) -> Tensor:
        prediction, loss_prediction = prediction_tuple

        task_loss_per_sample = self.task_loss(prediction, target)

        if self.detach_task_loss:
            task_loss_target = task_loss_per_sample.detach()
        else:
            task_loss_target = task_loss_per_sample

        uncertainty_loss = self.uncertainty_loss(loss_prediction, task_loss_target)
        task_loss = task_loss_per_sample.mean()

        return task_loss + self.lambda_uncertainty_loss * uncertainty_loss
