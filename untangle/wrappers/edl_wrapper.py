"""EDL model wrapper class."""

import torch.nn.functional as F
from torch import Tensor, nn

from untangle.wrappers.model_wrapper import DirichletWrapper


class EDLWrapper(DirichletWrapper):
    """Wrapper that creates an EDL model from an input model.

    Args:
        model: The base model to be wrapped.
        activation: Activation function to use ('exp' or 'softplus').
    """

    def __init__(
        self,
        model: nn.Module,
        activation: str,
    ) -> None:
        super().__init__(model)

        if activation == "exp":
            self._activation = lambda x: x.clamp(-10, 10).exp()
        elif activation == "softplus":
            self._activation = F.softplus
        else:
            msg = f'Invalid activation "{activation}" provided'
            raise ValueError(msg)

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> Tensor | dict[str, Tensor]:
        """Performs a forward pass through the head of the model.

        Args:
            input: Input tensor.
            pre_logits: If True, returns features before the final classification layer.

        Returns:
            Features, alphas, or a dictionary containing alphas.
        """
        # Always get pre_logits
        features = self.model.forward_head(input, pre_logits=True)

        if pre_logits:
            return features

        logits = self.get_classifier()(features)
        alphas = self._activation(logits).add(1)  # [B, C]

        if self.training:
            return alphas

        return {
            "alpha": alphas,  # [B, C]
        }
