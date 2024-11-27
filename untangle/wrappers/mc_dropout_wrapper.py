"""Dropout implementation as a wrapper class.

The dropout layout is based on https://github.com/google/uncertainty-baselines.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from untangle.utils.replace import replace
from untangle.wrappers.model_wrapper import DistributionalWrapper


class ActivationDropout(nn.Module):
    """Applies an activation function followed by Dropout.

    Args:
        dropout_probability: Probability of an element to be zeroed.
        use_filterwise_dropout: Whether to use filter-wise dropout.
        activation: Activation function to be applied.
    """

    def __init__(
        self,
        dropout_probability: float,
        use_filterwise_dropout: bool,
        activation: nn.Module,
    ) -> None:
        super().__init__()
        self._activation = activation
        dropout_function = F.dropout2d if use_filterwise_dropout else F.dropout
        self._dropout = partial(dropout_function, p=dropout_probability, training=True)

    def forward(self, inputs: Tensor) -> Tensor:
        """Applies activation and dropout to the input tensor.

        Args:
            inputs: Input tensor.

        Returns:
            Tensor after applying activation and dropout.
        """
        x = self._activation(inputs)
        x = self._dropout(x)
        return x


class MCDropoutWrapper(DistributionalWrapper):
    """Wrapper that creates an MC-Dropout model from an input model.

    Args:
        model: The neural network model to be wrapped.
        dropout_probability: Probability of an element to be zeroed.
        use_filterwise_dropout: Whether to use filter-wise dropout.
        num_mc_samples: Number of Monte Carlo samples to generate.
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_probability: float,
        use_filterwise_dropout: bool,
        num_mc_samples: int,
    ) -> None:
        super().__init__(model)

        self._num_mc_samples = num_mc_samples

        replace(
            model,
            "ReLU",
            partial(ActivationDropout, dropout_probability, use_filterwise_dropout),
        )
        replace(
            model,
            "GELU",
            partial(ActivationDropout, dropout_probability, use_filterwise_dropout),
        )

    def forward(self, input: Tensor) -> Tensor | dict[str, Tensor]:
        """Performs forward pass with MC-Dropout.

        Args:
            input: Input tensor.

        Returns:
            Model output during training or dictionary containing sampled logits during
            inference.
        """
        if self.training:
            return self.model(input)  # [B, C]

        sampled_logit = []
        for _ in range(self._num_mc_samples):
            feature = self.model.forward_head(
                self.model.forward_features(input), pre_logits=True
            )
            logit = self.model.get_classifier()(feature)  # [B, C]

            sampled_logit.append(logit)

        sampled_logit = torch.stack(sampled_logit, dim=1)  # [B, S, C]

        return {"logit": sampled_logit}

    def forward_features(self, inputs: Tensor) -> None:
        """Raises an error as it cannot be called directly for this wrapper.

        Args:
            inputs: Input tensor.

        Raises:
            ValueError: Always raised when this method is called.
        """
        del inputs
        msg = f"forward_features cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> Tensor | dict[str, Tensor]:
        """Raises an error as it cannot be called directly for this wrapper.

        Args:
            input: Input tensor.
            pre_logits: Whether to return pre-logits instead of logits.

        Raises:
            ValueError: Always raised when this method is called.
        """
        del input, pre_logits
        msg = f"forward_head cannot be called directly for {type(self)}"
        raise ValueError(msg)
