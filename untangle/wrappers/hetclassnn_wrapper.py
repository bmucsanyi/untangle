"""Heteroscedastic classification NN implementation as a wrapper class.

The dropout layout is based on https://github.com/google/uncertainty-baselines and the
method is based on https://arxiv.org/abs/1703.04977.
"""

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from untangle.utils.replace import replace
from untangle.wrappers.mc_dropout_wrapper import ActivationDropout
from untangle.wrappers.model_wrapper import DistributionalWrapper


class HetClassNNWrapper(DistributionalWrapper):
    """Wrapper that creates a HetClassNN from an input model.

    Args:
        model: The base model to be wrapped.
        dropout_probability: Probability of dropout.
        use_filterwise_dropout: Whether to use filterwise dropout.
        num_mc_samples: Number of Monte Carlo samples.
        num_mc_samples_integral: Number of Monte Carlo samples for integral.
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_probability: float,
        use_filterwise_dropout: bool,
        num_mc_samples: int,
        num_mc_samples_integral: int,
    ) -> None:
        super().__init__(model)

        self._num_mc_samples = num_mc_samples
        self._num_integral_mc_samples = num_mc_samples_integral

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

        self._log_var = nn.Linear(
            in_features=self.model.num_features, out_features=self.model.num_classes
        )
        nn.init.normal_(self._log_var.weight, mean=0, std=0.01)
        nn.init.zeros_(self._log_var.bias)
        self.eps = 1e-10

    def forward(self, input: Tensor) -> Tensor | dict[str, Tensor]:
        """Performs a forward pass through the model.

        Args:
            input: Input tensor.

        Returns:
            Output tensor or dictionary of tensors.
        """
        if self.training:
            return self._predict_single(input=input, return_bundle=False)

        sampled_logits = []
        sampled_internal_logits = []
        for _ in range(self._num_mc_samples):
            internal_logits, logit_mc_samples = self._predict_single(
                input=input, return_bundle=True
            )
            logits = (
                F.softmax(logit_mc_samples, dim=-1).mean(dim=1).add(self.eps).log()
            )  # [B, C]

            sampled_logits.append(logits)
            sampled_internal_logits.append(internal_logits)

        sampled_logits = torch.stack(sampled_logits, dim=1)  # [B, S, C]
        sampled_internal_logits = torch.stack(
            sampled_internal_logits, dim=1
        )  # [B, S, C]

        return {
            "logit": sampled_logits,
            "internal_logit": sampled_internal_logits,
        }

    def reset_classifier(self, num_classes: int, *args: Any, **kwargs: Any) -> None:
        """Resets the classifier with a new number of classes.

        Args:
            num_classes: New number of classes.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._log_var = nn.Linear(
            in_features=self.num_features, out_features=num_classes
        )
        nn.init.normal_(self._log_var.weight, mean=0, std=0.01)
        nn.init.zeros_(self._log_var.bias)
        self.model.reset_classifier(num_classes, *args, **kwargs)

    def forward_features(self, inputs: Tensor) -> None:
        """Raises an error as this method cannot be called directly.

        Args:
            inputs: Input tensor.

        Raises:
            ValueError: Always raised.
        """
        del inputs
        msg = f"forward_features cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> Tensor | dict[str, Tensor]:
        """Raises an error as this method cannot be called directly.

        Args:
            input: Input tensor.
            pre_logits: Whether to return pre-logits instead of logits.

        Raises:
            ValueError: Always raised.
        """
        del input, pre_logits
        msg = f"forward_head cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def _predict_single(
        self, input: Tensor, *, return_bundle: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Performs a single prediction.

        Args:
            input: Input tensor.
            return_bundle: Whether to return a bundle of outputs.

        Returns:
            Tensor or tuple of tensors.
        """
        pre_logits = self.model.forward_head(
            self.model.forward_features(input), pre_logits=True
        )  # [B, C]
        logits = self.model.get_classifier()(pre_logits)  # [B, C]
        variances = self._log_var(pre_logits).exp()  # [B, C]
        stds = variances.sqrt()  # [B, C]

        logit_mc_samples = logits.unsqueeze(1) + stds.unsqueeze(1) * torch.randn(
            input.shape[0],
            self._num_integral_mc_samples,
            self.model.num_classes,
            device=logits.device,
        )  # [B, S', C]

        if return_bundle:
            return logits, logit_mc_samples

        return logit_mc_samples
