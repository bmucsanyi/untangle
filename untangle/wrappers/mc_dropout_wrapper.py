"""Dropout implementation as a wrapper class.

The dropout layout is based on https://github.com/google/uncertainty-baselines.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from untangle.utils.replace import replace
from untangle.wrappers.model_wrapper import DistributionalWrapper


class ActivationDropout(nn.Module):
    """Activation function followed by Dropout."""

    def __init__(self, dropout_probability, use_filterwise_dropout, activation):
        super().__init__()
        self._activation = activation
        dropout_function = F.dropout2d if use_filterwise_dropout else F.dropout
        self._dropout = partial(dropout_function, p=dropout_probability, training=True)

    def forward(self, inputs):
        x = self._activation(inputs)
        x = self._dropout(x)
        return x


class MCDropoutWrapper(DistributionalWrapper):
    """This module takes a model as input and creates an MC-Dropout model from it."""

    def __init__(
        self,
        model: nn.Module,
        dropout_probability: float,
        use_filterwise_dropout: bool,
        num_mc_samples: int,
    ):
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

    def forward(self, inputs):
        if self.training:
            return self.model(inputs)  # [B, C]

        sampled_logits = []
        for _ in range(self._num_mc_samples):
            features = self.model.forward_head(
                self.model.forward_features(inputs), pre_logits=True
            )
            logits = self.model.get_classifier()(features)  # [B, C]

            sampled_logits.append(logits)

        sampled_logits = torch.stack(sampled_logits, dim=1)  # [B, S, C]

        return {"logit": sampled_logits}

    def forward_features(self, inputs):
        del inputs
        msg = f"forward_features cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward_head(self, features):
        del features
        msg = f"forward_head cannot be called directly for {type(self)}"
        raise ValueError(msg)
