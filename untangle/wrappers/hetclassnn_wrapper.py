"""Heteroscedastic classification NN implementation as a wrapper class.

The dropout layout is based on https://github.com/google/uncertainty-baselines and the
method is based on https://arxiv.org/abs/1703.04977.
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from untangle.utils.replace import replace
from untangle.wrappers.mc_dropout_wrapper import ActivationDropout
from untangle.wrappers.model_wrapper import DistributionalWrapper


class HetClassNNWrapper(DistributionalWrapper):
    """This module takes a model as input and creates a Dropout model from it."""

    def __init__(
        self,
        model: nn.Module,
        dropout_probability: float,
        is_filterwise_dropout: bool,
        num_mc_samples: int,
        num_mc_samples_integral: int,
    ):
        super().__init__(model)

        self._num_mc_samples = num_mc_samples
        self._num_integral_mc_samples = num_mc_samples_integral

        replace(
            model,
            "ReLU",
            partial(ActivationDropout, dropout_probability, is_filterwise_dropout),
        )
        replace(
            model,
            "GELU",
            partial(ActivationDropout, dropout_probability, is_filterwise_dropout),
        )

        self._log_var = nn.Linear(
            in_features=self.model.num_features, out_features=self.model.num_classes
        )
        nn.init.normal_(self._log_var.weight, mean=0, std=0.01)
        nn.init.zeros_(self._log_var.bias)
        self.eps = 1e-10

    def forward(self, inputs):
        if self.training:
            return self._predict_single(inputs)

        sampled_features = []
        sampled_logits = []
        sampled_internal_logits = []
        for _ in range(self._num_mc_samples):
            features, internal_logits, logit_mc_samples = self._predict_single(
                inputs=inputs, return_bundle=True
            )
            logits = (
                F.softmax(logit_mc_samples, dim=-1).mean(dim=1).add(self.eps).log()
            )  # [B, C]

            sampled_features.append(features)
            sampled_logits.append(logits)
            sampled_internal_logits.append(internal_logits)

        sampled_features = torch.stack(sampled_features, dim=1)  # [B, S, D]
        mean_features = sampled_features.mean(dim=1)  # [B, D]
        sampled_logits = torch.stack(sampled_logits, dim=1)  # [B, S, C]
        sampled_internal_logits = torch.stack(
            sampled_internal_logits, dim=1
        )  # [B, S, C]

        return {
            "logit": sampled_logits,
            "internal_logit": sampled_internal_logits,
            "feature": mean_features,
        }

    def reset_classifier(self, num_classes, *args, **kwargs):
        self._log_var = nn.Linear(
            in_features=self.num_features, out_features=num_classes
        )
        nn.init.normal_(self._log_var.weight, mean=0, std=0.01)
        nn.init.zeros_(self._log_var.bias)
        self.model.reset_classifier(num_classes, *args, **kwargs)

    def forward_features(self, inputs):
        del inputs
        msg = f"forward_features cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward_head(self, features):
        del features
        msg = f"forward_head cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def _predict_single(self, inputs, *, return_bundle=False):
        pre_logits = self.model.forward_head(
            self.model.forward_features(inputs), pre_logits=True
        )  # [B, C]
        logits = self.model.get_classifier()(pre_logits)  # [B, C]
        variances = self._log_var(pre_logits).exp()  # [B, C]
        stds = variances.sqrt()  # [B, C]

        logit_mc_samples = logits.unsqueeze(1) + stds.unsqueeze(1) * torch.randn(
            inputs.shape[0],
            self._num_integral_mc_samples,
            self.model.num_classes,
            device=logits.device,
        )  # [B, S', C]

        if return_bundle:
            return pre_logits, logits, logit_mc_samples
        return logit_mc_samples
