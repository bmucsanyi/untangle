"""DUQ implementation as a wrapper class.

Based on https://github.com/y0ast/deterministic-uncertainty-quantification.
"""

import torch
import torch.nn.functional as F
from torch import nn

from untangle.wrappers.model_wrapper import SpecialWrapper


class DUQHead(nn.Module):
    """Classification head for the DUQ method."""

    def __init__(
        self,
        num_classes,
        num_features,
        rbf_length_scale,
        ema_momentum,
        num_hidden_features,
    ):
        super().__init__()
        self._num_classes = num_classes

        if num_hidden_features < 0:
            num_hidden_features = num_features

        self._num_hidden_features = num_hidden_features
        self._num_features = num_features
        self._rbf_length_scale = rbf_length_scale
        self._ema_momentum = ema_momentum

        self._weight = nn.Parameter(
            torch.empty(num_classes, num_hidden_features, num_features)
        )
        nn.init.kaiming_normal_(self._weight, nonlinearity="relu")

        self.register_buffer(
            "_ema_num_samples_per_class", torch.full((num_classes,), 128 / num_classes)
        )
        self.register_buffer(
            "_ema_embedding_sums_per_class",
            torch.randn(num_classes, num_hidden_features),
        )

    def update_centroids(self, features, targets):
        prev_state = self.training

        self.eval()

        num_samples_per_class = targets.sum(dim=0)
        self._ema_num_samples_per_class = (
            self._ema_momentum * self._ema_num_samples_per_class
            + (1 - self._ema_momentum) * num_samples_per_class
        )

        latent_features = torch.einsum(
            "clf,bf->bcl", self._weight, features
        )  # [B, C, L]
        embedding_sums_per_class = torch.einsum(
            "bcl,bc->cl", latent_features, targets
        )  # [C, L]

        self._ema_embedding_sums_per_class = (
            self._ema_momentum * self._ema_embedding_sums_per_class
            + (1 - self._ema_momentum) * embedding_sums_per_class
        )

        self.train(prev_state)

    def forward(self, features):
        rbf_values = self._rbf(features)

        if self.training:
            return rbf_values
        min_real = torch.finfo(rbf_values.dtype).min
        logit = (
            rbf_values.div(rbf_values.sum(dim=1, keepdim=True))
            .log()
            .clamp(min=min_real)
        )

        return {
            "logit": logit,  # [B, C]
            "feature": features,  # [B, D]
            "duq_value": 1 - rbf_values.max(dim=1)[0],  # [B]
        }

    def _rbf(self, features):
        latent_features = torch.einsum(
            "clf,bf->bcl", self._weight, features
        )  # [B, C, L]

        centroids = (
            self._ema_embedding_sums_per_class
            / self._ema_num_samples_per_class.unsqueeze(dim=1)
        )  # [C, L]

        diffs = latent_features - centroids  # [B, C, L]
        rbf_values = (
            diffs.square().mean(dim=-1).div(2 * self._rbf_length_scale**2).mul(-1).exp()
        )  # [B, C]

        return rbf_values



class DUQWrapper(SpecialWrapper):
    """This module takes a model as input and creates a DUQ model from it."""

    def __init__(
        self,
        model: nn.Module,
        num_hidden_features,
        rbf_length_scale,
        ema_momentum,
    ):
        super().__init__(model)

        self._num_hidden_features = num_hidden_features
        self._rbf_length_scale = rbf_length_scale
        self._ema_momentum = ema_momentum

        self._classifier = DUQHead(
            num_classes=self.num_classes,
            num_features=self.num_features,
            rbf_length_scale=self._rbf_length_scale,
            ema_momentum=self._ema_momentum,
            num_hidden_features=self._num_hidden_features,
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self._classifier

    def reset_classifier(
        self,
        num_hidden_features: int | None = None,
        rbf_length_scale: float | None = None,
        ema_momentum: float | None = None,
        *args,
        **kwargs,
    ):
        if num_hidden_features is not None:
            self._num_hidden_features = num_hidden_features

        if rbf_length_scale is not None:
            self._rbf_length_scale = rbf_length_scale

        if ema_momentum is not None:
            self._ema_momentum = ema_momentum

        self.model.reset_classifier(*args, **kwargs)
        self._classifier = DUQHead(
            num_classes=self.num_classes,
            num_features=self.num_features,
            rbf_length_scale=self._rbf_length_scale,
            ema_momentum=self._ema_momentum,
            num_hidden_features=self._num_hidden_features,
        )

    def update_centroids(self, inputs, targets):
        features = self.model.forward_head(
            self.model.forward_features(inputs), pre_logits=True
        )
        self._classifier.update_centroids(features, targets)

    def forward_head(self, x, *, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        out = self.get_classifier()(features)

        return out

    def prepare_data(self, input, target):
        input.requires_grad_(True)
        target = F.one_hot(target, self.num_classes).float()

    @staticmethod
    def calc_gradient_penalty(x, pred):
        gradients = DUQWrapper._calc_gradients_input(x, pred)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two-sided penalty
        gradient_penalty = (grad_norm - 1).square().mean()

        return gradient_penalty

    @staticmethod
    def _calc_gradients_input(x, pred):
        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=x,
            grad_outputs=torch.ones_like(pred),
            retain_graph=True,  # Graph still needed for loss backprop
        )[0]

        gradients = gradients.flatten(start_dim=1)

        return gradients

