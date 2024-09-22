"""HET implementation as a wrapper class.

Heteroscedastic Gaussian sampling based on https://github.com/google/uncertainty-baselines.
"""

import torch
import torch.nn.functional as F
from torch import nn

from untangle.wrappers.model_wrapper import DistributionalWrapper


class HETHead(nn.Module):
    """Classification head for the HET method."""

    def __init__(
        self,
        matrix_rank,
        num_mc_samples,
        num_classes,
        temperature,
        classifier,
        use_sampling,
    ):
        super().__init__()
        self._matrix_rank = matrix_rank
        self._num_mc_samples = num_mc_samples
        self._use_sampling = use_sampling
        self._num_classes = num_classes

        if self._matrix_rank > 0:
            self._low_rank_cov_layer = nn.Linear(
                in_features=self._num_classes,
                out_features=self._num_classes * self._matrix_rank,
            )
        self._diagonal_var_layer = nn.Linear(
            in_features=self._num_classes, out_features=self._num_classes
        )
        self._min_scale_monte_carlo = 1e-3

        self._temperature = temperature
        self._classifier = classifier

    def forward(self, features):
        logits = self._classifier(features)  # [B, C]

        # Shape variables
        B, C = logits.shape
        S = self._num_mc_samples
        R = self._matrix_rank

        if R > 0:
            low_rank_cov = self._low_rank_cov_layer(logits).reshape(
                -1, C, R
            )  # [B, C, R]

        diagonal_var = (
            F.softplus(self._diagonal_var_layer(logits)) + self._min_scale_monte_carlo
        )  # [B, C]

        if self._use_sampling:
            diagonal_samples = diagonal_var.sqrt().unsqueeze(1) * torch.randn(
                B, S, C, device=features.device
            )  # [B, S, C]

            if R > 0:
                standard_samples = torch.randn(
                    B, S, R, device=features.device
                )  # [B, S, R]
                einsum_res = torch.einsum(
                    "bcr,bsr->bsc", low_rank_cov, standard_samples
                )  # [B, S, D]
                samples = einsum_res + diagonal_samples  # [B, S, C]
            else:
                samples = diagonal_samples

            logits = logits.unsqueeze(1) + samples  # [B, S, C]

            return (logits / self._temperature,)

        if R > 0:
            vars = low_rank_cov.square().sum(dim=-1) + diagonal_var  # [B, C]
        else:
            vars = diagonal_var

        return logits / self._temperature, vars / self._temperature**2


class HETWrapper(DistributionalWrapper):
    """This module takes a model as input and creates a HET model from it."""

    def __init__(
        self,
        model: nn.Module,
        matrix_rank: int,
        num_mc_samples: int,
        temperature: float,
        use_sampling: bool,
    ):
        super().__init__(model)

        self._matrix_rank = matrix_rank
        self._num_mc_samples = num_mc_samples
        self._temperature = temperature
        self._use_sampling = use_sampling

        self._classifier = HETHead(
            matrix_rank=self._matrix_rank,
            num_mc_samples=self._num_mc_samples,
            num_classes=self.num_classes,
            temperature=self._temperature,
            classifier=self.model.get_classifier(),
            use_sampling=self._use_sampling,
        )

    def get_classifier(self):
        return self._classifier

    def reset_classifier(
        self,
        matrix_rank: int | None = None,
        num_mc_samples: int | None = None,
        temperature: float | None = None,
        use_sampling: bool | None = None,
        *args,
        **kwargs,
    ):
        if matrix_rank is not None:
            self._matrix_rank = matrix_rank

        if num_mc_samples is not None:
            self._num_mc_samples = num_mc_samples

        if temperature is not None:
            self._temperature = temperature

        if use_sampling is not None:
            self._use_sampling = use_sampling

        self.model.reset_classifier(*args, **kwargs)
        self._classifier = HETHead(
            matrix_rank=self._matrix_rank,
            num_mc_samples=self._num_mc_samples,
            num_classes=self.num_classes,
            temperature=self._temperature,
            classifier=self.model.get_classifier(),
            use_sampling=self._use_sampling,
        )
