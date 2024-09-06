"""HET-XL implementation as a wrapper class.

Heteroscedastic Gaussian sampling based on https://github.com/google/uncertainty-baselines.
"""

import torch
import torch.nn.functional as F
from torch import nn

from untangle.wrappers.model_wrapper import DistributionalWrapper


class HETXLHead(nn.Module):
    """Classification head for the HET-XL method."""

    def __init__(
        self, matrix_rank, num_mc_samples, num_features, temperature, classifier, is_het
    ):
        super().__init__()
        self._matrix_rank = matrix_rank
        self._num_mc_samples = num_mc_samples
        self._num_features = num_features

        self._low_rank_cov_layer = nn.Linear(
            in_features=self._num_features,
            out_features=self._num_features * self._matrix_rank,
        )
        self._diagonal_std_layer = nn.Linear(
            in_features=self._num_features, out_features=self._num_features
        )
        self._min_scale_monte_carlo = 1e-3

        self._temperature = temperature
        self._classifier = classifier
        self._is_het = is_het

    def forward(self, features):
        if self._is_het:
            features = self._classifier(features)  # D = C

        # Shape variables
        B, D = features.shape
        R = self._matrix_rank
        S = self._num_mc_samples

        low_rank_cov = self._low_rank_cov_layer(features).reshape(-1, D, R)  # [B, D, R]
        diagonal_std = (
            F.softplus(self._diagonal_std_layer(features)) + self._min_scale_monte_carlo
        )  # [B, D]

        # TODO(bmucsanyi): https://github.com/google/edward2/blob/main/edward2/jax/nn/heteroscedastic_lib.py#L189
        diagonal_samples = diagonal_std.unsqueeze(1) * torch.randn(
            B, S, D, device=features.device
        )  # [B, S, D]
        standard_samples = torch.randn(B, S, R, device=features.device)  # [B, S, R]
        einsum_res = torch.einsum(
            "bdr,bsr->bsd", low_rank_cov, standard_samples
        )  # [B, S, D]
        samples = einsum_res + diagonal_samples  # [B, S, D]

        pre_logits = features.unsqueeze(1) + samples  # [B, S, D]

        logits = self._classifier(pre_logits) if not self._is_het else pre_logits
        logits_temperature = logits / self._temperature

        # TODO(bmucsanyi): https://github.com/google/edward2/blob/main/edward2/jax/nn/heteroscedastic_lib.py#L325

        return logits_temperature


class HETXLWrapper(DistributionalWrapper):
    """This module takes a model as input and creates a HET-XL model from it."""

    def __init__(
        self,
        model: nn.Module,
        matrix_rank: int,
        num_mc_samples: int,
        temperature: float,
        is_het: bool,
    ):
        super().__init__(model)

        self._matrix_rank = matrix_rank
        self._num_mc_samples = num_mc_samples
        self._temperature = temperature
        self._is_het = is_het

        self._classifier = HETXLHead(
            matrix_rank=self._matrix_rank,
            num_mc_samples=self._num_mc_samples,
            num_features=self.num_features if not self._is_het else self.num_classes,
            classifier=self.model.get_classifier(),
            temperature=self._temperature,
            is_het=self._is_het,
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self._classifier

    def reset_classifier(
        self,
        matrix_rank: int | None = None,
        num_mc_samples: int | None = None,
        temperature: float | None = None,
        is_het: bool | None = None,
        *args,
        **kwargs,
    ):
        if matrix_rank is not None:
            self._matrix_rank = matrix_rank

        if num_mc_samples is not None:
            self._num_mc_samples = num_mc_samples

        if temperature is not None:
            self._temperature = temperature

        if is_het is not None:
            self._is_het = is_het

        self.model.reset_classifier(*args, **kwargs)
        self._classifier = HETXLHead(
            matrix_rank=self._matrix_rank,
            num_mc_samples=self._num_mc_samples,
            num_features=self.num_features if not self._is_het else self.num_classes,
            classifier=self.model.get_classifier(),
            temperature=self._temperature,
            is_het=self._is_het,
        )
