"""HET-XL implementation as a wrapper class.

Heteroscedastic Gaussian sampling based on https://github.com/google/uncertainty-baselines.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from untangle.wrappers.model_wrapper import DistributionalWrapper


class HETXLHead(nn.Module):
    """Classification head for the HET-XL method.

    Args:
        matrix_rank: Rank of the low-rank covariance matrix.
        num_mc_samples: Number of Monte Carlo samples.
        num_in_features: Number of input features.
        num_out_features: Number of output features.
        temperature: Temperature for scaling logits.
        classifier: Classifier module.
        use_het: Whether to use heteroscedastic model.
    """

    def __init__(
        self,
        matrix_rank: int,
        num_mc_samples: int,
        num_in_features: int,
        num_out_features: int,
        temperature: float,
        classifier: nn.Module,
        use_het: bool,
    ) -> None:
        super().__init__()
        self._matrix_rank = matrix_rank
        self._num_mc_samples = num_mc_samples
        self._num_out_features = num_out_features

        self._low_rank_cov_layer = nn.Linear(
            in_features=num_in_features,
            out_features=num_out_features * matrix_rank,
        )
        self._diagonal_std_layer = nn.Linear(
            in_features=num_in_features, out_features=num_out_features
        )
        self._min_scale_monte_carlo = 1e-3

        self._temperature = temperature
        self._classifier = classifier
        self._use_het = use_het

    def forward(self, features: Tensor) -> Tensor:
        """Performs a forward pass through the HET-XL head.

        Args:
            features: Input features.

        Returns:
            Temperature-scaled logits.
        """
        # Shape variables
        B = features.shape[0]
        D_out = self._num_out_features  # Either C or D
        R = self._matrix_rank
        S = self._num_mc_samples

        low_rank_cov = self._low_rank_cov_layer(features).reshape(
            -1, D_out, R
        )  # [B, C | D, R]
        diagonal_std = (
            F.softplus(self._diagonal_std_layer(features)) + self._min_scale_monte_carlo
        )  # [B, C | D]

        diagonal_samples = diagonal_std.unsqueeze(1) * torch.randn(
            B, S, D_out, device=features.device
        )  # [B, S, C | D]
        standard_samples = torch.randn(B, S, R, device=features.device)  # [B, S, R]
        einsum_res = torch.einsum(
            "bdr,bsr->bsd", low_rank_cov, standard_samples
        )  # [B, S, C | D]
        samples = einsum_res + diagonal_samples  # [B, S, C | D]

        if self._use_het:
            logits = self._classifier(features)  # [B, C]
            logits = logits.unsqueeze(1) + samples  # [B, S, C]
        else:
            pre_logits = features.unsqueeze(1) + samples  # [B, S, D]
            logits = self._classifier(pre_logits)

        logits_temperature = logits / self._temperature

        return logits_temperature


class HETXLWrapper(DistributionalWrapper):
    """Wrapper that creates a HET(-XL) model from an input model.

    Args:
        model: The backbone model to wrap.
        matrix_rank: Rank of the low-rank covariance matrix.
        num_mc_samples: Number of Monte Carlo samples.
        temperature: Temperature for scaling logits.
        use_het: Whether to use HET instead of HET-XL.
    """

    def __init__(
        self,
        model: nn.Module,
        matrix_rank: int,
        num_mc_samples: int,
        temperature: float,
        use_het: bool,
    ) -> None:
        super().__init__(model)

        self._matrix_rank = matrix_rank
        self._num_mc_samples = num_mc_samples
        self._temperature = temperature
        self._use_het = use_het

        self._classifier = HETXLHead(
            matrix_rank=self._matrix_rank,
            num_mc_samples=self._num_mc_samples,
            num_in_features=self.num_features,
            num_out_features=self.num_classes if self._use_het else self.num_features,
            classifier=self.model.get_classifier(),
            temperature=self._temperature,
            use_het=self._use_het,
        )

    def get_classifier(self) -> HETXLHead:
        """Returns the HET-XL classifier head.

        Returns:
            The HETXLHead instance.
        """
        return self._classifier

    def reset_classifier(
        self,
        matrix_rank: int | None = None,
        num_mc_samples: int | None = None,
        temperature: float | None = None,
        use_het: bool | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Resets the classifier with new parameters.

        Args:
            matrix_rank: New matrix rank for the low-rank covariance.
            num_mc_samples: New number of Monte Carlo samples.
            temperature: New temperature for scaling logits.
            use_het: New flag for using heteroscedastic model.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if matrix_rank is not None:
            self._matrix_rank = matrix_rank

        if num_mc_samples is not None:
            self._num_mc_samples = num_mc_samples

        if temperature is not None:
            self._temperature = temperature

        if use_het is not None:
            self._use_het = use_het

        self.model.reset_classifier(*args, **kwargs)
        self._classifier = HETXLHead(
            matrix_rank=self._matrix_rank,
            num_mc_samples=self._num_mc_samples,
            num_in_features=self.num_features,
            num_out_features=self.num_classes if self._use_het else self.num_features,
            classifier=self.model.get_classifier(),
            temperature=self._temperature,
            use_het=self._use_het,
        )
