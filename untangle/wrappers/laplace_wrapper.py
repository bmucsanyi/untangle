"""Laplace approximation wrapper class."""

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from laplace import Laplace, LLLaplace
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader

from untangle.utils.loader import PrefetchLoader
from untangle.utils.metric import calibration_error
from untangle.wrappers.model_wrapper import DistributionalWrapper

logger = logging.getLogger(__name__)


class LaplaceWrapper(DistributionalWrapper):
    """Wrapper that creates a Laplace-approximated posterior from an input model.

    Args:
        model: The neural network model to be wrapped.
        num_mc_samples: Number of Monte Carlo samples for prediction.
        num_mc_samples_cv: Number of Monte Carlo samples for cross-validation.
        weight_path: Path to the model weights.
        pred_type: Type of prediction ("glm" or "nn").
        hessian_structure: Structure of the Hessian approximation
            ("kron", "full", or "diag").
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        num_mc_samples_cv: int,
        weight_path: str,
        pred_type: str,
        hessian_structure: str,
    ) -> None:
        super().__init__(model)

        self._num_mc_samples = num_mc_samples
        self._num_mc_samples_cv = num_mc_samples_cv
        self._is_laplace_approximated = False
        self._pred_type = pred_type
        self._hessian_structure = hessian_structure

        self._load_model(weight_path)

    def perform_laplace_approximation(
        self,
        train_loader: DataLoader | PrefetchLoader,
        val_loader: DataLoader | PrefetchLoader,
    ) -> None:
        """Performs Laplace approximation and optimize prior precision.

        Args:
            train_loader: DataLoader or PrefetchLoader for the training data.
            val_loader: DataLoader or PrefetchLoader for the validation data.
        """
        with torch.enable_grad():
            self._laplace_model: LLLaplace = Laplace(
                self.model,
                "classification",
                subset_of_weights="last_layer",
                hessian_structure=self._hessian_structure,
            )
            logger.info("Starting Laplace approximation.")
            self._laplace_model.fit(train_loader)
            logger.info("Laplace approximation done.")

        logger.info("Starting prior precision optimization.")
        self._optimize_prior_precision_cv(
            val_loader=val_loader,
        )
        self._is_laplace_approximated = True
        logger.info("Prior precision optimization done.")

    def forward_head(self, *args: Any, **kwargs: Any) -> Tensor | dict[str, Tensor]:
        """Raises an error as it cannot be called directly for this class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            ValueError: Always raised when this method is called.
        """
        del args, kwargs
        msg = f"forward_head cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward(self, input: Tensor) -> Tensor | dict[str, Tensor]:
        """Performs forward pass with Laplace-approximated model.

        Args:
            input: Input tensor to the model.

        Returns:
            Dictionary containing the logit tensor.

        Raises:
            ValueError: If the model hasn't been Laplace-approximated yet.
        """
        if not self._is_laplace_approximated:
            msg = "Model has to be Laplace-approximated first"
            raise ValueError(msg)

        if self.training:
            return self.model(input)
        feature = self.model.forward_head(
            self.model.forward_features(input), pre_logits=True
        )

        logit = self._logit_samples(
            feature=feature,
            num_samples=self._num_mc_samples,
        )  # [B, S, C]

        return {
            "logit": logit,
        }

    @staticmethod
    def _get_ece(out_dist: Tensor, targets: Tensor) -> Tensor:
        """Calculates the Expected Calibration Error.

        Args:
            out_dist: Output distribution from the model.
            targets: True labels.

        Returns:
            Calculated Expected Calibration Error.
        """
        confidences, predictions = out_dist.max(dim=-1)  # [B]
        correctnesses = predictions.eq(targets).int()

        return calibration_error(
            confidences=confidences, correctnesses=correctnesses, num_bins=15, norm="l1"
        )

    def _optimize_prior_precision_cv(
        self,
        val_loader: DataLoader | PrefetchLoader,
        log_prior_prec_min: float = -1,
        log_prior_prec_max: float = 2,
        grid_size: int = 500,
    ) -> None:
        """Optimizes prior precision using cross-validation.

        Args:
            val_loader: DataLoader or PrefetchLoader for the validation data.
            log_prior_prec_min: Minimum log prior precision.
            log_prior_prec_max: Maximum log prior precision.
            grid_size: Number of grid points for optimization.
        """
        interval = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        self._laplace_model.prior_precision = self._grid_search(
            interval=interval,
            val_loader=val_loader,
        )

        logger.info(
            f"Optimized prior precision is {self._laplace_model.prior_precision}."
        )

    def _grid_search(
        self,
        interval: Tensor,
        val_loader: DataLoader | PrefetchLoader,
    ) -> float:
        """Performs grid search to find optimal prior precision.

        Args:
            interval: Tensor of prior precision values to search.
            val_loader: DataLoader or PrefetchLoader for the validation data.

        Returns:
            Optimal prior precision value.
        """
        results = []
        prior_precs = []
        for prior_prec in interval:
            logger.info(f"Trying {prior_prec}...")
            start_time = time.perf_counter()
            self._laplace_model.prior_precision = prior_prec

            try:
                out_dist, targets = self._validate(val_loader=val_loader)
                result = self._get_ece(out_dist, targets).item()
                accuracy = out_dist.argmax(dim=-1).eq(targets).float().mean()
            except RuntimeError as error:
                logger.info(f"Caught an exception in validate: {error}")
                result = float("inf")
                accuracy = float("NaN")
            logger.info(
                f"Took {time.perf_counter() - start_time} seconds, result: {result}, "
                f"accuracy: {accuracy}."
            )
            results.append(result)
            prior_precs.append(prior_prec)

        return prior_precs[np.argmin(results)]

    @torch.no_grad()
    def _validate(
        self, val_loader: DataLoader | PrefetchLoader
    ) -> tuple[Tensor, Tensor]:
        """Validates the model on the validation set.

        Args:
            val_loader: DataLoader or PrefetchLoader for the validation data.

        Returns:
            Tuple of output means and targets.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        output_means = []
        targets = []

        for input, target in val_loader:
            if not isinstance(val_loader, PrefetchLoader):
                input, target = input.to(device), target.to(device)

            feature = self.model.forward_head(
                self.model.forward_features(input), pre_logits=True
            )

            out = self._logit_samples(
                feature=feature,
                num_samples=self._num_mc_samples_cv,
            )  # [B, S, C]
            out = F.softmax(out, dim=-1).mean(dim=1)  # [B, C]

            if out.device.type == "cuda":
                torch.cuda.synchronize()

            output_means.append(out)
            targets.append(target)

        return torch.cat(output_means, dim=0), torch.cat(targets, dim=0)

    def _nn_logit_samples(self, feature: Tensor, num_samples: int = 100) -> Tensor:
        """Generates logit samples using neural network sampling.

        Args:
            feature: Input features.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of logit samples.
        """
        fs = []

        classifier = self.model.get_classifier()

        for sample in self._laplace_model.sample(num_samples):
            vector_to_parameters(sample, classifier.parameters())
            fs.append(classifier(feature).detach())

        vector_to_parameters(self._laplace_model.mean, classifier.parameters())
        fs = torch.stack(fs, dim=1)

        return fs

    def _glm_logit_distribution(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        """Calculates the logit distribution using GLM.

        Args:
            feature: Input features.

        Returns:
            Tuple of mean and variance of the logit distribution.
        """
        Js, f_mu = self._last_layer_jacobians(feature)
        f_var = self._laplace_model.functional_variance(Js)

        return f_mu.detach(), f_var.detach()

    def _logit_samples(self, feature: Tensor, num_samples: int = 100) -> Tensor:
        """Samples from the posterior logits on the given features.

        Args:
            feature: Input features.
            num_samples: Number of samples to generate.

        Returns:
            Tensor of logit samples.

        Raises:
            ValueError: If an invalid prediction type is specified.
        """
        if self._pred_type not in {"glm", "nn"}:
            msg = "Only glm and nn supported as prediction types"
            raise ValueError(msg)

        if self._pred_type == "glm":
            f_mu, f_var = self._glm_logit_distribution(feature=feature)
            dist = MultivariateNormal(f_mu, f_var)
            samples = dist.sample(torch.Size((num_samples,)))

            return samples.permute(1, 0, 2)
        # 'nn'
        return self._nn_logit_samples(feature=feature, num_samples=num_samples)

    def _last_layer_jacobians(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        """Computes Jacobians only at current last-layer parameter.

        Args:
            feature: Input features.

        Returns:
            Tuple of Jacobians and logits.
        """
        classifier = self.model.get_classifier()

        logit = classifier(feature)

        batch_size = feature.shape[0]
        num_classes = logit.shape[-1]

        # Calculate Jacobians using the feature vector 'feature'
        identity = (
            torch.eye(num_classes, device=feature.device)
            .unsqueeze(0)
            .tile(batch_size, 1, 1)
        )
        # Jacobians are batch x output x params
        Js = torch.einsum("kp,kij->kijp", feature, identity).reshape(
            batch_size, num_classes, -1
        )

        if classifier.bias is not None:
            Js = torch.cat([Js, identity], dim=2)

        return Js.detach(), logit.detach()
