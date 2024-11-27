"""DDU implementation as a wrapper class.

Implementation based on https://github.com/omegafragger/DDU
"""

import logging
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from untangle.utils import centered_cov
from untangle.utils.loader import PrefetchLoader
from untangle.utils.replace import register, register_cond, replace
from untangle.wrappers.sngp_wrapper import (
    Conv2dSpectralNormalizer,
    LinearSpectralNormalizer,
    SpectralNormalizedBatchNorm2d,
)
from untangle.wrappers.temperature_wrapper import TemperatureWrapper

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [10**exp for exp in range(-20, 0, 1)]

logger = logging.getLogger(__name__)


class DDUWrapper(TemperatureWrapper):
    """Wrapper that creates a DDU model from an input model.

    This wrapper applies spectral normalization and other techniques to create
    a Spectral-normalized Neural Gaussian Process (SNGP) from the input model.

    Args:
        model: The neural network model to be wrapped.
        use_spectral_normalization: Whether to use spectral normalization.
        spectral_normalization_iteration: Number of iterations for spectral
            normalization.
        spectral_normalization_bound: Upper bound for spectral normalization.
        use_spectral_normalized_batch_norm: Whether to use spectral normalized batch
            normalization.
        use_tight_norm_for_pointwise_convs: Whether to use tight norm for pointwise
            convolutions.
    """

    def __init__(
        self,
        model: nn.Module,
        use_spectral_normalization: bool,
        spectral_normalization_iteration: int,
        spectral_normalization_bound: float,
        use_spectral_normalized_batch_norm: bool,
        use_tight_norm_for_pointwise_convs: bool,
    ) -> None:
        super().__init__(model, None)

        self.register_buffer("_gmm_loc", None)
        self.register_buffer("_gmm_covariance_matrix", None)
        self._gmm = None

        if use_spectral_normalization:
            LSN = partial(
                LinearSpectralNormalizer,
                spectral_normalization_iteration=spectral_normalization_iteration,
                spectral_normalization_bound=spectral_normalization_bound,
                dim=0,
                eps=1e-12,
            )

            CSN = partial(
                Conv2dSpectralNormalizer,
                spectral_normalization_iteration=spectral_normalization_iteration,
                spectral_normalization_bound=spectral_normalization_bound,
                eps=1e-12,
            )

            SNBN = partial(
                SpectralNormalizedBatchNorm2d,
                spectral_normalization_bound=spectral_normalization_bound,
            )

            if use_tight_norm_for_pointwise_convs:

                def is_pointwise_conv(conv2d: nn.Module) -> bool:
                    return conv2d.kernel_size == (1, 1)

                register_cond(
                    model=model,
                    source_regex="Conv2d",
                    attribute_name="weight",
                    cond=is_pointwise_conv,
                    target_parametrization_true=LSN,
                    target_parametrization_false=CSN,
                )
            else:
                register(
                    model=model,
                    source_regex="Conv2d",
                    attribute_name="weight",
                    target_parametrization=CSN,
                )

            if use_spectral_normalized_batch_norm:
                replace(
                    model=model,
                    source_regex="BatchNorm2d",
                    target_module=SNBN,
                )

    def forward(
        self, input: Tensor, amp_autocast: Any | None = None
    ) -> Tensor | dict[str, Tensor]:
        """Performs forward pass through the entire model.

        Args:
            input: Input tensor.
            amp_autocast: Automatic mixed precision autocast.

        Returns:
            Model output or dictionary containing output tensors.
        """
        x = self.forward_features(input)
        x = self.forward_head(x, amp_autocast)

        return x

    def forward_head(
        self, x: Tensor, amp_autocast: Any | None, *, pre_logits: bool = False
    ) -> Tensor | dict[str, Tensor]:
        """Performs the forward pass through the head of the model.

        Args:
            x: The input tensor.
            pre_logits: Whether to return pre-logits.
            amp_autocast: Automatic mixed precision autocast.

        Returns:
            Either the features (if pre_logits is True), the logits (during training),
            or a dictionary containing logits and GMM negative log density
            (during inference).
        """
        # Always get pre_logits
        if amp_autocast is not None:
            with amp_autocast():
                features = self.model.forward_head(x, pre_logits=True)

                if pre_logits:
                    return features

                logits = self.get_classifier()(features)
        else:
            features = self.model.forward_head(x, pre_logits=True)

            if pre_logits:
                return features

            logits = self.get_classifier()(features)

        if self.training:
            return logits

        logits /= self._temperature

        if self._gmm_loc is None:
            # GMM has not been fit yet; giving constant EU estimates
            gmm_log_density = torch.ones((x.shape[0],))
        else:
            if self._gmm is None:
                self._gmm = torch.distributions.MultivariateNormal(
                    loc=self._gmm_loc,
                    covariance_matrix=self._gmm_covariance_matrix,
                )

            gmm_log_densities = self._gmm.log_prob(
                features[:, None, :].float().cpu()
            ).cuda()  # [B, C]
            gmm_log_density = gmm_log_densities.logsumexp(dim=1)

        return {
            "logit": logits,
            "gmm_neg_log_density": -gmm_log_density,
        }

    def fit_gmm(
        self,
        train_loader: DataLoader | PrefetchLoader,
        max_num_training_samples: int,
        channels_last: bool,
    ) -> None:
        """Fits a Gaussian Mixture Model to the features of the training data.

        Args:
            train_loader: The DataLoader or PrefetchLoader for the training set.
            max_num_training_samples: The maximum number of training samples to use.
            channels_last: Whether a channels_last memory layout should be used.
        """
        features, labels = self._get_features(
            train_loader=train_loader,
            max_num_training_samples=max_num_training_samples,
            channels_last=channels_last,
        )
        self._get_gmm(features=features, labels=labels)

    def _get_features(
        self,
        train_loader: DataLoader | PrefetchLoader,
        max_num_training_samples: int,
        channels_last: bool,
    ) -> tuple[Tensor, Tensor]:
        """Extracts features from the training data.

        Args:
            train_loader: The DataLoader or PrefetchLoader for the training set.
            max_num_training_samples: The maximum number of training samples to use.
            channels_last: Whether a channels_last memory layout should be used.

        Returns:
            A tuple containing the extracted features and corresponding labels.
        """
        batch_size = train_loader.batch_size
        dataset_size = len(train_loader.dataset)
        drop_last = train_loader.drop_last

        # Calculate the effective number of samples based on drop_last setting
        if drop_last:
            total_full_batches = dataset_size // batch_size
            effective_dataset_size = total_full_batches * batch_size
        else:
            effective_dataset_size = dataset_size

        # Determine the actual number of samples to process
        num_samples = min(effective_dataset_size, max_num_training_samples)

        features = torch.empty((num_samples, self.num_features))
        labels = torch.empty((num_samples,), dtype=torch.int)
        device = next(self.model.parameters()).device

        with torch.no_grad():
            current_num_samples = 0
            for input, label in train_loader:
                if current_num_samples >= num_samples:
                    # Ensure we don't process beyond the desired number of samples
                    break

                # Calculate how many samples we can process in this batch
                actual_batch_size = min(
                    input.shape[0], num_samples - current_num_samples
                )
                input = input[:actual_batch_size]
                label = label[:actual_batch_size]

                if not isinstance(train_loader, PrefetchLoader):
                    input = input.to(device)
                else:
                    label = label.cpu()

                if channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                feature = self.model.forward_head(
                    self.model.forward_features(input), pre_logits=True
                )

                if feature.device.type == "cuda":
                    torch.cuda.synchronize()

                end = current_num_samples + actual_batch_size

                features[current_num_samples:end] = feature.detach().cpu()
                labels[current_num_samples:end] = label

                current_num_samples += actual_batch_size

        return features, labels

    def _get_gmm(self, features: Tensor, labels: Tensor) -> None:
        """Computes the Gaussian Mixture Model parameters.

        Args:
            features: The extracted features.
            labels: The corresponding labels.

        Raises:
            ValueError: If all jitters fail to make the covariance matrix positive
            definite.
        """
        num_classes = self.model.num_classes

        classwise_mean_features = torch.stack([
            torch.mean(features[labels == c], dim=0) for c in range(num_classes)
        ])  # [C, D]

        classwise_cov_features = torch.stack([
            centered_cov(features[labels == c] - classwise_mean_features[c])
            for c in range(num_classes)
        ])  # [C, D, D]

        used_jitter_eps = None

        for jitter_eps in JITTERS:
            logger.info(f"Trying {jitter_eps}...")

            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1]
                ).unsqueeze(0)

                jittered_classwise_cov_features = classwise_cov_features + jitter

                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features,
                    covariance_matrix=jittered_classwise_cov_features,
                )

                used_jitter_eps = jitter_eps
                self._gmm = gmm
                self._gmm_loc = classwise_mean_features
                self._gmm_covariance_matrix = jittered_classwise_cov_features
            except RuntimeError:
                continue
            except ValueError:
                continue
            break

        logger.info(f"Used jitter: {used_jitter_eps}")

        if used_jitter_eps is None:
            msg = "All jitters failed making the covariance matrix positive definite"
            raise ValueError(msg)
