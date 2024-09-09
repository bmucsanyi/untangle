"""DDU implementation as a wrapper class.

Implementation based on https://github.com/omegafragger/DDU
"""

import logging
from functools import partial

import torch
from torch import nn

from untangle.utils import centered_cov
from untangle.utils.replace import register, register_cond, replace
from untangle.wrappers.sngp_wrapper import (
    Conv2dSpectralNormalizer,
    LinearSpectralNormalizer,
    SpectralNormalizedBatchNorm2d,
)
from untangle.wrappers.temperature_wrapper import TemperatureWrapper

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [10**exp for exp in range(-25, 0, 1)]

logger = logging.getLogger(__name__)


class DDUWrapper(TemperatureWrapper):
    """This module takes a model as input and creates an SNGP from it."""

    def __init__(
        self,
        model: nn.Module,
        use_spectral_normalization: bool,
        spectral_normalization_iteration: int,
        spectral_normalization_bound: float,
        use_spectral_normalized_batch_norm: bool,
        use_tight_norm_for_pointwise_convs: bool,
    ):
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

                def is_pointwise_conv(conv2d: nn.Conv2d) -> bool:
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

    def forward_head(self, x, *, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        logits = self.get_classifier()(features)

        if self.training:
            return logits

        logits /= self.temperature

        if self._gmm_loc is None:
            logger.warning("GMM has not been fit yet; giving constant EU estimates.")

            gmm_log_density = torch.ones((x.shape[0],))
        else:
            if self._gmm is None:
                self._gmm = torch.distributions.MultivariateNormal(
                    loc=self._gmm_loc,
                    covariance_matrix=self._gmm_covariance_matrix,
                )

            gmm_log_densities = self._gmm.log_prob(
                features[:, None, :].cpu()
            ).cuda()  # [B, C]
            gmm_log_density = gmm_log_densities.logsumexp(dim=1)

        return {
            "logit": logits,
            "feature": features,
            "gmm_neg_log_density": -gmm_log_density,
        }

    def fit_gmm(self, train_loader, max_num_training_samples):
        features, labels = self._get_features(
            train_loader=train_loader, max_num_training_samples=max_num_training_samples
        )
        self._get_gmm(features=features, labels=labels)

    def _get_features(self, train_loader, max_num_training_samples):
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

                input = input.to(device)

                feature = self.model.forward_head(
                    self.model.forward_features(input), pre_logits=True
                )

                end = current_num_samples + actual_batch_size

                features[current_num_samples:end] = feature.detach().cpu()
                labels[current_num_samples:end] = label.detach().cpu()

                current_num_samples += actual_batch_size

        return features, labels

    def _get_gmm(self, features, labels):
        num_classes = self.model.num_classes

        classwise_mean_features = torch.stack([
            torch.mean(features[labels == c], dim=0) for c in range(num_classes)
        ])  # [C, D]

        classwise_cov_features = torch.stack([
            centered_cov(features[labels == c] - classwise_mean_features[c])
            for c in range(num_classes)
        ])  # [C, D, D]

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

                self._gmm = gmm
                self._gmm_loc = classwise_mean_features
                self._gmm_covariance_matrix = jittered_classwise_cov_features
            except RuntimeError:
                continue
            except ValueError:
                continue
            break

        logger.info(f"Used jitter: {jitter_eps}")

        if self._gmm is None:
            msg = "All jitters failed making the covariance matrix positive definite"
            raise ValueError(msg)
