"""Laplace approximation wrapper class."""

import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from laplace import Laplace
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn.utils import vector_to_parameters

from untangle.utils.metric import calibration_error
from untangle.wrappers.model_wrapper import DistributionalWrapper

logger = logging.getLogger(__name__)


class LaplaceWrapper(DistributionalWrapper):
    """This module takes a model and creates a Laplace-approximated model posterior."""

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int,
        num_mc_samples_cv: int,
        weight_path: str,
        pred_type: str,  # "glm", "nn"
        hessian_structure: str,  # "kron", "full", "diag"
    ):
        super().__init__(model)

        self._num_mc_samples = num_mc_samples
        self._num_mc_samples_cv = num_mc_samples_cv
        self._weight_path = weight_path
        self._laplace_model = None
        self._pred_type = pred_type
        self._hessian_structure = hessian_structure

        self._load_model()

    def perform_laplace_approximation(self, train_loader, val_loader):
        with torch.enable_grad():
            self._laplace_model = Laplace(
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
        logger.info("Prior precision optimization done.")

    def forward_head(self, *args, **kwargs):
        # Warning! This class requires extra care, as the predictive samples are
        # sampled end-to-end from a black-box package. One can't use the usual strategy
        # of "obtain features => obtain logits". Instead, one has to obtain features
        # with `forward_features` and the logits with `forward`.
        del args, kwargs
        msg = f"forward_head cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward(self, input):
        if self._laplace_model is None:
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
    def _get_ece(out_dist, targets):
        confidences, predictions = out_dist.max(dim=-1)  # [B]
        correctnesses = predictions.eq(targets).int()

        return calibration_error(
            confidences=confidences, correctnesses=correctnesses, num_bins=15, norm="l1"
        )

    def _optimize_prior_precision_cv(
        self,
        val_loader,
        log_prior_prec_min=-1,
        log_prior_prec_max=2,
        grid_size=100,
    ):
        interval = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        self._laplace_model.prior_precision = self._gridsearch(
            interval=interval,
            val_loader=val_loader,
        )

        logger.info(
            f"Optimized prior precision is {self._laplace_model.prior_precision}."
        )

    def _gridsearch(
        self,
        interval,
        val_loader,
    ):
        results = []
        prior_precs = []
        for prior_prec in interval:
            logger.info(f"Trying {prior_prec}...")
            start_time = time.perf_counter()
            self._laplace_model.prior_precision = prior_prec

            try:
                out_dist, targets = self._validate(
                    val_loader=val_loader,
                )
                result = self._get_ece(out_dist, targets).item()
                accuracy = out_dist.argmax(dim=-1).eq(targets).float().mean()
            except RuntimeError as error:
                logger.info(f"Caught an exception in validate: {error}")
                result = float("inf")
                accuracy = float("NaN")
            logger.info(
                f"Took {time.perf_counter() - start_time} seconds, result: {result}, "
                f"accuracy {accuracy}"
            )
            results.append(result)
            prior_precs.append(prior_prec)

        return prior_precs[np.argmin(results)]

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        device = self._laplace_model._device  # noqa: SLF001
        output_means = []
        targets = []

        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)

            feature = self.model.forward_head(
                self.model.forward_features(input), pre_logits=True
            )

            out = self._logit_samples(
                feature=feature,
                num_samples=self._num_mc_samples_cv,
            )  # [B, S, C]
            out = F.softmax(out, dim=-1).mean(dim=1)  # [B, C]

            output_means.append(out)
            targets.append(target)

        return torch.cat(output_means, dim=0), torch.cat(targets, dim=0)

    def _nn_logit_samples(self, feature, num_samples=100):
        fs = []

        classifier = self.model.get_classifier()

        for sample in self._laplace_model.sample(num_samples):
            vector_to_parameters(sample, classifier.parameters())
            fs.append(classifier(feature).detach())

        vector_to_parameters(self._laplace_model.mean, classifier.parameters())
        fs = torch.stack(fs, dim=1)

        return fs

    def _glm_logit_distribution(self, feature):
        Js, f_mu = self._last_layer_jacobians(feature)
        f_var = self._laplace_model.functional_variance(Js)

        return f_mu.detach(), f_var.detach()

    def _logit_samples(self, feature, num_samples=100):
        """Sample from the posterior logits on `feature`.

        Parameters
        ----------
        feature : torch.Tensor
            pre-logit features `(batch_size, feature_dim)`.

        pred_type : {'glm', 'nn'}, default='glm'
            type of posterior predictive, linearized GLM predictive or neural
            network sampling predictive. The GLM predictive is consistent with
            the curvature approximations used here.

        num_samples : int
            number of samples

        Returns:
        -------
        samples : torch.Tensor
            samples `(batch_size, num_samples, output_shape)`
        """
        if self._pred_type not in {"glm", "nn"}:
            msg = "Only glm and nn supported as prediction types"
            raise ValueError(msg)

        if self._pred_type == "glm":
            f_mu, f_var = self._glm_logit_distribution(feature=feature)
            dist = MultivariateNormal(f_mu, f_var)
            samples = dist.sample((num_samples,))

            return samples.permute(1, 0, 2)
        # 'nn'
        return self._nn_logit_samples(feature=feature, num_samples=num_samples)

    def _last_layer_jacobians(self, feature):
        """Compute Jacobians only at current last-layer parameter."""
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
