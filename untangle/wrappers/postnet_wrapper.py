"""PostNet model wrapper class."""

import math
from collections.abc import Callable

import torch
import torch.distributions as tdist
import torch.nn.functional as F
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms.radial import Radial
from torch import Tensor, nn
from torch.distributions import MultivariateNormal, constraints
from torch.utils.data import DataLoader

from untangle.utils.loader import PrefetchLoader
from untangle.wrappers.model_wrapper import DirichletWrapper


class NormalizingFlowDensity(nn.Module):
    """Implements a non-batched normalizing flow density.

    Args:
        dim: Dimensionality of the input space.
        flow_length: Number of flow transformations.
    """

    def __init__(self, dim: int, flow_length: int) -> None:
        super().__init__()
        self._dim = dim
        self._flow_length = flow_length

        self.register_buffer("_standard_mean", torch.zeros(dim))
        self.register_buffer("_standard_cov", torch.eye(dim))

        self._transforms = nn.Sequential(*(Radial(dim) for _ in range(flow_length)))

    def log_prob(self, x: Tensor) -> Tensor:
        """Computes the log probability of the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Log probability of the input.
        """
        sum_log_jacobians = 0
        z = x

        for transform in self._transforms:
            z_next = transform(z)
            sum_log_jacobians += transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        log_prob_z = MultivariateNormal(
            self._standard_mean, self._standard_cov
        ).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians

        return log_prob_x


class ConditionedRadial(TransformModule):
    """Implements a non-batched radial transform used by non-batched normalizing flows.

    Args:
        params: Parameters for the radial transform, either a callable or a tuple.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, params: Callable | tuple) -> None:
        super().__init__(cache_size=1)
        self._params = params
        self._cached_log_abs_det_J = None

    def _call(self, x: Tensor) -> Tensor:
        """Applies the bijection x=>y.

        In the context of a TransformedDistribution, x is a sample from the base
        distribution or the output of a previous transform.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        x0, alpha_prime, beta_prime = (
            self._params() if callable(self._params) else self._params
        )

        # Ensure invertibility
        alpha = F.softplus(alpha_prime)
        beta = -alpha + F.softplus(beta_prime)

        # Compute y and log_det_J
        diff = x - x0[:, None, :]
        r = diff.norm(dim=-1, keepdim=True).squeeze()
        h = (alpha[:, None] + r).reciprocal()
        h_prime = -(h**2)
        beta_h = beta[:, None] * h

        self._cached_log_abs_det_J = (x0.shape[-1] - 1) * torch.log1p(
            beta_h
        ) + torch.log1p(beta_h + beta[:, None] * h_prime * r)

        return x + beta_h[:, :, None] * diff

    def log_abs_det_jacobian(self, x: Tensor, y: Tensor) -> Tensor:
        """Calculates the elementwise determinant of the log Jacobian.

        Args:
            x: Input tensor.
            y: Output tensor.

        Returns:
            Log absolute determinant of the Jacobian.
        """
        x_old, y_old = self._cached_x_y

        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cached_log_abs_det_J


class BatchedRadial(ConditionedRadial):
    """Implements a batched radial transform used by batched normalizing flows.

    Args:
        c: Number of components in the batch.
        input_dim: Dimensionality of the input space.
    """

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, c: int, input_dim: int) -> None:
        super().__init__(self._params)

        self._x0 = nn.Parameter(torch.empty(c, input_dim))
        self._alpha_prime = nn.Parameter(torch.empty(c))
        self._beta_prime = nn.Parameter(torch.empty(c))
        self._c = c
        self._input_dim = input_dim
        self.reset_parameters()

    def _params(self) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the parameters of the batched radial transform.

        Returns:
            Tuple containing x0, alpha_prime, and beta_prime tensors.
        """
        return self._x0, self._alpha_prime, self._beta_prime

    def reset_parameters(self) -> None:
        """Resets the parameters of the batched radial transform."""
        std = 1.0 / math.sqrt(self._x0.shape[1])
        self._alpha_prime.data.uniform_(-std, std)
        self._beta_prime.data.uniform_(-std, std)
        self._x0.data.uniform_(-std, std)


class BatchedNormalizingFlowDensity(nn.Module):
    """Implements a normalizing flow density that is batched for multiple classes.

    Args:
        c: Number of classes.
        dim: Dimensionality of the input space.
        flow_length: Number of flow transformations.
    """

    def __init__(self, c: int, dim: int, flow_length: int) -> None:
        super().__init__()
        self._c = c
        self._dim = dim
        self._flow_length = flow_length

        self._mean = nn.Parameter(torch.zeros(self._c, self._dim), requires_grad=False)
        self._cov = nn.Parameter(
            torch.eye(self._dim).repeat(self._c, 1, 1), requires_grad=False
        )

        self._transforms = nn.Sequential(
            *(BatchedRadial(c, dim) for _ in range(self._flow_length))
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Applies the normalizing flow transformation.

        Args:
            z: Input tensor.

        Returns:
            Tuple containing the transformed tensor and sum of log Jacobians.
        """
        sum_log_abs_det_jacobians = 0
        z = z.repeat(self._c, 1, 1)

        for transform in self._transforms:
            z_next = transform(z)
            sum_log_abs_det_jacobians += transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        return z, sum_log_abs_det_jacobians

    def log_prob(self, x: Tensor) -> Tensor:
        """Computes the log probability of the input tensor.

        Args:
            x: Input tensor.

        Returns:
            Log probability of the input.
        """
        z, sum_log_abs_det_jacobians = self(x)
        log_prob_z = tdist.MultivariateNormal(
            self._mean.repeat(z.shape[1], 1, 1).permute(1, 0, 2),
            self._cov.repeat(z.shape[1], 1, 1, 1).permute(1, 0, 2, 3),
        ).log_prob(z)
        log_prob_x = log_prob_z + sum_log_abs_det_jacobians  # [B]

        return log_prob_x


class PostNetWrapper(DirichletWrapper):
    """Wrapper that creates a PostNet from an input model.

    Args:
        model: The neural network model to be wrapped.
        latent_dim: Dimensionality of the latent space.
        hidden_dim: Dimensionality of the hidden layer in the classifier.
        num_density_components: Number of components in the normalizing flow.
        use_batched_flow: Whether to use batched normalizing flow.
    """

    def __init__(
        self,
        model: nn.Module,
        latent_dim: int,
        hidden_dim: int,
        num_density_components: int,
        use_batched_flow: bool,
    ) -> None:
        super().__init__(model)

        self._latent_dim = latent_dim
        self.num_features = latent_dim  # For public access & compatibility
        self._hidden_dim = hidden_dim
        self.num_classes = model.num_classes
        self.register_buffer("_sample_count_per_class", torch.zeros(self.num_classes))

        # Use wrapped model as a feature extractor
        self.model.reset_classifier(num_classes=latent_dim)

        self._batch_norm = nn.BatchNorm1d(num_features=latent_dim)

        self._classifier = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=self.num_classes),
        )

        if use_batched_flow:
            self._density_estimator = BatchedNormalizingFlowDensity(
                c=self.num_classes,
                dim=self._latent_dim,
                flow_length=num_density_components,
            )
        else:
            self._density_estimator = nn.ModuleList([
                NormalizingFlowDensity(
                    dim=self._latent_dim, flow_length=num_density_components
                )
                for _ in range(self.num_classes)
            ])

    def calculate_sample_counts(
        self, train_loader: DataLoader | PrefetchLoader
    ) -> None:
        """Calculates the sample counts per class from the training data.

        Args:
            train_loader: DataLoader or PrefetchLoader for the training data.
        """
        device = next(self.model.parameters()).device
        sample_count_per_class = torch.zeros(self.num_classes)

        for _, targets in train_loader:
            if isinstance(train_loader, PrefetchLoader):
                targets = targets.cpu()

            sample_count_per_class.scatter_add_(
                0, targets, torch.ones_like(targets, dtype=torch.float)
            )

        self._sample_count_per_class = sample_count_per_class.to(device)

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> Tensor | dict[str, Tensor]:
        """Performs forward pass through the head of the model.

        Args:
            input: Input tensor.
            pre_logits: Whether to return pre-logits features.

        Returns:
            Features, alphas, or a dictionary containing alphas during inference.

        Raises:
            ValueError: If sample counts haven't been calculated.
        """
        if self._sample_count_per_class is None:
            msg = "Call to `calculate_sample_counts` needed first"
            raise ValueError(msg)

        # Pre-logits are the outputs of the wrapped model
        features = self._batch_norm(self.model.forward_head(input))  # [B, D]

        if pre_logits:
            return features

        if isinstance(self._density_estimator, nn.ModuleList):
            batch_size = features.shape[0]
            log_probs = torch.zeros(
                (batch_size, self.num_classes), device=features.device
            )
            alphas = torch.zeros((batch_size, self.num_classes), device=features.device)

            for c in range(self.num_classes):
                log_probs[:, c] = self._density_estimator[c].log_prob(features)
                alphas[:, c] = (
                    1 + self._sample_count_per_class[c] * log_probs[:, c].exp()
                )
        else:
            log_probs = self._density_estimator.log_prob(features)  # [C, B]
            alphas = (
                1 + self._sample_count_per_class.unsqueeze(1).mul(log_probs.exp()).T
            )  # [B, C]

        if self.training:
            return alphas

        return {
            "alpha": alphas,  # [B, C]
        }

    def get_classifier(self) -> nn.Sequential:
        """Returns the classifier part of the model."""
        return self._classifier

    def reset_classifier(
        self,
        hidden_dim: int | None = None,
        num_classes: int | None = None,
    ) -> None:
        """Resets the classifier with new dimensions.

        Args:
            hidden_dim: New hidden dimension size.
            num_classes: New number of classes.
        """
        if hidden_dim is not None:
            self._hidden_dim = hidden_dim

        if num_classes is not None:
            self.num_classes = num_classes

        self._classifier = nn.Sequential(
            nn.Linear(in_features=self._latent_dim, out_features=self._hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self._hidden_dim, out_features=self.num_classes),
        )
