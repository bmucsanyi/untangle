"""PostNet model wrapper class."""

import math

import torch
import torch.distributions as tdist
import torch.nn.functional as F
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.util import copy_docs_from
from torch import nn
from torch.distributions import MultivariateNormal, Transform, constraints

from untangle.wrappers.model_wrapper import DirichletWrapper


class NormalizingFlowDensity(nn.Module):
    """Non-batched normalizing flow density."""

    def __init__(self, dim: int, flow_length: int):
        super().__init__()
        self._dim = dim
        self._flow_length = flow_length

        self.register_buffer("_standard_mean", torch.zeros(dim))
        self.register_buffer("_standard_cov", torch.eye(dim))

        self._transforms = nn.Sequential(*(Radial(dim) for _ in range(flow_length)))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
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


@copy_docs_from(Transform)
class ConditionedRadial(Transform):
    """Non-batched radial transform used by non-batched normalizing flows."""

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, params):
        super().__init__(cache_size=1)
        self._params = params
        self._cached_logDetJ = None

    # This method ensures that torch(u_hat, w) > -1, required for invertibility
    @staticmethod
    def u_hat(u, w):
        del u, w
        raise NotImplementedError

    def _call(self, x):
        """Invokes the bijection x=>y.

        In the prototypical context of a `pyro.distributions.TransformedDistribution`,
        `x` is a sample from the
        base distribution (or the output of a previous transform)
        """
        x0, alpha_prime, beta_prime = (
            self._params() if callable(self._params) else self._params
        )

        # Ensure invertibility using approach in appendix A.2
        alpha = F.softplus(alpha_prime)
        beta = -alpha + F.softplus(beta_prime)

        # Compute y and logDet using Equation 14.
        diff = x - x0[:, None, :]
        r = diff.norm(dim=-1, keepdim=True).squeeze()
        h = (alpha[:, None] + r).reciprocal()
        h_prime = -(h**2)
        beta_h = beta[:, None] * h

        self._cached_logDetJ = (x0.shape[-1] - 1) * torch.log1p(beta_h) + torch.log1p(
            beta_h + beta[:, None] * h_prime * r
        )
        return x + beta_h[:, :, None] * diff

    @staticmethod
    def _inverse(y):
        """Inverts y => x.

        As noted above, this implementation is incapable of
        inverting arbitrary values `y`; rather it assumes `y` is the result of a
        previously computed application of the bijector to some `x` (which was
        cached on the forward call)
        """
        del y

        msg = (
            "ConditionedRadial object expected to find key in intermediates cache "
            "but didn't"
        )
        raise KeyError(msg)

    def log_abs_det_jacobian(self, x, y):
        """Calculates the elementwise determinant of the log Jacobian."""
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cached_logDetJ


@copy_docs_from(ConditionedRadial)
class BatchedRadial(ConditionedRadial, TransformModule):
    """Batched radial transform used by batched normalizing flows."""

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, c, input_dim):
        super().__init__(self._params)

        self._x0 = nn.Parameter(
            torch.Tensor(
                c,
                input_dim,
            )
        )
        self._alpha_prime = nn.Parameter(
            torch.Tensor(
                c,
            )
        )
        self._beta_prime = nn.Parameter(
            torch.Tensor(
                c,
            )
        )
        self._c = c
        self._input_dim = input_dim
        self.reset_parameters()

    def _params(self):
        return self._x0, self._alpha_prime, self._beta_prime

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self._x0.shape[1])
        self._alpha_prime.data.uniform_(-stdv, stdv)
        self._beta_prime.data.uniform_(-stdv, stdv)
        self._x0.data.uniform_(-stdv, stdv)


class BatchedNormalizingFlowDensity(nn.Module):
    """Normalizing flow density that is batched for multiple classes."""

    def __init__(self, c, dim, flow_length):
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

    def forward(self, z):
        sum_log_jacobians = 0
        z = z.repeat(self._c, 1, 1)
        for transform in self._transforms:
            z_next = transform(z)
            sum_log_jacobians += transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(
            self._mean.repeat(z.shape[1], 1, 1).permute(1, 0, 2),
            self._cov.repeat(z.shape[1], 1, 1, 1).permute(1, 0, 2, 3),
        ).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        return log_prob_x


class PostNetWrapper(DirichletWrapper):
    """This module takes a model as input and creates a PostNet model from it."""

    def __init__(
        self,
        model: nn.Module,
        latent_dim: int,
        hidden_dim: int,
        num_density_components: int,
        use_batched_flow: bool,
    ):
        super().__init__(model)

        # TODO(bmucsanyi): Come back to check if these are needed/useful
        self._latent_dim = latent_dim
        self.num_features = latent_dim  # For public access & compatibility
        self._hidden_dim = hidden_dim
        self._num_density_components = num_density_components
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
                flow_length=self._num_density_components,
            )
        else:
            self._density_estimator = nn.ModuleList([
                NormalizingFlowDensity(
                    dim=self._latent_dim, flow_length=self._num_density_components
                )
                for _ in range(self.num_classes)
            ])

    def calculate_sample_counts(self, train_loader):
        device = next(self.model.parameters()).device
        sample_count_per_class = torch.zeros(self.num_classes)

        for _, targets in train_loader:
            targets_cpu = targets.cpu()
            sample_count_per_class.scatter_add_(
                0, targets_cpu, torch.ones_like(targets_cpu, dtype=torch.float)
            )

        self._sample_count_per_class = sample_count_per_class.to(device)

    def forward_head(self, x, *, pre_logits: bool = False):
        if self._sample_count_per_class is None:
            msg = "Call to `calculate_sample_counts` needed first"
            raise ValueError(msg)

        # Pre-logits are the outputs of the wrapped model
        features = self._batch_norm(self.model.forward_head(x))  # [B, D]

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

        return (alphas,)

    def get_classifier(self):
        return self._classifier

    def reset_classifier(
        self,
        hidden_dim: int | None = None,
        num_classes: int | None = None,
    ):
        if hidden_dim is not None:
            self._hidden_dim = hidden_dim

        if num_classes is not None:
            self.num_classes = num_classes

        self._classifier = nn.Sequential(
            nn.Linear(in_features=self._latent_dim, out_features=self._hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self._hidden_dim, out_features=self.num_classes),
        )
