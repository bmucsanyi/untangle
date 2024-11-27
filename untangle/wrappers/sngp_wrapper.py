"""SNGP/GP implementation as a wrapper class.

SNGP implementation based on https://github.com/google/edward2
"""

import itertools
import math
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _NormBase  # noqa: PLC2701
from torch.nn.parameter import is_lazy

from untangle.utils import calculate_output_padding, calculate_same_padding
from untangle.utils.metric import (
    diag_hessian_softmax,
)
from untangle.utils.replace import register, register_cond, replace
from untangle.wrappers.model_wrapper import DistributionalWrapper


class GPOutputLayer(nn.Module):
    """Random feature GP output layer.

    This layer implements a Gaussian Process output using random features.

    Args:
        num_features: Number of input features.
        num_classes: Number of output classes.
        num_mc_samples: Number of Monte Carlo samples.
        num_random_features: Number of random features to use.
        gp_kernel_scale: Scale of the GP kernel.
        gp_output_bias: Output bias for the GP.
        gp_random_feature_type: Type of random features ('orf' or 'rff').
        use_input_normalized_gp: Whether to use input normalization for GP.
        gp_cov_momentum: Momentum for covariance update.
        gp_cov_ridge_penalty: Ridge penalty for covariance.
        likelihood: Likelihood type ('gaussian' or 'softmax').
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        num_mc_samples: int,
        num_random_features: int,
        gp_kernel_scale: float | None,
        gp_output_bias: float,
        gp_random_feature_type: str,
        use_input_normalized_gp: bool,
        gp_cov_momentum: float,
        gp_cov_ridge_penalty: float,
        likelihood: str,
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._num_random_features = num_random_features
        self._num_mc_samples = num_mc_samples

        self._use_input_normalized_gp = use_input_normalized_gp
        self._gp_input_scale = (
            1 / gp_kernel_scale**0.5 if gp_kernel_scale is not None else None
        )

        self._gp_kernel_scale = gp_kernel_scale
        self._gp_output_bias = gp_output_bias
        self._likelihood = likelihood

        if gp_random_feature_type == "orf":
            self._random_features_weight_initializer = partial(
                self._orthogonal_random_features_initializer, std=0.05
            )
        elif gp_random_feature_type == "rff":
            self._random_features_weight_initializer = partial(
                nn.init.normal_, mean=0.0, std=0.05
            )
        else:
            msg = (
                "gp_random_feature_type must be one of 'orf' or 'rff', got "
                f"{gp_random_feature_type}"
            )
            raise ValueError(msg)

        self._gp_cov_momentum = gp_cov_momentum
        self._gp_cov_ridge_penalty = gp_cov_ridge_penalty

        # Default to Gaussian RBF kernel with orthogonal random features.
        self._random_features_bias_initializer = partial(
            nn.init.uniform_, a=0, b=2 * torch.pi
        )

        if self._use_input_normalized_gp:
            self._input_norm_layer = nn.LayerNorm(num_features)

        self._random_feature = self._make_random_feature_layer(num_features)

        num_cov_layers = 1 if self._likelihood == "gaussian" else num_classes
        self._gp_cov_layers = nn.ModuleList(
            LaplaceRandomFeatureCovariance(
                gp_feature_dim=self._num_random_features,
                momentum=self._gp_cov_momentum,
                ridge_penalty=self._gp_cov_ridge_penalty,
            )
            for _ in range(num_cov_layers)
        )

        self._gp_output_layer = nn.Linear(
            in_features=self._num_random_features,
            out_features=self._num_classes,
            bias=False,
        )

        self._gp_output_bias = nn.Parameter(
            torch.tensor([self._gp_output_bias] * self._num_classes),
            requires_grad=False,
        )

    def reset_covariance_matrix(self) -> None:
        """Resets covariance matrix of the GP layer."""
        for cov_layer in self._gp_cov_layers:
            cov_layer.reset_precision_matrix()

    @staticmethod
    def mean_field_logits(
        logits: Tensor, vars: Tensor, mean_field_factor: float
    ) -> Tensor:
        """Computes mean-field logits.

        Args:
            logits: Input logits.
            vars: Variances.
            mean_field_factor: Mean field factor.

        Returns:
            Mean-field logits.
        """
        # Compute scaling coefficient for mean-field approximation.
        logits_scale = (1 + vars * mean_field_factor).sqrt()

        # Cast logits_scale to compatible dimension.
        logits_scale = logits_scale.reshape(-1, 1)

        return logits / logits_scale

    @staticmethod
    def monte_carlo_sample_logits(
        logits: Tensor, vars: Tensor, num_samples: int
    ) -> Tensor:
        """Performs Monte Carlo sampling of logits.

        Args:
            logits: Input logits.
            vars: Variances.
            num_samples: Number of samples.

        Returns:
            Sampled logits.
        """
        batch_size, num_classes = logits.shape
        vars = vars.unsqueeze(dim=1)  # [B, 1, C]

        std_normal_samples = torch.randn(
            batch_size, num_samples, num_classes, device=logits.device
        )  # [B, S, C]

        return vars.sqrt() * std_normal_samples + logits.unsqueeze(dim=1)

    def forward(self, gp_inputs: Tensor) -> Tensor:
        """Performs forward pass of the GP output layer.

        Args:
            gp_inputs: Input features.

        Returns:
            Output logits or samples.
        """
        # Computes random features.
        if self._use_input_normalized_gp:
            gp_inputs = self._input_norm_layer(gp_inputs)
        elif self._gp_input_scale is not None:
            # Supports lengthscale for custom random feature layer by directly
            # rescaling the input.
            gp_inputs *= self._gp_input_scale

        gp_features = self._random_feature(gp_inputs).cos()

        # Computes posterior center (i.e., MAP estimate) and variance
        gp_outputs = self._gp_output_layer(gp_features) + self._gp_output_bias  # [B, C]

        if self.training:
            if self._likelihood == "gaussian":
                self._gp_cov_layers[0].update(gp_features)
            else:  # self._likelihood == "softmax"
                multipliers = diag_hessian_softmax(gp_outputs)

                for cov_layer, multiplier in zip(
                    self._gp_cov_layers, multipliers.T, strict=True
                ):
                    cov_layer.update(gp_features, multiplier)

            return gp_outputs  # [B, C]

        if self._likelihood == "gaussian":
            gp_vars = (
                self._gp_cov_layers[0](gp_features)
                .unsqueeze(1)
                .repeat(1, gp_outputs.shape[-1])
            )
        else:
            with torch.no_grad():
                gp_vars = torch.zeros_like(gp_outputs)
                for i, cov_layer in enumerate(self._gp_cov_layers):
                    gp_vars[:, i] = cov_layer(gp_features)

        return self.monte_carlo_sample_logits(
            logits=gp_outputs, vars=gp_vars, num_samples=self._num_mc_samples
        )  # [B, S, C]

    @staticmethod
    def _orthogonal_random_features_initializer(tensor: Tensor, std: float) -> Tensor:
        """Initializes orthogonal random features.

        Args:
            tensor: Tensor to initialize.
            std: Standard deviation.

        Returns:
            Initialized tensor.
        """
        num_rows, num_cols = tensor.shape
        if num_rows < num_cols:
            # When num_rows < num_cols, sample multiple (num_rows, num_rows) matrices
            # and then concatenate.
            ortho_mat_list = []
            num_cols_sampled = 0

            while num_cols_sampled < num_cols:
                matrix = torch.empty_like(tensor[:, :num_rows])
                ortho_mat_square = nn.init.orthogonal_(matrix, gain=std)
                ortho_mat_list.append(ortho_mat_square)
                num_cols_sampled += num_rows

            # Reshape the matrix to the target shape (num_rows, num_cols)
            ortho_mat = torch.cat(ortho_mat_list, dim=-1)
            ortho_mat = ortho_mat[:, :num_cols]
        else:
            matrix = torch.empty_like(tensor)
            ortho_mat = nn.init.orthogonal_(matrix, gain=std)

        # Sample random feature norms.
        # Construct Monte-Carlo estimate of squared column norm of a random
        # Gaussian matrix.
        feature_norms_square = torch.randn_like(ortho_mat) ** 2
        feature_norms = feature_norms_square.sum(dim=0).sqrt()

        # Sets a random feature matrix with orthogonal columns and Gaussian-like
        # column norms.
        value = ortho_mat * feature_norms
        with torch.no_grad():
            tensor.copy_(value)

        return tensor

    def _make_random_feature_layer(self, num_features: int) -> nn.Module:
        """Creates a random feature layer.

        Args:
            num_features: Number of input features.

        Returns:
            Random feature layer.
        """
        # Use user-supplied configurations.
        custom_random_feature_layer = nn.Linear(
            in_features=num_features,
            out_features=self._num_random_features,
        )
        self._random_features_weight_initializer(custom_random_feature_layer.weight)
        self._random_features_bias_initializer(custom_random_feature_layer.bias)
        custom_random_feature_layer.weight.requires_grad_(False)
        custom_random_feature_layer.bias.requires_grad_(False)

        return custom_random_feature_layer


class LaplaceRandomFeatureCovariance(nn.Module):
    """Empirical covariance matrix for random feature GPs.

    This module computes and maintains the covariance matrix for random feature
    Gaussian Processes.

    Args:
        gp_feature_dim: Dimension of GP features.
        momentum: Momentum for updating precision matrix.
        ridge_penalty: Ridge penalty for covariance matrix.
    """

    def __init__(
        self,
        gp_feature_dim: int,
        momentum: float,
        ridge_penalty: float,
    ) -> None:
        super().__init__()
        self._ridge_penalty = ridge_penalty
        self._momentum = momentum

        # Posterior precision matrix for the GP's random feature coefficients
        precision_matrix = torch.zeros((gp_feature_dim, gp_feature_dim))
        self.register_buffer("_precision_matrix", precision_matrix)
        covariance_matrix = torch.eye(gp_feature_dim)
        self.register_buffer("_covariance_matrix", covariance_matrix)
        self._gp_feature_dim = gp_feature_dim

        # Boolean flag to indicate whether to update the covariance matrix (i.e.,
        # by inverting the newly updated precision matrix) during inference.
        self._update_covariance = False

    def reset_precision_matrix(self) -> None:
        """Resets precision matrix to its initial value."""
        gp_feature_dim = self._precision_matrix.shape[0]
        self._precision_matrix.copy_(torch.zeros((gp_feature_dim, gp_feature_dim)))

    def forward(self, gp_features: Tensor) -> Tensor:
        """Computes GP posterior predictive variance.

        Args:
            gp_features: GP features.

        Returns:
            GP posterior predictive variance.
        """
        # Lazily computes feature covariance matrix during inference.
        covariance_matrix_updated = self._update_feature_covariance_matrix()

        # Store updated covariance matrix.
        self._covariance_matrix.copy_(covariance_matrix_updated)

        # Disable covariance update in future inference calls (to avoid the
        # expensive torch.linalg.inv op) unless there are new update to precision
        # matrix.
        self._update_covariance = False

        gp_var = self._compute_predictive_variance(gp_features)

        return gp_var

    @torch.no_grad()
    def update(self, gp_features: Tensor, multiplier: float = 1) -> None:
        """Updates the feature precision matrix.

        Args:
            gp_features: GP features.
            multiplier: Multiplier for the update.
        """
        # Computes the updated feature precision matrix.
        precision_matrix_updated = self._update_feature_precision_matrix(
            gp_features=gp_features,
            multiplier=multiplier,
        )

        # Updates precision matrix.
        self._precision_matrix.copy_(precision_matrix_updated)

        # Enables covariance update in the next inference call.
        self._update_covariance = True

    def _update_feature_precision_matrix(
        self, gp_features: Tensor, multiplier: float
    ) -> Tensor:
        """Computes the updated precision matrix of feature weights.

        Args:
            gp_features: GP features.
            multiplier: Multiplier for the update.

        Returns:
            Updated precision matrix.
        """
        batch_size = gp_features.shape[0]

        # Computes batch-specific normalized precision matrix.
        precision_matrix_minibatch = (multiplier * gp_features.T) @ gp_features

        # Updates the population-wise precision matrix.
        if self._momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch /= batch_size
            precision_matrix_new = (
                self._momentum * self._precision_matrix
                + (1 - self._momentum) * precision_matrix_minibatch
            )
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self._precision_matrix + precision_matrix_minibatch

        return precision_matrix_new

    def _update_feature_covariance_matrix(self) -> Tensor:
        """Computes the feature covariance matrix.

        Returns:
            Updated covariance matrix.
        """
        precision_matrix = self._precision_matrix
        covariance_matrix = self._covariance_matrix

        # Compute covariance matrix update only when `update_covariance = True`.
        if self._update_covariance:
            covariance_matrix_updated = torch.linalg.inv(
                self._ridge_penalty
                * torch.eye(self._gp_feature_dim, device=precision_matrix.device)
                + precision_matrix
            )
        else:
            covariance_matrix_updated = covariance_matrix

        return covariance_matrix_updated

    def _compute_predictive_variance(self, gp_feature: Tensor) -> Tensor:
        """Computes posterior predictive variance.

        Args:
            gp_feature: GP features.

        Returns:
            Predictive covariance matrix.
        """
        # Computes the variance of the posterior gp prediction.
        gp_var = torch.einsum(
            "ij,jk,ik->i",
            gp_feature,
            self._covariance_matrix,
            gp_feature,
        )

        return gp_var


class LinearSpectralNormalizer(nn.Module):
    """Module that augments Linear modules with spectral normalization.

    This module applies spectral normalization to Linear layers.

    Args:
        module: The Linear module to be normalized.
        spectral_normalization_iteration: Number of power iterations.
        spectral_normalization_bound: Upper bound for spectral norm.
        dim: Dimension along which to normalize.
        eps: Small value for numerical stability.
    """

    def __init__(
        self,
        module: nn.Module,
        spectral_normalization_iteration: int,
        spectral_normalization_bound: float,
        dim: int,
        eps: float,
    ) -> None:
        super().__init__()

        weight = module.weight
        ndim = weight.ndim

        self._spectral_normalization_iteration = spectral_normalization_iteration
        self._spectral_normalization_bound = spectral_normalization_bound
        self._dim = dim if dim >= 0 else dim + ndim
        self._eps = eps

        if ndim > 1:
            weight_matrix = self._reshape_weight_to_matrix(weight)
            height, width = weight_matrix.shape

            u = weight_matrix.new_empty(height).normal_(mean=0, std=1)
            v = weight_matrix.new_empty(width).normal_(mean=0, std=1)
            self.register_buffer("_u", F.normalize(u, dim=0, eps=self._eps))
            self.register_buffer("_v", F.normalize(v, dim=0, eps=self._eps))

            self._power_method(
                weight_matrix=weight_matrix, spectral_normalization_iteration=15
            )

    def forward(self, weight: Tensor) -> Tensor:
        """Applies spectral normalization to the input weight.

        Args:
            weight: Input weight tensor.

        Returns:
            Spectrally normalized weight tensor.
        """
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            weight_norm = weight.norm(p=2, dim=0).clamp_min(self._eps)
            division_factor = torch.max(
                torch.ones_like(weight_norm),
                weight_norm / self._spectral_normalization_bound,
            )

            return weight / division_factor
        weight_matrix = self._reshape_weight_to_matrix(weight=weight)

        if self.training:
            self._power_method(
                weight_matrix=weight_matrix,
                spectral_normalization_iteration=self._spectral_normalization_iteration,
            )

        # See above on why we need to clone
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)

        sigma = u @ weight_matrix @ v
        division_factor = torch.max(
            torch.ones_like(sigma), sigma / self._spectral_normalization_bound
        )

        return weight / division_factor

    @staticmethod
    def right_inverse(value: Tensor) -> Tensor:
        """Computes the right inverse of the spectral normalization.

        Args:
            value: Input tensor.

        Returns:
            Right inverse of the input.
        """
        return value

    def _reshape_weight_to_matrix(self, weight: Tensor) -> Tensor:
        """Reshapes the weight tensor to a matrix.

        Args:
            weight: Input weight tensor.

        Returns:
            Reshaped weight matrix.
        """
        if self._dim > 0:
            # Permute self._dim to front
            weight = weight.permute(
                self._dim, *(dim for dim in range(weight.dim()) if dim != self._dim)
            )

        weight_matrix = weight.flatten(start_dim=1)

        return weight_matrix

    @torch.no_grad()
    def _power_method(
        self, weight_matrix: Tensor, spectral_normalization_iteration: int
    ) -> None:
        """Applies the power method to estimate singular vectors.

        Args:
            weight_matrix: Weight matrix.
            spectral_normalization_iteration: Number of power iterations.
        """
        # See original note at torch/nn/utils/spectral_norm.py
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallelized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.

        # Precondition
        if weight_matrix.ndim != 2:
            msg = "Invalid matrix dimensionality"
            raise ValueError(msg)

        for _ in range(spectral_normalization_iteration):
            # u^\top W v = u^\top (\sigma u) = \sigma u^\top u = \sigma
            # where u and v are the first left and right (unit) singular vectors,
            # respectively. This power iteration produces approximations of u and v.
            self._u = F.normalize(
                weight_matrix @ self._v, dim=0, eps=self._eps, out=self._u
            )
            self._v = F.normalize(
                weight_matrix.T @ self._u, dim=0, eps=self._eps, out=self._v
            )


class Conv2dSpectralNormalizer(nn.Module):
    """Module that augments Conv2d modules with spectral normalization.

    This module applies spectral normalization to Conv2d layers.

    Args:
        module: The Conv2d module to be normalized.
        spectral_normalization_iteration: Number of power iterations.
        spectral_normalization_bound: Upper bound for spectral norm.
        eps: Small value for numerical stability.
    """

    def __init__(
        self,
        module: nn.Module,
        spectral_normalization_iteration: int,
        spectral_normalization_bound: float,
        eps: float,
    ) -> None:
        super().__init__()

        if (
            hasattr(module, "parametrizations")
            and hasattr(module.parametrizations, "weight")
            and Conv2dSpectralNormalizer in module.parametrizations.weight
        ):
            msg = "Cannot register spectral normalization more than once"
            raise ValueError(msg)

        weight = module.weight
        ndim = weight.ndim

        if ndim != 4:
            msg = f"Invalid weight shape: expected ndim = 4, received ndim = {ndim}"
            raise ValueError(msg)

        self._spectral_normalization_iteration = spectral_normalization_iteration
        self._spectral_normalization_bound = spectral_normalization_bound
        self._eps = eps

        self.stride = module.stride
        self.dilation = module.dilation
        self._groups = module.groups
        self.output_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.weight_shape = weight.shape

        self.register_buffer("_u", nn.UninitializedBuffer())
        self.register_buffer("_v", nn.UninitializedBuffer())

        self.load_hook = self._register_load_state_dict_pre_hook(self._lazy_load_hook)
        self.module_input_shape_hook = module.register_forward_pre_hook(
            Conv2dSpectralNormalizer._module_set_input_shape, with_kwargs=True
        )
        self.initialize_hook = self.register_forward_pre_hook(
            Conv2dSpectralNormalizer._infer_attributes, with_kwargs=True
        )

    def forward(self, weight: Tensor) -> Tensor:
        """Applies spectral normalization to the input weight.

        Args:
            weight: Input weight tensor.

        Returns:
            Spectrally normalized weight tensor.
        """
        if self.training:
            self._power_method(
                weight=weight,
                spectral_normalization_iteration=self._spectral_normalization_iteration,
            )

        # See above on why we need to clone
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)

        # Pad v to have the "same" padding effect
        v_padded = F.pad(
            v.view(self.single_input_shape), self.left_right_top_bottom_padding
        )

        # Apply the _unnormalized_ weight to v
        weight_v = F.conv2d(
            input=v_padded,
            weight=weight,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            groups=self._groups,
        )

        # Estimate largest singular value
        sigma = weight_v.contiguous().view(-1) @ u.view(-1)

        # Calculate factor to divide weight by; pay attention to numerical stability
        division_factor = torch.max(
            torch.ones_like(sigma), sigma / self._spectral_normalization_bound
        ).clamp_min(self._eps)

        return weight / division_factor

    def initialize_buffers(self, weight: Tensor) -> None:
        """Initializes buffers for spectral normalization.

        Args:
            weight: Weight tensor to initialize buffers for.
        """
        if self.has_uninitialized_buffers():
            with torch.no_grad():
                flattened_input_shape = math.prod(self.single_input_shape)
                flattened_output_shape = math.prod(self.single_output_shape)

                device = weight.device

                # Materialize buffers
                self._u.materialize(shape=flattened_output_shape, device=device)
                self._v.materialize(shape=flattened_input_shape, device=device)

                # Initialize buffers randomly
                nn.init.normal_(self._u)
                nn.init.normal_(self._v)

                # Initialize buffers with correct values. We do 50 iterations to have
                # a good approximation of the correct singular vectors and the value
                self._power_method(weight=weight, spectral_normalization_iteration=50)

    def has_uninitialized_buffers(self) -> bool:
        """Checks if there are uninitialized buffers.

        Returns:
            True if there are uninitialized buffers, False otherwise.
        """
        buffers = self._buffers.values()
        return any(is_lazy(buffer) for buffer in buffers)

    @staticmethod
    def right_inverse(value: Tensor) -> Tensor:
        """Computes the right inverse of the parametrization.

        Args:
            value: Input tensor.

        Returns:
            Right inverse of the input.
        """
        return value

    @staticmethod
    def _module_set_input_shape(
        module: nn.Module, args: tuple, kwargs: dict | None = None
    ) -> None:
        """Sets input shape for the module.

        Args:
            module: Module to set input shape for.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        kwargs = kwargs or {}
        input: Tensor = kwargs["input"] if "input" in kwargs else args[0]

        for parametrization in module.parametrizations.weight:
            if isinstance(parametrization, Conv2dSpectralNormalizer):
                input_channels, input_height, input_width = input.shape[1:]
                parametrization.single_input_shape = (
                    1,
                    input_channels,
                    input_height,
                    input_width,
                )

                # Infer output shape with batch size = 1. We know this without having
                # to run the Conv2d module, as we use "same" padding in our internal
                # calculations
                output_channels = parametrization.output_channels
                output_height = input_height // parametrization.stride[0]
                output_width = input_width // parametrization.stride[1]
                parametrization.single_output_shape = (
                    1,
                    output_channels,
                    output_height,
                    output_width,
                )

                # Infer input padding
                parametrization.left_right_top_bottom_padding = calculate_same_padding(
                    parametrization.single_output_shape,
                    parametrization.weight_shape,
                    parametrization.stride,
                    parametrization.dilation,
                )
                total_width_height_padding = (
                    parametrization.left_right_top_bottom_padding[0]
                    + parametrization.left_right_top_bottom_padding[1],
                    parametrization.left_right_top_bottom_padding[2]
                    + parametrization.left_right_top_bottom_padding[3],
                )
                parametrization.per_side_width_height_padding = (
                    math.ceil(total_width_height_padding[0] / 2),
                    math.ceil(total_width_height_padding[1] / 2),
                )

                # Infer output padding
                parametrization.output_padding = calculate_output_padding(
                    input_shape=parametrization.single_output_shape,
                    output_shape=parametrization.single_input_shape,
                    stride=parametrization.stride,
                    padding=parametrization.per_side_width_height_padding,
                    kernel_size=parametrization.kernel_size,
                    dilation=parametrization.dilation,
                )

                # Invariant: there is only one Conv2dSpectralNormalizer registered
                break

    def _save_to_state_dict(
        self, destination: dict, prefix: str, keep_vars: bool
    ) -> None:
        """Saves the module state to a dictionary.

        Args:
            destination: Destination dictionary.
            prefix: Prefix for parameter names.
            keep_vars: Whether to keep variables.
        """
        # This should be ideally implemented as a hook,
        # but we should override `detach` in the UninitializedParameter to return itself
        # which is not clean
        for name, param in self._parameters.items():
            if param is not None:
                if not (is_lazy(param) or keep_vars):
                    param = param.detach()
                destination[prefix + name] = param
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                if not (is_lazy(buf) or keep_vars):
                    buf = buf.detach()
                destination[prefix + name] = buf

    def _lazy_load_hook(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Hook for lazy loading of state dict.

        Args:
            state_dict: State dictionary.
            prefix: Prefix for parameter names.
            local_metadata: Local metadata.
            strict: Whether to strictly enforce that the keys match.
            missing_keys: List to store missing keys.
            unexpected_keys: List to store unexpected keys.
            error_msgs: List to store error messages.
        """
        del local_metadata, strict, missing_keys, unexpected_keys, error_msgs

        for name, param in itertools.chain(
            self._parameters.items(), self._buffers.items()
        ):
            key = prefix + name
            if key in state_dict and param is not None:
                input_param = state_dict[key]
                if is_lazy(param) and not is_lazy(input_param):
                    # The current parameter is not initialized but the one being loaded
                    # is
                    # Create a new parameter based on the uninitialized one
                    with torch.no_grad():
                        param.materialize(input_param.shape)

    @staticmethod
    def _infer_attributes(
        module: nn.Module, args: tuple, kwargs: dict | None = None
    ) -> None:
        """Infers attributes for the module.

        Args:
            module: Module to infer attributes for.
            args: Positional arguments.
            kwargs: Keyword arguments.
        """
        # Infer buffers
        kwargs = kwargs or {}
        module.initialize_buffers(*args, **kwargs)
        if module.has_uninitialized_buffers():
            msg = f"Module {module.__class__.__name__} has not been fully initialized"
            raise RuntimeError(msg)

        # Remove hooks
        module.load_hook.remove()
        module.module_input_shape_hook.remove()
        module.initialize_hook.remove()
        delattr(module, "load_hook")
        delattr(module, "module_input_shape_hook")
        delattr(module, "initialize_hook")

    def _replicate_for_data_parallel(self) -> "Conv2dSpectralNormalizer":
        """Replicates the module for data parallel processing."""
        if self.has_uninitialized_buffers():
            msg = (
                "Modules with uninitialized parameters can't be used with "
                "`DataParallel`. Run a dummy forward pass to correctly initialize the "
                "modules"
            )
            raise RuntimeError(msg)

        return super()._replicate_for_data_parallel()

    @torch.no_grad()
    def _power_method(
        self, weight: Tensor, spectral_normalization_iteration: int
    ) -> None:
        """Applies the power method to estimate singular vectors.

        Args:
            weight: Weight tensor.
            spectral_normalization_iteration: Number of power iterations.
        """
        # See original note at torch/nn/utils/spectral_norm.py
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallelized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.

        for _ in range(spectral_normalization_iteration):
            # u^\top W v = u^\top (\sigma u) = \sigma u^\top u = \sigma
            # where u and v are the first left and right (unit) singular vectors,
            # respectively. This power iteration produces approximations of u and v.

            v_shaped = F.conv_transpose2d(
                input=self._u.view(self.single_output_shape),
                weight=weight,
                bias=None,
                stride=self.stride,
                padding=self.per_side_width_height_padding,
                output_padding=self.output_padding,
                groups=self._groups,
                dilation=self.dilation,
            )

            self._v = F.normalize(
                input=v_shaped.contiguous().view(-1), dim=0, eps=self._eps, out=self._v
            )

            v_padded = F.pad(
                self._v.view(self.single_input_shape),
                self.left_right_top_bottom_padding,
            )
            u_shaped = F.conv2d(
                input=v_padded,
                weight=weight,
                bias=None,
                stride=self.stride,
                dilation=self.dilation,
                groups=self._groups,
            )

            self._u = F.normalize(
                input=u_shaped.contiguous().view(-1), dim=0, eps=self._eps, out=self._u
            )


class _SpectralNormalizedBatchNorm(_NormBase):
    """Base class for spectral normalized batch normalization layers.

    Args:
        num_features: Number of features.
        spectral_normalization_bound: Upper bound for spectral norm.
        eps: Small value for numerical stability.
        momentum: Momentum for running statistics.
        device: Device to use.
        dtype: Data type.
        affine: Whether to use learnable affine parameters.
        track_running_stats: Whether to track running statistics.
    """

    def __init__(
        self,
        num_features: int,
        spectral_normalization_bound: float,
        eps: float = 1e-5,
        momentum: float = 0.01,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        affine: bool = True,
    ) -> None:
        # Momentum is 0.01 by default instead of 0.1 of BN which alleviates noisy power
        # iteration. Code is based on torch.nn.modules._NormBase

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=True,
            **factory_kwargs,
        )
        self._spectral_normalization_bound = spectral_normalization_bound

    def forward(self, input: Tensor) -> Tensor:
        """Applies spectral normalized batch normalization.

        Args:
            input: Input tensor.

        Returns:
            Normalized output tensor.
        """
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum

        if self.training:  # noqa: SIM102
            # If statement only here to tell the jit to skip emitting this when it
            # is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # Use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # Use exponential moving average
                    exponential_average_factor = self.momentum

        # Buffers are only updated if they are to be tracked and we are in training
        # mode. Thus they only need to be passed when the update should occur (i.e. in
        # training mode when they are tracked), or when buffer stats are used for
        # normalization (i.e. in eval mode when buffers are not None)

        # Before the forward pass, estimate the Lipschitz constant of the layer and
        # divide through by it so that the Lipschitz constant of the batch norm operator
        # is at most self.coeff
        weight = self.weight

        if weight is not None:
            # See https://arxiv.org/pdf/1804.04368.pdf,
            # equation 28 for why this is correct
            lipschitz = torch.max(
                torch.abs(weight * (self.running_var + self.eps) ** -0.5)
            )

            # If the Lipschitz constant of the operation is greater than coeff, then we
            # want to divide the input by a constant to force the overall Lipchitz
            # factor of the batch norm to be exactly coeff
            lipschitz_factor = torch.max(
                lipschitz / self._spectral_normalization_bound,
                torch.ones_like(lipschitz),
            )

            weight /= lipschitz_factor

        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            weight,
            self.bias,
            self.training,
            exponential_average_factor,
            self.eps,
        )


class SpectralNormalizedBatchNorm2d(_SpectralNormalizedBatchNorm):
    """Spectral normalized BatchNorm2d module.

    This module applies spectral normalization to BatchNorm2d layers.

    Args:
        module: The BatchNorm2d module to be normalized.
        spectral_normalization_bound: Upper bound for spectral norm.
    """

    def __init__(self, module: nn.Module, spectral_normalization_bound: float) -> None:
        if not module.track_running_stats:
            msg = (
                f"track_running_stats=False is not supported with {type(self).__name__}"
            )
            raise ValueError(msg)

        super().__init__(
            module.num_features,
            spectral_normalization_bound,
            module.eps,
            module.momentum,
            module.affine,
        )

    @staticmethod
    def _check_input_dim(input: Tensor) -> None:
        """Checks the input dimension.

        Args:
            input: Input tensor.

        Raises:
            ValueError: If input dimension is not 4.
        """
        if input.dim() != 4:
            msg = f"Expected 4D input (got {input.dim()}D input)"
            raise ValueError(msg)


class SNGPWrapper(DistributionalWrapper):
    """Wrapper that creates an SNGP from an input model.

    Args:
        model: Base model to wrap.
        use_spectral_normalization: Whether to use spectral normalization.
        use_tight_norm_for_pointwise_convs: Whether to use tight norm for pointwise
            convolutions.
        spectral_normalization_iteration: Number of power iterations for spectral
            normalization.
        spectral_normalization_bound: Upper bound for spectral norm.
        use_spectral_normalized_batch_norm: Whether to use spectral normalized batch
            norm.
        num_mc_samples: Number of Monte Carlo samples.
        num_random_features: Number of random features for GP.
        gp_kernel_scale: Scale of the GP kernel.
        gp_output_bias: Output bias for GP.
        gp_random_feature_type: Type of random features for GP.
        use_input_normalized_gp: Whether to use input normalization for GP.
        gp_cov_momentum: Momentum for GP covariance update.
        gp_cov_ridge_penalty: Ridge penalty for GP covariance.
        gp_input_dim: Input dimension for GP.
        likelihood: Likelihood type for GP.
    """

    def __init__(
        self,
        model: nn.Module,
        use_spectral_normalization: bool,
        use_tight_norm_for_pointwise_convs: bool,
        spectral_normalization_iteration: int,
        spectral_normalization_bound: float,
        use_spectral_normalized_batch_norm: bool,
        num_mc_samples: int,
        num_random_features: int,
        gp_kernel_scale: float,
        gp_output_bias: float,
        gp_random_feature_type: str,
        use_input_normalized_gp: bool,
        gp_cov_momentum: float,
        gp_cov_ridge_penalty: float,
        gp_input_dim: int,
        likelihood: str,
    ) -> None:
        super().__init__(model)

        self._num_mc_samples = num_mc_samples
        self._num_random_features = num_random_features
        self._gp_kernel_scale = gp_kernel_scale
        self._gp_output_bias = gp_output_bias
        self._gp_random_feature_type = gp_random_feature_type
        self._use_input_normalized_gp = use_input_normalized_gp
        self._gp_cov_momentum = gp_cov_momentum
        self._gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self._gp_input_dim = gp_input_dim

        if likelihood not in {"gaussian", "softmax"}:
            msg = f"Invalid likelihood '{likelihood}' provided"
            raise ValueError(msg)

        self._likelihood = likelihood

        classifier = nn.Sequential()

        if self._gp_input_dim > 0:
            random_projection = nn.Linear(
                in_features=self.num_features,
                out_features=self._gp_input_dim,
                bias=False,
            )
            nn.init.normal_(random_projection.weight, mean=0, std=0.05)
            random_projection.weight.requires_grad_(False)
            num_gp_features = self._gp_input_dim

            classifier.append(random_projection)
        else:
            num_gp_features = self.num_features

        gp_output_layer = GPOutputLayer(
            num_features=num_gp_features,
            num_classes=self.num_classes,
            num_mc_samples=self._num_mc_samples,
            num_random_features=self._num_random_features,
            gp_kernel_scale=self._gp_kernel_scale,
            gp_output_bias=self._gp_output_bias,
            gp_random_feature_type=self._gp_random_feature_type,
            use_input_normalized_gp=self._use_input_normalized_gp,
            gp_cov_momentum=self._gp_cov_momentum,
            gp_cov_ridge_penalty=self._gp_cov_ridge_penalty,
            likelihood=self._likelihood,
        )
        classifier.append(gp_output_layer)

        self._classifier = classifier

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

    def get_classifier(self) -> nn.Sequential:
        """Returns the classifier of the SNGP.

        Returns:
            The classifier module.
        """
        return self._classifier

    def reset_classifier(
        self,
        num_mc_samples: int | None = None,
        num_random_features: int | None = None,
        gp_kernel_scale: float | None = None,
        gp_output_bias: float | None = None,
        gp_random_feature_type: str | None = None,
        use_input_normalized_gp: bool | None = None,
        gp_cov_momentum: float | None = None,
        gp_cov_ridge_penalty: float | None = None,
        gp_input_dim: int | None = None,
        likelihood: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Resets the classifier with new parameters.

        Args:
            num_mc_samples: New number of Monte Carlo samples.
            num_random_features: New number of random features.
            gp_kernel_scale: New GP kernel scale.
            gp_output_bias: New GP output bias.
            gp_random_feature_type: New GP random feature type.
            use_input_normalized_gp: Whether to use input normalized GP.
            gp_cov_momentum: New GP covariance momentum.
            gp_cov_ridge_penalty: New GP covariance ridge penalty.
            gp_input_dim: New GP input dimension.
            likelihood: New likelihood type.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if num_mc_samples is not None:
            self._num_mc_samples = num_mc_samples

        if num_random_features is not None:
            self._num_random_features = num_random_features

        if gp_kernel_scale is not None:
            self._gp_kernel_scale = gp_kernel_scale

        if gp_output_bias is not None:
            self._gp_output_bias = gp_output_bias

        if gp_random_feature_type is not None:
            self._gp_random_feature_type = gp_random_feature_type

        if use_input_normalized_gp is not None:
            self._use_input_normalized_gp = use_input_normalized_gp

        if gp_cov_momentum is not None:
            self._gp_cov_momentum = gp_cov_momentum

        if gp_cov_ridge_penalty is not None:
            self._gp_cov_ridge_penalty = gp_cov_ridge_penalty

        if gp_input_dim is not None:
            self._gp_input_dim = gp_input_dim

        if likelihood is not None:
            self._likelihood = likelihood

        # Resets global pooling in `self.classifier`
        self.model.reset_classifier(*args, **kwargs)
        classifier = nn.Sequential()

        if self._gp_input_dim > 0:
            random_projection = nn.Linear(
                in_features=self.num_features,
                out_features=self._gp_input_dim,
                bias=False,
            )
            nn.init.normal_(random_projection.weight, mean=0, std=0.05)
            random_projection.weight.requires_grad_(False)
            num_gp_features = self._gp_input_dim

            classifier.append(random_projection)
        else:
            num_gp_features = self.num_features

        gp_output_layer = GPOutputLayer(
            num_features=num_gp_features,
            num_classes=self.num_classes,
            num_mc_samples=self._num_mc_samples,
            num_random_features=self._num_random_features,
            gp_kernel_scale=self._gp_kernel_scale,
            gp_output_bias=self._gp_output_bias,
            gp_random_feature_type=self._gp_random_feature_type,
            use_input_normalized_gp=self._use_input_normalized_gp,
            gp_cov_momentum=self._gp_cov_momentum,
            gp_cov_ridge_penalty=self._gp_cov_ridge_penalty,
            likelihood=self._likelihood,
        )
        classifier.append(gp_output_layer)

        self._classifier = classifier

    def reset_covariance_matrix(self) -> None:
        """Resets covariance matrix of the GP layer."""
        self._classifier[-1].reset_covariance_matrix()
