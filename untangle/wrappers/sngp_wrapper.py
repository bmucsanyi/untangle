"""SNGP/GP implementation as a wrapper class.

SNGP implementation based on https://github.com/google/edward2
"""

import itertools
import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _NormBase  # noqa: PLC2701
from torch.nn.parameter import is_lazy

from untangle.utils import calculate_output_padding, calculate_same_padding
from untangle.utils.predictive import (
    diag_hessian_normalized_normcdf,
    diag_hessian_normalized_sigmoid,
    diag_hessian_softmax,
)
from untangle.utils.replace import register, register_cond, replace
from untangle.wrappers.model_wrapper import DistributionalWrapper

LIKELIHOOD_TO_HESSIAN_DIAG = {
    "softmax": diag_hessian_softmax,
    "sigmoid": diag_hessian_normalized_sigmoid,
    "normcdf": diag_hessian_normalized_normcdf,
}


class GPOutputLayer(nn.Module):
    """Random feature GP output layer."""

    def __init__(
        self,
        num_features,
        num_classes,
        num_mc_samples,
        num_random_features,
        gp_kernel_scale,
        gp_output_bias,
        gp_random_feature_type,
        use_input_normalized_gp,
        gp_cov_momentum,
        gp_cov_ridge_penalty,
        likelihood,
    ):
        super().__init__()
        self._num_classes = num_classes
        self._num_random_features = num_random_features
        self._num_mc_samples = num_mc_samples

        self.use_input_normalized_gp = use_input_normalized_gp
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

        if self.use_input_normalized_gp:
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

    def reset_covariance_matrix(self):
        """Resets covariance matrix of the GP layer.

        This function is useful for resetting the model's covariance matrix at the
        beginning of a new epoch.
        """
        for cov_layer in self._gp_cov_layers:
            cov_layer.reset_precision_matrix()

    def forward(self, gp_inputs, targets=None):
        # Computes random features.
        if self.use_input_normalized_gp:
            gp_inputs = self._input_norm_layer(gp_inputs)
        elif self._gp_input_scale is not None:
            # Supports lengthscale for custom random feature layer by directly
            # rescaling the input.
            gp_inputs *= self._gp_input_scale

        gp_features = self._random_feature(gp_inputs).cos()

        # Computes posterior center (i.e., MAP estimate) and variance
        # TODO(bmucsanyi): https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L260
        gp_outputs = self._gp_output_layer(gp_features) + self._gp_output_bias  # [B, C]

        if self.training:
            if targets is None:
                msg = "`targets` must be provided during training"
                raise ValueError(msg)

            if self._likelihood == "gaussian":
                self._gp_cov_layers[0].update(gp_features)
            else:
                multipliers = LIKELIHOOD_TO_HESSIAN_DIAG[self._likelihood](
                    gp_outputs, targets
                )  # [B, C]

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
            gp_vars = torch.zeros_like(gp_outputs)
            for i, cov_layer in enumerate(self._gp_cov_layers):
                gp_vars[:, i] = cov_layer(gp_features)

        return gp_outputs, gp_vars

    @staticmethod
    def _orthogonal_random_features_initializer(tensor, std):
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

    def _make_random_feature_layer(self, num_features):
        """Defines random feature layer depending on kernel type."""
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
    """Empirical covariance matrix for random feature GPs."""

    def __init__(
        self,
        gp_feature_dim,
        momentum,
        ridge_penalty,
    ):
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
        self.update_covariance = False

    def reset_precision_matrix(self):
        """Resets precision matrix to its initial value.

        This function is useful for resetting the model's covariance matrix at the
        beginning of a new epoch.
        """
        gp_feature_dim = self._precision_matrix.shape[0]
        self._precision_matrix.copy_(torch.zeros((gp_feature_dim, gp_feature_dim)))

    def forward(self, gp_features):
        """Minibatch updates the GP's posterior precision matrix estimate.

        Args:
        gp_features: (torch.Tensor) Pre-activation output from the model. Needed
            for Laplace approximation under a non-Gaussian likelihood.

        Returns:
        gp_var (torch.Tensor): GP posterior predictive variance,
            shape (batch_size, batch_size).
        """
        # Lazily computes feature covariance matrix during inference.
        covariance_matrix_updated = self._update_feature_covariance_matrix()

        # Store updated covariance matrix.
        self._covariance_matrix.copy_(covariance_matrix_updated)

        # Disable covariance update in future inference calls (to avoid the
        # expensive torch.linalg.inv op) unless there are new update to precision
        # matrix.
        self.update_covariance = False

        gp_var = self._compute_predictive_variance(gp_features)

        return gp_var

    @torch.no_grad()
    def update(self, gp_features, multiplier=1):  # [B, C], [B]
        # Computes the updated feature precision matrix.
        precision_matrix_updated = self._update_feature_precision_matrix(
            gp_features=gp_features,
            multiplier=multiplier,
        )

        # Updates precision matrix.
        self._precision_matrix.copy_(precision_matrix_updated)

        # Enables covariance update in the next inference call.
        self.update_covariance = True

    def _update_feature_precision_matrix(self, gp_features, multiplier):
        """Computes the update precision matrix of feature weights."""
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

    def _update_feature_covariance_matrix(self):
        """Computes the feature covariance if self.update_covariance=True.

        GP layer computes the covariance matrix of the random feature coefficient
        by inverting the precision matrix. Since this inversion op is expensive,
        we will invoke it only when there is new update to the precision matrix
        (where self.update_covariance will be flipped to `True`.).

        Returns:
        The updated covariance_matrix.
        """
        precision_matrix = self._precision_matrix
        covariance_matrix = self._covariance_matrix

        # Compute covariance matrix update only when `update_covariance = True`.
        if self.update_covariance:
            covariance_matrix_updated = torch.linalg.inv(
                self._ridge_penalty
                * torch.eye(self._gp_feature_dim, device=precision_matrix.device)
                + precision_matrix
            )
        else:
            covariance_matrix_updated = covariance_matrix

        return covariance_matrix_updated

    def _compute_predictive_variance(self, gp_feature):
        """Computes posterior predictive variance.

        Approximates the Gaussian process posterior variance using random features.

        Args:
        gp_feature: (torch.Tensor) The random feature of testing data to be used for
            computing the covariance matrix. Shape (batch_size, gp_hidden_size).

        Returns:
        (torch.Tensor) Predictive covariance matrix, shape (batch_size, batch_size).
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
    """Module that augments Linear modules with spectral normalization."""

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

        self.spectral_normalization_iteration = spectral_normalization_iteration
        self.spectral_normalization_bound = spectral_normalization_bound
        self.dim = dim if dim >= 0 else dim + ndim
        self.eps = eps

        if ndim > 1:
            weight_matrix = self._reshape_weight_to_matrix(weight)
            height, width = weight_matrix.shape

            u = weight_matrix.new_empty(height).normal_(mean=0, std=1)
            v = weight_matrix.new_empty(width).normal_(mean=0, std=1)
            self.register_buffer("_u", F.normalize(u, dim=0, eps=self.eps))
            self.register_buffer("_v", F.normalize(v, dim=0, eps=self.eps))

            self.power_method(
                weight_matrix=weight_matrix, spectral_normalization_iteration=15
            )

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            weight_norm = weight.norm(p=2, dim=0).clamp_min(self.eps)
            division_factor = torch.max(
                torch.ones_like(weight_norm),
                weight_norm / self.spectral_normalization_bound,
            )

            return weight / division_factor
        weight_matrix = self._reshape_weight_to_matrix(weight=weight)

        if self.training:
            self._power_method(
                weight_matrix=weight_matrix,
                spectral_normalization_iteration=self.spectral_normalization_iteration,
            )

        # See above on why we need to clone
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)

        sigma = u @ weight_matrix @ v
        division_factor = torch.max(
            torch.ones_like(sigma), sigma / self.spectral_normalization_bound
        )

        return weight / division_factor

    @staticmethod
    def right_inverse(value: torch.Tensor) -> torch.Tensor:
        return value

    def _reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        if self.dim > 0:
            # Permute self.dim to front
            weight = weight.permute(
                self.dim, *(dim for dim in range(weight.dim()) if dim != self.dim)
            )

        weight_matrix = weight.flatten(start_dim=1)

        return weight_matrix

    @torch.no_grad()
    def power_method(
        self, weight_matrix: torch.Tensor, spectral_normalization_iteration: int
    ) -> None:
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
                weight_matrix @ self._v, dim=0, eps=self.eps, out=self._u
            )
            self._v = F.normalize(
                weight_matrix.T @ self._u, dim=0, eps=self.eps, out=self._v
            )


class Conv2dSpectralNormalizer(nn.Module):
    """Module that augments Conv2d modules with spectral normalization."""

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

        self.spectral_normalization_iteration = spectral_normalization_iteration
        self.spectral_normalization_bound = spectral_normalization_bound
        self.eps = eps

        self.stride = module.stride
        self.dilation = module.dilation
        self.groups = module.groups
        self.output_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.device = weight.device
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

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._power_method(
                weight=weight,
                spectral_normalization_iteration=self.spectral_normalization_iteration,
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
            groups=self.groups,
        )

        # Estimate largest singular value
        sigma = weight_v.view(-1) @ u.view(-1)

        # Calculate factor to divide weight by; pay attention to numerical stability
        division_factor = torch.max(
            torch.ones_like(sigma), sigma / self.spectral_normalization_bound
        ).clamp_min(self.eps)

        return weight / division_factor

    def initialize_buffers(self, weight) -> None:
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
        buffers = self._buffers.values()
        return any(is_lazy(buffer) for buffer in buffers)

    @staticmethod
    def right_inverse(value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def _module_set_input_shape(module, args, kwargs=None):
        kwargs = kwargs or {}
        input = kwargs["input"] if "input" in kwargs else args[0]

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

    def _save_to_state_dict(self, destination, prefix, keep_vars):
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
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """load_state_dict pre-hook function for lazy buffers and parameters.

        The purpose of this hook is to adjust the current state and/or
        ``state_dict`` being loaded so that a module instance serialized in
        both un/initialized state can be deserialized onto both un/initialized
        module instance.
        See comment in ``torch.nn.Module._register_load_state_dict_pre_hook``
        for the details of the hook specification.
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
    def _infer_attributes(module, args, kwargs=None):
        r"""Infers the size and initializes the parameters according to the input batch.

        Given a module that contains parameters that were declared inferable
        using :class:`torch.nn.parameter.ParameterMode.Infer`, runs a forward pass
        in the complete module using the provided input to initialize all the parameters
        as needed.
        The module is set into evaluation mode before running the forward pass in order
        to avoid saving statistics or calculating gradients
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

    def _replicate_for_data_parallel(self):
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
        self, weight: torch.Tensor, spectral_normalization_iteration: int
    ) -> None:
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

            # TODO(bmucsanyi): Possibly get rid of "same" padding?
            v_shaped = F.conv_transpose2d(
                input=self._u.view(self.single_output_shape),
                weight=weight,
                bias=None,
                stride=self.stride,
                padding=self.per_side_width_height_padding,
                output_padding=self.output_padding,
                groups=self.groups,
                dilation=self.dilation,
            )

            self._v = F.normalize(
                input=v_shaped.view(-1), dim=0, eps=self.eps, out=self._v
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
                groups=self.groups,
            )

            self._u = F.normalize(
                input=u_shaped.view(-1), dim=0, eps=self.eps, out=self._u
            )


class _SpectralNormalizedBatchNorm(_NormBase):
    def __init__(
        self,
        num_features: int,
        spectral_normalization_bound: float,
        eps: float = 1e-5,
        momentum: float = 0.01,
        device=None,
        dtype=None,
        *,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        # Momentum is 0.01 by default instead of 0.1 of BN which alleviates noisy power
        # iteration. Code is based on torch.nn.modules._NormBase

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self._spectral_normalization_bound = spectral_normalization_bound

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum

        if self.training and self.track_running_stats:  # noqa: SIM102
            # If statement only here to tell the jit to skip emitting this when it
            # is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # Use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # Use exponential moving average
                    exponential_average_factor = self.momentum

        # Decide whether the mini-batch stats should be used for normalization rather
        # than the buffers. Mini-batch stats are used in training mode, and in eval mode
        # when buffers are None
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # Buffers are only updated if they are to be tracked and we are in training
        # mode. Thus they only need to be passed when the update should occur (i.e. in
        # training mode when they are tracked), or when buffer stats are used for
        # normalization (i.e. in eval mode when buffers are not None)

        # Before the forward pass, estimate the Lipschitz constant of the layer and
        # divide through by it so that the Lipschitz constant of the batch norm operator
        # is at most self.coeff
        weight = (
            torch.ones_like(self.running_var) if self.weight is None else self.weight
        )
        # See https://arxiv.org/pdf/1804.04368.pdf, equation 28 for why this is correct
        lipschitz = torch.max(torch.abs(weight * (self.running_var + self.eps) ** -0.5))

        # If the Lipschitz constant of the operation is greater than coeff, then we want
        # to divide the input by a constant to force the overall Lipchitz factor of the
        # batch norm to be exactly coeff
        lipschitz_factor = torch.max(
            lipschitz / self._spectral_normalization_bound, torch.ones_like(lipschitz)
        )

        weight /= lipschitz_factor

        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            (
                self.running_mean
                if not self.training or self.track_running_stats
                else None
            ),
            self.running_var if not self.training or self.track_running_stats else None,
            weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class SpectralNormalizedBatchNorm2d(_SpectralNormalizedBatchNorm):
    """Spectral normalized BatchNorm2d module."""

    def __init__(self, module, spectral_normalization_bound: float) -> None:
        # TODO(bmucsanyi): Set bn-momentum to 0.01 if we use this!
        super().__init__(
            module.num_features,
            spectral_normalization_bound,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )

    @staticmethod
    def _check_input_dim(input):
        if input.dim() != 4:
            msg = f"Expected 4D input (got {input.dim()}D input)"
            raise ValueError(msg)


class SNGPWrapper(DistributionalWrapper):
    """This module takes a model as input and creates an SNGP from it."""

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
    ):
        super().__init__(model)

        self._num_mc_samples = num_mc_samples
        self._num_random_features = num_random_features
        self._gp_kernel_scale = gp_kernel_scale
        self._gp_output_bias = gp_output_bias
        self._gp_random_feature_type = gp_random_feature_type
        self.use_input_normalized_gp = use_input_normalized_gp
        self._gp_cov_momentum = gp_cov_momentum
        self._gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self._gp_input_dim = gp_input_dim

        if likelihood not in {"gaussian", "softmax", "sigmoid", "normcdf"}:
            msg = f"Invalid likelihood '{likelihood}' provided"
            raise ValueError(msg)

        self._likelihood = likelihood

        if self._gp_input_dim > 0:
            random_projection = nn.Linear(
                in_features=self.num_features,
                out_features=self._gp_input_dim,
                bias=False,
            )
            nn.init.normal_(random_projection.weight, mean=0, std=0.05)
            random_projection.weight.requires_grad_(False)
            num_gp_features = self._gp_input_dim

            self._random_projection = random_projection
        else:
            num_gp_features = self.num_features
            self._random_projection = lambda x: x

        gp_output_layer = GPOutputLayer(
            num_features=num_gp_features,
            num_classes=self.num_classes,
            num_mc_samples=self._num_mc_samples,
            num_random_features=self._num_random_features,
            gp_kernel_scale=self._gp_kernel_scale,
            gp_output_bias=self._gp_output_bias,
            gp_random_feature_type=self._gp_random_feature_type,
            use_input_normalized_gp=self.use_input_normalized_gp,
            gp_cov_momentum=self._gp_cov_momentum,
            gp_cov_ridge_penalty=self._gp_cov_ridge_penalty,
            likelihood=self._likelihood,
        )
        self._classifier = gp_output_layer

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

    def _classifier_fn(self, x, y=None):
        x = self._random_projection(x)
        x = self._classifier(x, y)

        return x

    def get_classifier(self):
        return self._classifier_fn

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
        *args,
        **kwargs,
    ):
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
            self.use_input_normalized_gp = use_input_normalized_gp

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

        if self._gp_input_dim > 0:
            random_projection = nn.Linear(
                in_features=self.num_features,
                out_features=self._gp_input_dim,
                bias=False,
            )
            nn.init.normal_(random_projection.weight, mean=0, std=0.05)
            random_projection.weight.requires_grad_(False)
            num_gp_features = self._gp_input_dim

            self._random_projection = random_projection
        else:
            num_gp_features = self.num_features
            self._random_projection = lambda x: x

        gp_output_layer = GPOutputLayer(
            num_features=num_gp_features,
            num_classes=self.num_classes,
            num_mc_samples=self._num_mc_samples,
            num_random_features=self._num_random_features,
            gp_kernel_scale=self._gp_kernel_scale,
            gp_output_bias=self._gp_output_bias,
            gp_random_feature_type=self._gp_random_feature_type,
            use_input_normalized_gp=self.use_input_normalized_gp,
            gp_cov_momentum=self._gp_cov_momentum,
            gp_cov_ridge_penalty=self._gp_cov_ridge_penalty,
            likelihood=self._likelihood,
        )

        self._classifier = gp_output_layer

    def reset_covariance_matrix(self):
        self._classifier.reset_covariance_matrix()

    def forward(self, x, y=None):
        x = self.forward_features(x)
        x = self.forward_head(x, y)

        return x

    def forward_head(self, x, y=None, *, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        out = self.get_classifier()(features, y)

        return out
