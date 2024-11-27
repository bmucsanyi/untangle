"""Fast deep ensemble wrapper class."""

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from untangle.wrappers.model_wrapper import DistributionalWrapper

logger = logging.getLogger(__name__)


class FastDeepEnsembleWrapper(DistributionalWrapper):
    """Wrapper that creates a fast deep ensemble from a set of input models.

    Args:
        model: The base model to wrap.
        weight_path: Path to the model weights.
        use_low_rank_cov: Whether to use low-rank plus diagonal covariance.
        max_rank: Maximum rank for the low-rank component.
    """

    def __init__(self, model: nn.Module, weight_paths: list[Path]) -> None:
        super().__init__(model)
        self._ensemble_params_device = torch.device("cpu")
        self._weight_paths = weight_paths
        self.num_models = len(weight_paths)
        num_params = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        self.register_buffer(
            "_model_params_static",
            torch.zeros((0, num_params), device=self._ensemble_params_device),
        )
        self._modules_and_names = []
        self.model.apply(self._add_ensemble_params)
        self._populate_ensemble_params()
        self._set_model(model_index=0)

    def forward(
        self, input: Tensor, amp_autocast: Any | None = None
    ) -> dict[str, Tensor] | Tensor:
        """Performs forward pass through the deep ensemble.

        Args:
            input: Input tensor.
            amp_autocast: Automatic mixed precision autocast.

        Returns:
            During training, returns logits. During inference, returns a
            dictionary with sampled logits.
        """
        if self.training:
            return self.model(input)  # [B, C]

        sampled_logits = []
        for model_index in range(self.num_models):
            self._set_model(model_index=model_index)

            if amp_autocast is not None:
                with amp_autocast():
                    logits = self.model(input)  # [B, C]
            else:
                logits = self.model(input)  # [B, C]

            sampled_logits.append(logits)

        sampled_logits = torch.stack(sampled_logits, dim=1)  # [B, S, C]

        return {"logit": sampled_logits}

    def forward_features(self, inputs: Tensor) -> None:
        """Not implemented for fast deep ensembles.

        Args:
            inputs: Input tensor.
        """
        del inputs
        msg = f"forward_features cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> Tensor | dict[str, Tensor]:
        """Not implemented for fast deep ensembles.

        Args:
            input: Input tensor.
            pre_logits: Whether to return pre-logits instead of logits.
        """
        del input, pre_logits
        msg = f"forward_head cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def to(self, *args: Any, **kwargs: Any) -> "FastDeepEnsembleWrapper":
        """Moves and/or casts the parameters and buffers.

        Args:
            *args: Arguments to specify the destination device, dtype, and
                other options.
            **kwargs: Keyword arguments for the same purpose.

        Returns:
            Self with parameters and buffers moved and/or cast.
        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )

        if device is not None:
            for name, buf in self.named_buffers():
                if "static" not in name:
                    buf.data = buf.data.to(device, non_blocking=non_blocking)

            for name, param in self.named_parameters():
                if "static" not in name:
                    param.data = param.data.to(device, non_blocking=non_blocking)

        return super().to(
            device=None,
            dtype=dtype,
            non_blocking=non_blocking,
            memory_format=convert_to_format,
        )

    def cuda(
        self, device: int | torch.device | None = None
    ) -> "FastDeepEnsembleWrapper":
        """Moves all model parameters and buffers to the GPU.

        Args:
            device: The destination GPU device. Defaults to the current CUDA device.

        Returns:
            Self with all parameters and buffers moved to GPU.
        """
        if device is None:
            device = torch.device("cuda")
        return self.to(device)

    def cpu(self) -> "FastDeepEnsembleWrapper":
        """Moves all model parameters and buffers to the CPU.

        Returns:
            Self with all parameters and buffers moved to CPU.
        """
        return self.to("cpu")

    def xpu(
        self, device: int | torch.device | None = None
    ) -> "FastDeepEnsembleWrapper":
        """Moves all model parameters and buffers to the XPU.

        Args:
            device: The destination XPU device. Defaults to the current XPU device.

        Returns:
            Self with all parameters and buffers moved to XPU.
        """
        if device is None:
            device = torch.device("xpu")

        return self.to(device)

    def ipu(
        self, device: int | torch.device | None = None
    ) -> "FastDeepEnsembleWrapper":
        """Moves all model parameters and buffers to the IPU.

        Args:
            device: The destination IPU device. Defaults to the current IPU device.

        Returns:
            Self with all parameters and buffers moved to IPU.
        """
        if device is None:
            device = torch.device("ipu")

        return self.to(device)

    def _set_model(self, model_index: int) -> None:
        """Sets the model parameters to a specific sampled model.

        Args:
            model_index: Index of the sampled model to use.

        Raises:
            ValueError: If an invalid model index is provided.
        """
        if model_index >= self.num_models or model_index < 0:
            msg = "Invalid model index provided"
            raise ValueError(msg)

        params = self._model_params_static[model_index]
        self._unflatten_and_set_params(params)
        self.model.apply(
            lambda module: self._load_bn_stats(module=module, model_index=model_index)
        )

    def _add_ensemble_params(self, module: nn.Module) -> None:
        """Adds ensemble-specific parameters to a module.

        Args:
            module: The module to add ensemble parameters to.
        """
        for name, _ in module.named_parameters(recurse=False):
            self._modules_and_names.append((module, name))

        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ):
            module.register_buffer(
                "running_means_static",
                torch.zeros(
                    0, *module.running_mean.shape, device=self._ensemble_params_device
                ),
            )
            module.register_buffer(
                "running_vars_static",
                torch.zeros(
                    0, *module.running_var.shape, device=self._ensemble_params_device
                ),
            )
            module.register_buffer(
                "num_batches_tracked_static",
                torch.zeros(
                    (0,), dtype=torch.long, device=self._ensemble_params_device
                ),
            )

    def _populate_ensemble_params(self) -> None:
        """Populates ensemble parameters from `self._weight_paths`."""
        for weight_path in self._weight_paths:
            self._load_model(weight_path, strict=False)

            param_list = []
            for module, name in self._modules_and_names:
                param_list.append(getattr(module, name).detach())

            params = self._flatten_params(param_list)
            self._model_params_static = torch.cat(
                [self._model_params_static, params.unsqueeze(dim=0)], dim=0
            )
            self.model.apply(FastDeepEnsembleWrapper._store_bn_stats)

    @torch.no_grad()
    def _unflatten_and_set_params(self, flat_params: Tensor) -> None:
        """Unflattens and sets model parameters from a flat tensor.

        Args:
            flat_params: Flattened tensor of model parameters.
        """
        ind = 0
        for module, name in self._modules_and_names:
            param = getattr(module, name)
            numel = param.numel()
            new_param = (
                flat_params[ind : ind + numel].reshape_as(param).to(param.device)
            )
            param.copy_(new_param)
            ind += numel

    @staticmethod
    def _flatten_params(params: Iterable[Tensor]) -> Tensor:
        """Flattens a list of parameter tensors into a single tensor.

        Args:
            params: Iterable of parameter tensors.

        Returns:
            Flattened tensor of parameters.
        """
        return torch.cat([param.flatten() for param in params])

    @staticmethod
    def _load_bn_stats(module: nn.Module, model_index: int) -> None:
        """Loads BatchNorm statistics for a specific model.

        Args:
            module: The module containing BatchNorm layers.
            model_index: Index of the model to load statistics for.
        """
        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ):
            module.running_mean.copy_(module.running_means_static[model_index])
            module.running_var.copy_(module.running_vars_static[model_index])
            module.num_batches_tracked.copy_(
                module.num_batches_tracked_static[model_index]
            )

    @staticmethod
    def _store_bn_stats(module: nn.Module) -> None:
        """Stores BatchNorm statistics for the ensemble.

        Args:
            module: The module containing BatchNorm layers.
        """
        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ):
            device = module.running_means_static.device
            module.running_means_static = torch.cat(
                [
                    module.running_means_static,
                    module.running_mean.to(device).unsqueeze(0),
                ],
                dim=0,
            )
            module.running_vars_static = torch.cat(
                [
                    module.running_vars_static,
                    module.running_var.to(device).unsqueeze(0),
                ],
                dim=0,
            )
            module.num_batches_tracked_static = torch.cat([
                module.num_batches_tracked_static,
                module.num_batches_tracked.to(device).unsqueeze(0),
            ])
