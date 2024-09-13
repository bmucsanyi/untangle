"""Linearized SWAG wrapper class based on https://github.com/wjmaddox/swa_gaussian."""

import logging

import torch

from untangle.utils.derivative import jvp, vjp
from untangle.wrappers.model_wrapper import DistributionalWrapper

logger = logging.getLogger(__name__)


class LinearizedSWAGWrapper(DistributionalWrapper):
    """This module takes a model and creates a SWAG model posterior."""

    def __init__(self, model, weight_path, use_low_rank_cov, max_rank):
        super().__init__(model)

        self._use_low_rank_cov = use_low_rank_cov
        self._max_rank = max_rank
        self._min_var = 1e-30
        self._swag_params_device = torch.device("cpu")

        self._weight_path = weight_path
        self._load_model()

        self.register_buffer(
            "_num_checkpoints_swag",
            torch.zeros((), dtype=torch.long, device=self._swag_params_device),
        )

        self._modules_and_names = []
        self.model.apply(self._add_swag_params)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(  # noqa: SLF001
            *args, **kwargs
        )

        if device is not None:
            for name, buf in self.named_buffers():
                if "swag" not in name:
                    buf.data = buf.data.to(device, non_blocking=non_blocking)

            for name, param in self.named_parameters():
                if "swag" not in name:
                    param.data = param.data.to(device, non_blocking=non_blocking)

        return super().to(
            device=None,
            dtype=dtype,
            non_blocking=non_blocking,
            memory_format=convert_to_format,
        )

    def cuda(self, device=None):
        device = device or "cuda"
        self.to(device)

    def cpu(self):
        self.to("cpu")

    def xpu(self, device=None):
        device = device or "xpu"
        self.to(device)

    def ipu(self, device=None):
        device = device or "ipu"
        self.to(device)

    @torch.no_grad()
    def update_stats(self):
        for module, name in self._modules_and_names:
            param = getattr(module, name)

            prev_num_checkpoints = self._num_checkpoints_swag
            curr_num_checkpoints = prev_num_checkpoints + 1

            # Update mean
            mean = getattr(module, f"{name}_mean_swag")
            param = param.to(mean.device)
            new_mean = (
                prev_num_checkpoints * mean / curr_num_checkpoints
                + param / curr_num_checkpoints
            )
            setattr(module, f"{name}_mean_swag", new_mean)

            # Update second non-centered moment
            sq_mean = getattr(module, f"{name}_sq_mean_swag")
            new_sq_mean = (
                prev_num_checkpoints * sq_mean / curr_num_checkpoints
                + param.square() / curr_num_checkpoints
            )
            setattr(module, f"{name}_sq_mean_swag", new_sq_mean)

            if self._use_low_rank_cov:
                # Update square root of covariance matrix
                sqrt_cov = getattr(module, f"{name}_sqrt_cov_swag")
                new_row = (param - new_mean).reshape(1, -1)
                new_sqrt_cov = torch.cat([sqrt_cov, new_row], dim=0)

                # Adhere to max_rank
                if new_sqrt_cov.shape[0] > self._max_rank:
                    new_sqrt_cov = new_sqrt_cov[-self._max_rank :]

                setattr(module, f"{name}_sqrt_cov_swag", new_sqrt_cov)

        self._num_checkpoints_swag.add_(1)

    @staticmethod
    def calculate_checkpoint_batches(
        num_batches, num_checkpoints_per_epoch, accumulation_steps
    ):
        if num_checkpoints_per_epoch <= 0:
            msg = "num_checkpoints_per_epoch must be positive"
            raise ValueError(msg)

        if num_checkpoints_per_epoch > num_batches:
            msg = (
                "num_checkpoints_per_epoch cannot be greater than the number of batches"
            )
            raise ValueError(msg)

        checkpoint_batches = set()

        # Always include the last batch
        checkpoint_batches.add(num_batches - 1)

        if num_checkpoints_per_epoch > 1:
            # Calculate the step size between checkpoints
            step = num_batches // num_checkpoints_per_epoch

            for i in range(1, num_checkpoints_per_epoch):
                # Calculate the ideal batch index
                ideal_batch = i * step

                # Find the nearest batch index that satisfies the accumulation condition
                batch = (
                    ideal_batch
                    - (ideal_batch % accumulation_steps)
                    + accumulation_steps
                    - 1
                )

                # Ensure we don't go past the last batch
                batch = min(batch, num_batches - 2)

                if batch not in checkpoint_batches:
                    checkpoint_batches.add(batch)

        # Check if we have the correct number of checkpoints
        if len(checkpoint_batches) != num_checkpoints_per_epoch:
            msg = f"Could not generate {num_checkpoints_per_epoch} unique checkpoints"
            raise ValueError(msg)

        return checkpoint_batches

    @torch.no_grad()
    def set_map_weights(self):
        for module, name in self._modules_and_names:
            param = getattr(module, name)
            mean_param = getattr(module, f"{name}_mean_swag")
            param.copy_(mean_param)

    def forward_features(self, inputs):
        del inputs
        msg = f"forward_features cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward_head(self, features):
        del features
        msg = f"forward_head cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward(self, x):
        param_list = []
        std_list = []

        if self._use_low_rank_cov:
            sqrt_cov_list = []

        for module, name in self._modules_and_names:
            param_list.append(getattr(module, name))
            std_list.append(
                torch.clamp(
                    getattr(module, f"{name}_sq_mean_swag")
                    - getattr(module, f"{name}_mean_swag").square(),
                    min=self._min_var,
                ).sqrt()
            )

            if self._use_low_rank_cov:
                sqrt_cov_list.append(getattr(module, f"{name}_sqrt_cov_swag"))

        with torch.enable_grad():
            y = self.model(x)  # [B, C]
            a = torch.zeros((self.model.num_classes,), dtype=y.dtype, device=y.device)
            for i in range(self.model.num_classes):
                e_i = torch.zeros_like(y)
                e_i[i] = 1.0
                J_i = vjp(y=y, x=param_list, v=e_i, retain_graph=True)
                a[i] = self.mul_square_sum(J_i, std_list)

            if not self._use_low_rank_cov:
                return y, a

            B = jvp(y=y, x=param_list, v=sqrt_cov_list, is_grads_batched=True)[0]
            b = B.square().sum(dim=0)

        return y, a / 2 + b / (2 * (sqrt_cov_list[0].shape[0] - 1))

    @staticmethod
    def mul_square_sum(a_list, b_list):
        res = 0.0
        for a_param, b_param in zip(a_list, b_list, strict=True):
            res += a_param.mul(b_param).square().sum()

    def _add_swag_params(self, module):
        for name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue

            module.register_buffer(
                f"{name}_mean_swag",
                torch.zeros_like(param, device=self._swag_params_device),
            )
            module.register_buffer(
                f"{name}_sq_mean_swag",
                torch.zeros_like(param, device=self._swag_params_device),
            )

            if self._use_low_rank_cov:
                module.register_buffer(
                    f"{name}_sqrt_cov_swag",
                    torch.zeros((0, param.numel()), device=self._swag_params_device),
                )

            self._modules_and_names.append((module, name))
