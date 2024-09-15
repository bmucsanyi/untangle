"""SWAG wrapper class based on https://github.com/wjmaddox/swa_gaussian."""

import itertools
import logging
import time
from math import sqrt

import torch

from untangle.wrappers.model_wrapper import DistributionalWrapper

logger = logging.getLogger(__name__)


class SWAGWrapper(DistributionalWrapper):
    """This module takes a model and creates a SWAG model posterior."""

    def __init__(self, model, weight_path, use_low_rank_cov, max_rank):
        super().__init__(model)

        self._use_low_rank_cov = use_low_rank_cov
        self._max_rank = max_rank
        self._min_var = 1e-30
        self._swag_params_device = torch.device("cpu")

        self._weight_path = weight_path
        self._load_model()

        num_params = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        self.register_buffer(
            "_sampled_params_swag",
            torch.zeros((0, num_params), device=self._swag_params_device),
        )
        self.register_buffer(
            "_num_checkpoints_swag",
            torch.zeros((), dtype=torch.long, device=self._swag_params_device),
        )

        self._modules_and_names = []
        self.model.apply(self._add_swag_params)

    def forward(self, inputs):
        if self.training:
            return self.model(inputs)  # [B, C]

        sampled_logits = []
        for model_index in range(self.num_models):
            self._set_model(model_index=model_index)
            logits = self.model(inputs)  # [B, C]

            sampled_logits.append(logits)

        sampled_logits = torch.stack(sampled_logits, dim=1)  # [B, S, C]

        return (sampled_logits,)

    def forward_features(self, inputs):
        del inputs
        msg = f"forward_features cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def forward_head(self, features):
        del features
        msg = f"forward_head cannot be called directly for {type(self)}"
        raise ValueError(msg)

    def get_mc_samples(self, train_loader, num_mc_samples):
        logger.info("Starting MC sampling the SWAG weights.")
        for i in range(num_mc_samples):
            time_start = time.perf_counter()
            self._sample_and_store_params(train_loader=train_loader, fraction=0.1)
            time_end = time.perf_counter()
            logger.info(
                f"Sample {i + 1}/{num_mc_samples} took "
                f"{time_end - time_start} seconds."
            )

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
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

    @property
    def num_models(self):
        return self._sampled_params_swag.shape[0]

    @torch.no_grad()
    def _sample_and_store_params(self, train_loader, fraction):
        mean_list = []
        sq_mean_list = []

        if self._use_low_rank_cov:
            sqrt_cov_list = []

        for module, name in self._modules_and_names:
            mean_list.append(getattr(module, f"{name}_mean_swag"))
            sq_mean_list.append(getattr(module, f"{name}_sq_mean_swag"))

            if self._use_low_rank_cov:
                sqrt_cov_list.append(getattr(module, f"{name}_sqrt_cov_swag"))

        mean = self._flatten_params(mean_list)
        sq_mean = self._flatten_params(sq_mean_list)

        std = torch.clamp(sq_mean - mean.square(), min=self._min_var).sqrt()
        diag_sample = std * torch.randn_like(std)

        if self._use_low_rank_cov:
            diag_sample /= sqrt(2)

            sqrt_cov = torch.cat(sqrt_cov_list, dim=1)
            rank_cov = sqrt_cov.shape[0]
            low_rank_sample = sqrt_cov.T @ torch.randn(rank_cov, device=sqrt_cov.device)
            low_rank_sample /= sqrt(2 * (rank_cov - 1))

            sample = mean + diag_sample + low_rank_sample
        else:
            sample = mean + diag_sample

        self._sampled_params_swag = torch.cat(
            [self._sampled_params_swag, sample.unsqueeze(dim=0)], dim=0
        )
        self._unflatten_and_set_params(sample)
        self._set_and_store_bn_stats(train_loader=train_loader, fraction=fraction)

    def _set_model(self, model_index):
        if model_index >= self.num_models or model_index < 0:
            msg = "Invalid model index provided"
            raise ValueError(msg)

        sample = self._sampled_params_swag[model_index]
        self._unflatten_and_set_params(sample)
        self.model.apply(
            lambda module: self._load_bn_stats(module=module, model_index=model_index)
        )

    def _add_swag_params(self, module):
        for name, param in module.named_parameters(recurse=False):
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

        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ):
            module.register_buffer(
                "running_means_swag",
                torch.zeros(
                    0, *module.running_mean.shape, device=self._swag_params_device
                ),
            )
            module.register_buffer(
                "running_vars_swag",
                torch.zeros(
                    0, *module.running_var.shape, device=self._swag_params_device
                ),
            )
            module.register_buffer(
                "num_batches_tracked_swag",
                torch.zeros((0,), dtype=torch.long, device=self._swag_params_device),
            )

    def _unflatten_and_set_params(self, flat_params):
        ind = 0
        for module, name in self._modules_and_names:
            param = getattr(module, name)
            numel = param.numel()
            new_param = (
                flat_params[ind : ind + numel].reshape_as(param).to(param.device)
            )
            param.copy_(new_param)
            ind += numel

    def _check_bn(self):
        def _check_bn_module(module, flag):
            if (
                isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                and module.track_running_stats
            ):
                flag[0] = True

        flag = [False]
        self.model.apply(lambda module: _check_bn_module(module, flag))
        return flag[0]

    @torch.no_grad()
    def _set_and_store_bn_stats(self, train_loader, fraction=None):
        """Updates and saves the BatchNorm buffers using a `fraction` of `loader`."""
        device = next(self.model.parameters()).device

        if not self._check_bn():
            return

        self.model.train()
        self.model.apply(SWAGWrapper._reset_bn)
        momenta = {}
        self.model.apply(lambda module: SWAGWrapper._get_momenta(module, momenta))
        num_batches = len(train_loader)

        if fraction is not None:
            num_batches = int(num_batches * fraction)
            train_loader = itertools.islice(train_loader, num_batches)

        num_inputs = 0
        for input, _ in train_loader:
            input = input.to(device)
            batch_size = input.shape[0]

            momentum = batch_size / (num_inputs + batch_size)
            for module in momenta:
                module.momentum = momentum

            self.model(input)
            num_inputs += batch_size

        self.model.apply(lambda module: SWAGWrapper._set_momenta(module, momenta))
        self.model.apply(SWAGWrapper._store_bn_stats)

    @staticmethod
    def _flatten_params(params):
        return torch.cat([param.flatten() for param in params])

    @staticmethod
    def _load_bn_stats(module, model_index):
        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ):
            module.running_mean.copy_(module.running_means_swag[model_index])
            module.running_var.copy_(module.running_vars_swag[model_index])
            module.num_batches_tracked.copy_(
                module.num_batches_tracked_swag[model_index]
            )

    @staticmethod
    def _reset_bn(module):
        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ):
            module.reset_running_stats()

    @staticmethod
    def _get_momenta(module, momenta):
        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ):
            momenta[module] = module.momentum

    @staticmethod
    def _set_momenta(module, momenta):
        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ):
            module.momentum = momenta[module]

    @staticmethod
    def _store_bn_stats(module):
        if (
            isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and module.track_running_stats
        ):
            device = module.running_means_swag.device
            module.running_means_swag = torch.cat(
                [
                    module.running_means_swag,
                    module.running_mean.to(device).unsqueeze(0),
                ],
                dim=0,
            )
            module.running_vars_swag = torch.cat(
                [
                    module.running_vars_swag,
                    module.running_var.to(device).unsqueeze(0),
                ],
                dim=0,
            )
            module.num_batches_tracked_swag = torch.cat([
                module.num_batches_tracked_swag,
                module.num_batches_tracked.to(device).unsqueeze(0),
            ])
