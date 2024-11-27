"""Direct loss prediction implementation as a wrapper class."""

import re
from collections.abc import Callable

import torch
from torch import Tensor, nn

from untangle.models import NonNegativeRegressor
from untangle.wrappers.model_wrapper import SpecialWrapper


class BaseLossPredictionWrapper(SpecialWrapper):
    """Base class for loss prediction methods."""


class EmbeddingNetwork(nn.Module):
    """Embedding network used by deep loss prediction.

    Args:
        in_channels: Number of input channels.
        width: Width of the linear layer.
        pool: Pooling layer to be used.
    """

    def __init__(self, in_channels: int, width: int, pool: nn.Module) -> None:
        super().__init__()

        # Embedding layers
        self._norm = nn.LayerNorm(in_channels)
        self._pool = pool
        self._linear = nn.Linear(in_channels, width)
        self._leaky_relu = nn.LeakyReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the embedding network.

        Args:
            x: Input tensor.

        Returns:
            Processed tensor.
        """
        use_norm = x.dim() < 4

        if use_norm:
            x = self._norm(x)

        x = self._pool(x)

        if use_norm:
            x = self._norm(x)

        x = self._linear(x)
        x = self._leaky_relu(x)

        return x


class DeepLossPredictionWrapper(BaseLossPredictionWrapper):
    """Wrapper that creates a deep loss prediction model from an input model.

    Args:
        model: The base model to wrap.
        num_hidden_features: Number of hidden features.
        mlp_depth: Depth of the MLP.
        stopgrad: Whether to stop gradient computation.
        num_hooks: Number of hooks to use.
        module_type: Type of module to hook.
        module_name_regex: Regex pattern for module names to hook.
    """

    def __init__(
        self,
        model: nn.Module,
        num_hidden_features: int,
        mlp_depth: int,
        stopgrad: bool,
        num_hooks: int | None = None,
        module_type: type | None = None,
        module_name_regex: str | None = None,
    ) -> None:
        super().__init__(model)

        self._num_hidden_features = num_hidden_features
        self._mlp_depth = mlp_depth
        self._stopgrad = stopgrad

        # Register hooks to extract intermediate features
        self._feature_buffer = {}
        self._hook_handles = []
        self._hook_layer_names = []

        layer_candidates = self._get_layer_candidates(
            model, module_type, module_name_regex
        )
        chosen_layers = self._filter_layer_candidates(layer_candidates, num_hooks)
        self._attach_hooks(chosen_layers)

        # Initialize uncertainty network(s)
        self._add_embedding_modules()
        self._regressor = NonNegativeRegressor(
            in_channels=num_hidden_features * len(chosen_layers),
            width=num_hidden_features,
            depth=mlp_depth,
        )

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward pass through the feature extractor.

        Args:
            x: Input tensor.

        Returns:
            Feature tensor.
        """
        self._feature_buffer.clear()

        return self.model.forward_features(x)

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> tuple[Tensor, Tensor] | dict[str, Tensor]:
        """Forward pass through the head of the model.

        Args:
            input: Input tensor.
            pre_logits: Whether to return pre-logits.

        Returns:
            Output tensor or dictionary containing logits and loss values.
        """
        # Always get pre_logits
        feature = self.model.forward_head(input, pre_logits=True)

        if pre_logits:
            return feature

        logit = self.get_classifier()(feature)
        regressor_feature = torch.cat(
            [
                self.embedding_modules[
                    self.hook_layer_name_to_embedding_module[hook_layer_name]
                ](self._feature_buffer[hook_layer_name])
                for hook_layer_name in self._hook_layer_names
            ],
            dim=1,
        )

        loss_value = self._regressor(regressor_feature).squeeze()

        if self.training:
            return logit, loss_value
        return {
            "logit": logit,
            "loss_value": loss_value,
        }

    @staticmethod
    def _get_layer_candidates(
        model: nn.Module, module_type: type | None, module_name_regex: str | None
    ) -> dict[str, nn.Module]:
        """Gets candidate layers for hooking.

        Args:
            model: The model to extract layers from.
            module_type: Type of modules to consider.
            module_name_regex: Regex pattern for module names.

        Returns:
            Dictionary of candidate layers.
        """
        layer_candidates = {}

        compiled_module_name_regex = (
            re.compile(module_name_regex) if module_name_regex is not None else None
        )

        layer_candidates = {
            name: module
            for name, module in model.named_modules()
            if (
                compiled_module_name_regex is not None
                and compiled_module_name_regex.match(name)
            )
            or (module_type is not None and isinstance(module, module_type))
        }

        return layer_candidates

    @staticmethod
    def _filter_layer_candidates(
        layer_candidates: dict[str, nn.Module], num_hooks: int | None
    ) -> dict[str, nn.Module]:
        """Filters layer candidates based on the number of hooks.

        Args:
            layer_candidates: Dictionary of candidate layers.
            num_hooks: Number of hooks to use.

        Returns:
            Filtered dictionary of layers.
        """
        if num_hooks is None:
            return layer_candidates

        num_hooks = min(num_hooks, len(layer_candidates))

        chosen_layers = []
        chosen_indices = torch.linspace(
            start=0, end=len(layer_candidates) - 1, steps=num_hooks
        )
        chosen_layers = {
            name: module
            for i, (name, module) in enumerate(layer_candidates.items())
            if i in chosen_indices
        }

        return chosen_layers

    def _attach_hooks(self, chosen_layers: dict[str, nn.Module]) -> None:
        """Attaches hooks to the chosen layers.

        Args:
            chosen_layers: Dictionary of layers to attach hooks to.
        """

        def get_features(name: str) -> Callable:
            def hook(model: nn.Module, input: Tensor, output: Tensor) -> None:
                del model, input
                self._feature_buffer[name] = (
                    output.detach() if self._stopgrad else output
                )

            return hook

        # Attach hooks to all children layers
        for name, layer in chosen_layers.items():
            self._hook_layer_names.append(name)
            handle = layer.register_forward_hook(get_features(name))
            self._hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        """Removes all attached hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _add_embedding_modules(self) -> None:
        """Adds embedding modules to the wrapper."""
        # Get the feature map sizes
        empty_image = torch.zeros(
            [1, *self.model.default_cfg["input_size"]],
            device=next(self.model.parameters()).device,
        )
        with torch.no_grad():
            self._feature_buffer.clear()
            self.model(empty_image)
            feature_sizes = {
                key: self._get_feature_dim(feature.shape)
                for key, feature in self._feature_buffer.items()
            }

        modules = {}
        self.hook_layer_name_to_embedding_module = {}
        for i, (key, size) in enumerate(feature_sizes.items()):
            module_name = f"unc_{i}"
            modules[module_name] = EmbeddingNetwork(
                size,
                self._num_hidden_features,
                pool=self._get_pooling_layer(self._feature_buffer[key].shape),
            )
            self.hook_layer_name_to_embedding_module[key] = module_name
        self.embedding_modules = nn.ModuleDict(modules)

    @staticmethod
    def _get_feature_dim(shape: tuple[int, ...]) -> int:
        """Gets the feature dimension from the shape.

        Args:
            shape: Shape of the feature tensor.

        Returns:
            Feature dimension.

        Raises:
            ValueError: If the network structure is invalid.
        """
        # Exclude the batch dimension
        dims = shape[1:]

        # If there's only one dimension, return nn.Identity
        if len(dims) == 1:
            return dims[0]
        if len(dims) == 2:
            return dims[1]
        if len(dims) == 3:
            return dims[0]
        msg = "Invalid network structure"
        raise ValueError(msg)

    @staticmethod
    def _get_pooling_layer(shape: tuple[int, ...]) -> nn.Module:
        """Gets the appropriate pooling layer based on the shape.

        Args:
            shape: Shape of the feature tensor.

        Returns:
            Pooling layer.

        Raises:
            ValueError: If the network structure is invalid.
        """
        # Exclude the batch dimension
        dims = shape[1:]

        # If there's only one dimension, return nn.Identity
        if len(dims) == 1:
            return nn.Identity()
        if len(dims) == 2:
            dim = 1
        elif len(dims) == 3:
            dim = (2, 3)
        else:
            msg = "Invalid network structure"
            raise ValueError(msg)

        return AveragePool(dim)


class AveragePool(nn.Module):
    """Average pooling module.

    Args:
        dim: Dimension(s) to perform average pooling over.
    """

    def __init__(self, dim: int | tuple[int, ...]) -> None:
        super().__init__()
        self._dim = dim

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass of the average pooling module.

        Args:
            inputs: Input tensor.

        Returns:
            Pooled tensor.
        """
        return inputs.mean(self._dim)


class LossPredictionWrapper(BaseLossPredictionWrapper):
    """Wrapper that creates a loss prediction model from an input model.

    Args:
        model: The base model to wrap.
        num_hidden_features: Number of hidden features.
        mlp_depth: Depth of the MLP.
        stopgrad: Whether to stop gradient computation.
    """

    def __init__(
        self,
        model: nn.Module,
        num_hidden_features: int,
        mlp_depth: int,
        stopgrad: bool,
    ) -> None:
        super().__init__(model)

        self._num_hidden_features = num_hidden_features
        self._mlp_depth = mlp_depth
        self._stopgrad = stopgrad

        self._regressor = NonNegativeRegressor(
            in_channels=model.num_features,
            width=num_hidden_features,
            depth=mlp_depth,
        )

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> tuple[Tensor, Tensor] | dict[str, Tensor]:
        """Forward pass through the head of the model.

        Args:
            input: Input tensor.
            pre_logits: Whether to return pre-logits.

        Returns:
            Output tensor or dictionary containing logits, features, and loss values.
        """
        # Always get pre_logits
        feature = self.model.forward_head(input, pre_logits=True)

        if pre_logits:
            return feature

        logit = self.get_classifier()(feature)

        regressor_feature = feature.detach() if self._stopgrad else feature

        loss_value = self._regressor(regressor_feature).squeeze()

        if self.training:
            return logit, loss_value
        return {
            "logit": logit,
            "feature": feature,
            "loss_value": loss_value,
        }
