"""Direct correctness prediction implementation as a wrapper class."""

import re
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from untangle.models.utils import BinaryClassifier
from untangle.wrappers.loss_prediction_wrapper import AveragePool, EmbeddingNetwork
from untangle.wrappers.model_wrapper import SpecialWrapper


class BaseCorrectnessPredictionWrapper(SpecialWrapper):
    """Base class for correctness prediction methods."""


class DeepCorrectnessPredictionWrapper(BaseCorrectnessPredictionWrapper):
    """Wrapper that creates a deep correctness prediction model from an input model.

    Args:
        model: The neural network model to be wrapped.
        num_hidden_features: Number of hidden features in the embedding network.
        mlp_depth: Depth of the MLP in the binary classifier.
        stopgrad: Whether to stop gradient propagation in feature extraction.
        num_hooks: Number of hooks to attach for feature extraction.
        module_type: Type of modules to attach hooks to.
        module_name_regex: Regex pattern for module names to attach hooks to.
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
        self._binary_classifier = BinaryClassifier(
            in_channels=num_hidden_features * len(chosen_layers),
            width=num_hidden_features,
            depth=mlp_depth,
        )

    @staticmethod
    def _get_layer_candidates(
        model: nn.Module, module_type: type | None, module_name_regex: str | None
    ) -> dict[str, nn.Module]:
        """Gets candidate layers for hook attachment.

        Args:
            model: The model to search for candidate layers.
            module_type: Type of modules to consider as candidates.
            module_name_regex: Regex pattern for module names to consider as candidates.

        Returns:
            A dictionary of candidate layers.
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
            num_hooks: Number of hooks to select.

        Returns:
            A dictionary of filtered candidate layers.
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
        # Remove all hooks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _add_embedding_modules(self) -> None:
        """Adds embedding modules for feature processing."""
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
            The feature dimension.

        Raises:
            ValueError: If the shape is invalid.
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
            An instance of the appropriate pooling layer.

        Raises:
            ValueError: If the shape is invalid.
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

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward pass through the feature extraction part of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor from the feature extraction part.
        """
        self._feature_buffer.clear()

        return self.model.forward_features(x)

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> dict[str, Tensor] | tuple[Tensor, Tensor] | Tensor:
        """Forward pass through the head of the model.

        Args:
            input: Input tensor.
            pre_logits: Whether to return pre-logits.

        Returns:
            Either pre-logits, a tuple of logits and binary logits, or a dictionary
            containing logits and error probability.
        """
        # Always get pre_logits
        features = self.model.forward_head(input, pre_logits=True)

        if pre_logits:
            return features

        logits = self.get_classifier()(features)
        binary_classifier_features = torch.cat(
            [
                self.embedding_modules[
                    self.hook_layer_name_to_embedding_module[hook_layer_name]
                ](self._feature_buffer[hook_layer_name])
                for hook_layer_name in self._hook_layer_names
            ],
            dim=1,
        )

        binary_logits = self._binary_classifier(binary_classifier_features).squeeze()

        if self.training:
            return logits, binary_logits
        return {
            "logit": logits,
            "error_probability": 1 - F.sigmoid(binary_logits),  # prob. of error
        }


class CorrectnessPredictionWrapper(BaseCorrectnessPredictionWrapper):
    """Wrapper that creates a correctness prediction model from an input model.

    Args:
        model: The neural network model to be wrapped.
        num_hidden_features: Number of hidden features in the binary classifier.
        mlp_depth: Depth of the MLP in the binary classifier.
        stopgrad: Whether to stop gradient propagation in feature extraction.
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

        self._binary_classifier = BinaryClassifier(
            in_channels=model.num_features, width=num_hidden_features, depth=mlp_depth
        )

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> dict[str, Tensor] | tuple[Tensor, Tensor] | Tensor:
        """Forward pass through the head of the model.

        Args:
            input: Input tensor.
            pre_logits: Whether to return pre-logits.

        Returns:
            Either pre-logits, a tuple of logits and binary logits, or a dictionary
            containing logits, features, and error probability.
        """
        # Always get pre_logits
        features = self.model.forward_head(input, pre_logits=True)

        if pre_logits:
            return features

        logits = self.get_classifier()(features)

        binary_classifier_features = features.detach() if self._stopgrad else features

        binary_logits = self._binary_classifier(binary_classifier_features).squeeze()

        if self.training:
            return logits, binary_logits
        return {
            "logit": logits,
            "feature": features,
            "error_probability": 1 - F.sigmoid(binary_logits),  # prob. of error
        }
