"""Direct correctness prediction implementation as a wrapper class."""

import re

import torch
import torch.nn.functional as F
from torch import nn

from untangle.models.utils import BinaryClassifier
from untangle.wrappers.loss_prediction_wrapper import AveragePool, EmbeddingNetwork
from untangle.wrappers.model_wrapper import SpecialWrapper


class BaseCorrectnessPredictionWrapper(SpecialWrapper):
    """Base class for correctness prediction methods."""


class DeepCorrectnessPredictionWrapper(BaseCorrectnessPredictionWrapper):
    """This module takes a model and creates a correctness prediction module."""

    def __init__(
        self,
        model: nn.Module,
        num_hidden_features: int,
        mlp_depth: int,
        stopgrad: bool,
        num_hooks=None,
        module_type=None,
        module_name_regex=None,
    ):
        super().__init__(model)

        self._num_hidden_features = num_hidden_features
        self._mlp_depth = mlp_depth
        self._stopgrad = stopgrad
        self._num_hooks = num_hooks
        self._module_type = module_type
        self._module_name_regex = module_name_regex

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
            in_channels=num_hidden_features * num_hooks,
            width=num_hidden_features,
            depth=mlp_depth,
        )

    @staticmethod
    def _get_layer_candidates(model, module_type, module_name_regex):
        layer_candidates = {}

        if module_name_regex is not None:
            module_name_regex = re.compile(module_name_regex)

        layer_candidates = {
            name: module
            for name, module in model.named_modules()
            if (module_name_regex is not None and module_name_regex.match(name))
            or (module_type is not None and isinstance(module, module_type))
        }

        return layer_candidates

    @staticmethod
    def _filter_layer_candidates(layer_candidates, num_hooks):
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

    def _attach_hooks(self, chosen_layers):
        def get_features(name):
            def hook(model, input, output):
                del model, input
                self._feature_buffer[name] = output.detach() if self._stopgrad else output

            return hook

        # Attach hooks to all children layers
        for name, layer in chosen_layers.items():
            self._hook_layer_names.append(name)
            handle = layer.register_forward_hook(get_features(name))
            self._hook_handles.append(handle)

    def _remove_hooks(self):
        # Remove all hooks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _add_embedding_modules(self):
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
    def _get_feature_dim(shape):
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
    def _get_pooling_layer(shape):
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

    def forward_features(self, x):
        self._feature_buffer.clear()

        return self.model.forward_features(x)

    def forward_head(self, x, *, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

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
            "feature": features,
            "error_probability": 1 - F.sigmoid(binary_logits),  # prob. of error
        }


class CorrectnessPredictionWrapper(BaseCorrectnessPredictionWrapper):
    """This module takes a model as input and creates a risk prediction module."""

    def __init__(
        self,
        model: nn.Module,
        num_hidden_features: int,  # 256
        mlp_depth: int,
        stopgrad: bool,
    ):
        super().__init__(model)

        self._num_hidden_features = num_hidden_features
        self._mlp_depth = mlp_depth
        self._stopgrad = stopgrad

        self._binary_classifier = BinaryClassifier(
            in_channels=model.num_features, width=num_hidden_features, depth=mlp_depth
        )

    def forward_head(self, x, *, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

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
