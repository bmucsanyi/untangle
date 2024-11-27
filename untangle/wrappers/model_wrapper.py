"""Contains base wrapper classes."""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class ModelWrapper(nn.Module):
    """Serves as a general model wrapper base class.

    Args:
        model: The neural network model to be wrapped.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def __getattr__(self, name: str) -> Any:
        """Retrieves attributes from the wrapped model.

        Args:
            name: Name of the attribute to retrieve.

        Returns:
            The requested attribute from the wrapped model.
        """
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        return getattr(self.model, name)

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> Tensor | dict[str, Tensor]:
        """Performs forward pass through the model's head.

        Args:
            input: Input tensor.
            pre_logits: Flag to return pre-logits features.

        Returns:
            Output tensor during training or dictionary containing logits during
            inference.
        """
        # Always get pre_logits
        features = self.model.forward_head(input, pre_logits=True)

        if pre_logits:
            return features

        out = self.get_classifier()(features)

        if self.training:
            return out
        return {"logit": out}

    @staticmethod
    def _convert_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Converts state_dict by removing 'model.' prefix from keys.

        Args:
            state_dict: Original state dictionary.

        Returns:
            Converted state dictionary.
        """
        converted_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                converted_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                converted_state_dict[k] = v
        return converted_state_dict

    def _load_model(self, weight_path: Path, *, strict: bool = True) -> None:
        """Loads the model.

        Args:
            weight_path: Path to weights.
            strict: Whether to allow additional or missing keys.
        """
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        state_dict = self._convert_state_dict(state_dict)
        self.model.load_state_dict(state_dict, strict=strict)

    def forward(self, input: Tensor) -> Tensor | dict[str, Tensor]:
        """Performs forward pass through the entire model.

        Args:
            input: Input tensor.

        Returns:
            Model output or dictionary containing output tensors.
        """
        x = self.forward_features(input)
        x = self.forward_head(x)

        return x


class DistributionalWrapper(ModelWrapper):
    """Serves as a meta-class for distributional methods."""


class SpecialWrapper(ModelWrapper):
    """Serves as a meta-class for deterministic methods."""


class DirichletWrapper(DistributionalWrapper):
    """Serves as a meta-class for Dirichlet-based methods."""
