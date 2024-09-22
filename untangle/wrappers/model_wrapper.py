"""Contains base wrapper classes."""

import torch
from torch import nn


class ModelWrapper(nn.Module):
    """General model wrapper base class."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def __getattr__(self, name: str):
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

    def forward_head(self, x, *, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        out = self.get_classifier()(features)

        return out

    @staticmethod
    def _convert_state_dict(state_dict):
        """Converts state_dict by removing 'model.' prefix from keys."""
        converted_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                converted_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                converted_state_dict[k] = v
        return converted_state_dict

    def _load_model(self):
        """Loads the model."""
        weight_path = self._weight_path
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        state_dict = self._convert_state_dict(state_dict)

        self.model.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x


class DistributionalWrapper(ModelWrapper):
    """Meta-class of distributional methods."""


class SpecialWrapper(ModelWrapper):
    """Meta-class of deterministic methods."""


class DirichletWrapper(DistributionalWrapper):
    """Meta-class of Dirichlet-based methods."""
