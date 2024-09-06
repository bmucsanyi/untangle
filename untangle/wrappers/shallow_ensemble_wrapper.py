"""Shallow ensemble implementation as a wrapper class."""

import torch
from torch import Tensor, nn

from untangle.wrappers.model_wrapper import DistributionalWrapper


class ShallowEnsembleClassifier(nn.Module):
    """Simple shallow ensemble classifier."""

    def __init__(self, num_heads, num_features, num_classes) -> None:
        super().__init__()
        self._shallow_classifiers = nn.Linear(num_features, num_classes * num_heads)
        self._num_heads = num_heads
        self._num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        logits = self._shallow_classifiers(x).reshape(
            -1, self._num_heads, self._num_classes
        )  # [B, S, C]

        return logits


class ShallowEnsembleWrapper(DistributionalWrapper):
    """This module takes a model as input and creates a shallow ensemble from it."""

    def __init__(
        self,
        model: nn.Module,
        num_heads: int,
    ):
        super().__init__(model)

        self._num_heads = num_heads
        self._classifier = ShallowEnsembleClassifier(
            num_heads=self._num_heads,
            num_features=self.num_features,
            num_classes=self.num_classes,
        )

    @torch.jit.ignore
    def get_classifier(self):
        return self._classifier

    def reset_classifier(self, num_heads: int | None = None, *args, **kwargs):
        if num_heads is not None:
            self._num_heads = num_heads

        # Resets global pooling in `self.classifier`
        self.model.reset_classifier(*args, **kwargs)
        self._classifier = ShallowEnsembleClassifier(
            num_heads=self._num_heads,
            num_features=self.num_features,
            num_classes=self.num_classes,
        )
