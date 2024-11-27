"""Shallow ensemble implementation as a wrapper class."""

from typing import Any

from torch import Tensor, nn

from untangle.wrappers.model_wrapper import DistributionalWrapper


class ShallowEnsembleClassifier(nn.Module):
    """Simple shallow ensemble classifier.

    This class implements a shallow ensemble classifier with multiple heads.

    Args:
        num_heads: Number of ensemble heads.
        num_features: Number of input features.
        num_classes: Number of output classes.
    """

    def __init__(self, num_heads: int, num_features: int, num_classes: int) -> None:
        super().__init__()
        self._shallow_classifiers = nn.Linear(num_features, num_classes * num_heads)
        self._num_heads = num_heads
        self._num_classes = num_classes

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass of the shallow ensemble classifier.

        Args:
            x: Input tensor.

        Returns:
            Logits tensor.
        """
        logits = self._shallow_classifiers(x).reshape(
            -1, self._num_heads, self._num_classes
        )  # [B, S, C]

        return logits


class ShallowEnsembleWrapper(DistributionalWrapper):
    """Wrapper that creates a shallow ensemble from an input model.

    Args:
        model: Base model to wrap.
        num_heads: Number of ensemble heads.
    """

    def __init__(
        self,
        model: nn.Module,
        num_heads: int,
    ) -> None:
        super().__init__(model)

        self._num_heads = num_heads
        self._classifier = ShallowEnsembleClassifier(
            num_heads=self._num_heads,
            num_features=self.num_features,
            num_classes=self.num_classes,
        )

    def get_classifier(self) -> ShallowEnsembleClassifier:
        """Returns the shallow ensemble classifier.

        Returns:
            The shallow ensemble classifier.
        """
        return self._classifier

    def reset_classifier(
        self, num_heads: int | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """Resets the classifier with a new number of heads.

        Args:
            num_heads: New number of ensemble heads. If None, keeps the current number.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        if num_heads is not None:
            self._num_heads = num_heads

        # Resets global pooling in `self.classifier`
        self.model.reset_classifier(*args, **kwargs)
        self._classifier = ShallowEnsembleClassifier(
            num_heads=self._num_heads,
            num_features=self.num_features,
            num_classes=self.num_classes,
        )
