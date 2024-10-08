"""Deterministic model wrapper class."""

from torch import nn

from untangle.wrappers.model_wrapper import DistributionalWrapper


class CEBaselineWrapper(DistributionalWrapper):
    """A wrapper class that maintains the input model as-is.

    This module serves as a connector to the rest of the framework without modifying
    the input model's behavior.

    Args:
        model: The neural network model to be wrapped.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__(model)
