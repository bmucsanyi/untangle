"""Deterministic model wrapper class."""

from torch import nn

from untangle.wrappers.model_wrapper import DistributionalWrapper


class CEBaselineWrapper(DistributionalWrapper):
    """This module takes a model as input and keeps it as is.

    It only serves as connective tissue to the rest of the framework.
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__(model)
