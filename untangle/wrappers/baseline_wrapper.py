"""Deterministic model wrapper class."""

from torch import nn

from untangle.wrappers.model_wrapper import DistributionalWrapper


class BaselineWrapper(DistributionalWrapper):
    """This module takes a model as input and keeps it as is.

    It only serves as connective tissue to the rest of the framework.
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        super().__init__(model)

    def forward_head(self, x, *, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        out = self.model.get_classifier()(features)

        return out if self.training else (out.unsqueeze(dim=1),)
