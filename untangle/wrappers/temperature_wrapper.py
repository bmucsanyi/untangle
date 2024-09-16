"""Temperature scaling wrapper class."""

import torch
import torch.nn.functional as F
from torch import nn

from untangle.wrappers.model_wrapper import SpecialWrapper


class TemperatureWrapper(SpecialWrapper):
    """This module takes a model as input and temperature scales it."""

    def __init__(
        self,
        model: nn.Module,
        weight_path: str | None = None,
    ):
        super().__init__(model)
        self._weight_path = weight_path
        self.register_buffer("_temperature", torch.tensor(1.0))

        if self._weight_path:
            self._load_model()

    def forward_head(self, x, *, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        out = self.get_classifier()(features)
        out /= self._temperature

        if self.training:
            return out
        return {"logit": out}

    def set_temperature_loader(self, val_loader):
        device = next(self.model.parameters()).device

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                input = input.to(device)
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cpu()
            labels = torch.cat(labels_list).cpu()

        return self._set_temperature_logits(logits, labels)

    def _set_temperature_logits(self, logits, labels):
        nll_val = float("inf")
        T_opt_nll = 1.0
        T = 0.1
        for _ in range(100):
            after_temperature_nll = F.cross_entropy(logits / T, labels).item()

            if nll_val > after_temperature_nll:
                T_opt_nll = T
                nll_val = after_temperature_nll

            T += 0.1

        self._temperature.copy_(T_opt_nll)
