"""Temperature scaling wrapper class."""

import logging
import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from untangle.utils.loader import PrefetchLoader
from untangle.wrappers.model_wrapper import SpecialWrapper

logger = logging.getLogger(__name__)


class TemperatureWrapper(SpecialWrapper):
    """Wrapper that temperature-scales an input model.

    Temperature scaling is a simple post-processing method for calibrating
    neural network predictions.

    Args:
        model: The base model to wrap.
        weight_path: Path to the model weights. If provided, weights will be loaded.
    """

    def __init__(
        self,
        model: nn.Module,
        weight_path: str | None = None,
    ) -> None:
        super().__init__(model)
        self.register_buffer("_temperature", torch.tensor(1.0))

        if weight_path:
            self._load_model(weight_path)

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> dict[str, Tensor] | Tensor:
        """Performs the forward pass through the model's head.

        This method applies temperature scaling to the logits.

        Args:
            input: Input tensor.
            pre_logits: If True, return features before the final classifier layer.

        Returns:
            If pre_logits is True, returns the features.
            If training, returns the temperature-scaled logits.
            If not training, returns a dictionary with temperature-scaled logits.
        """
        # Always get pre_logits
        features = self.model.forward_head(input, pre_logits=True)

        if pre_logits:
            return features

        out = self.get_classifier()(features)
        out = out.div(self._temperature)

        if self.training:
            return out

        return {"logit": out}

    def set_temperature_loader(
        self,
        val_loader: DataLoader | PrefetchLoader,
        channels_last: bool,
    ) -> None:
        """Sets the temperature using a validation loader.

        This method finds the optimal temperature by minimizing the negative
        log-likelihood on the validation set.

        Args:
            val_loader: DataLoader or PrefetchLoader for the validation set.
            channels_last: Whether a channels_last memory layout should be used.
        """
        device = next(self.model.parameters()).device

        logger.info("Starting temperature scaling.")

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for input, label in val_loader:
                if not isinstance(val_loader, PrefetchLoader):
                    input, label = input.to(device), label.to(device)

                if channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                logits = self.model(input)

                if logits.device.type == "cuda":
                    torch.cuda.synchronize()

                logits_list.append(logits)
                labels_list.append(label)

            logits = torch.cat(logits_list).cpu()
            labels = torch.cat(labels_list).cpu()

        self._set_temperature_logits(logits, labels)

        logger.info(
            "Temperature scaling done. "
            f"Optimized temperature is {self._temperature}."
        )

    def _set_temperature_logits(self, logits: Tensor, labels: Tensor) -> None:
        """Sets the temperature using pre-computed logits and labels.

        This method finds the optimal temperature by minimizing the negative
        log-likelihood on the given logits and labels.

        Args:
            logits: Tensor of model logits.
            labels: Tensor of true labels.
        """
        nll_val = float("inf")
        T_opt_nll = 1.0
        T = 0.1

        for _ in range(100):
            logger.info(f"Trying {T}...")
            start_time = time.perf_counter()
            after_temperature_nll = F.cross_entropy(logits / T, labels).item()

            logger.info(
                f"Took {time.perf_counter() - start_time} seconds, "
                f"result: {after_temperature_nll}."
            )

            if after_temperature_nll < nll_val:
                T_opt_nll = T
                nll_val = after_temperature_nll

            T += 0.1

        self._temperature.copy_(T_opt_nll)
