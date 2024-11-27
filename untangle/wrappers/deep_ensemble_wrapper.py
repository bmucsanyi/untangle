"""Deep ensemble wrapper class."""

from pathlib import Path

from torch import nn

from untangle.wrappers.model_wrapper import DistributionalWrapper


class DeepEnsembleWrapper(DistributionalWrapper):
    """Wrapper to manage an ensemble of independently trained models.

    Args:
        model: The model to be wrapped.
        weight_paths: List of paths to model weights.
    """

    def __init__(
        self,
        model: nn.Module,
        weight_paths: list[Path],
    ) -> None:
        super().__init__(model=model)
        self._weight_paths = weight_paths
        self.num_models = len(weight_paths)

        self.load_model_with_index(model_index=0)

    def load_model_with_index(self, model_index: int) -> None:
        """Loads a model based on the model_index.

        Args:
            model_index: Index of the model to load.

        Raises:
            ValueError: If the model_index is out of bounds.
        """
        if model_index < 0 or model_index >= self.num_models:
            msg = "Index out of bounds"
            raise ValueError(msg)

        device = next(self.model.parameters()).device

        weight_path = self._weight_paths[model_index]
        self._load_model(weight_path)
        self.model.to(device)
