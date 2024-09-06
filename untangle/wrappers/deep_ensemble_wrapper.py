"""Deep ensemble wrapepr class."""

from untangle.wrappers.model_wrapper import DistributionalWrapper


class DeepEnsembleWrapper(DistributionalWrapper):
    """Wrapper to manage an ensemble of independently trained models."""

    def __init__(
        self,
        model,
        weight_paths: list,
        kwargs: dict,
    ):
        super().__init__(model=model)
        self._weight_paths = weight_paths
        self._weight_path = self._weight_paths[0]
        self.num_models = len(weight_paths)
        self.kwargs = kwargs

        self._load_model(0)

    def _load_model(self, model_index: int):
        """Loads a model based on the model_index."""
        if model_index < 0 or model_index >= self.num_models:
            msg = "Index out of bounds"
            raise ValueError(msg)

        device = next(self.model.parameters()).device

        self._weight_path = self._weight_paths[model_index]
        super()._load_model()
        self.model.to(device)
