"""Mahalanobis implementation as a wrapper class.

Latent density estimation based on https://github.com/pokaxpoka/deep_Mahalanobis_detector.
"""

import re

import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from torch import Tensor, nn
from torch.utils.data import DataLoader

from untangle.utils.loader import PrefetchLoader
from untangle.wrappers.model_wrapper import SpecialWrapper


class MahalanobisWrapper(SpecialWrapper):
    """Wrapper that creates a Mahalanobis model from an input model.

    Args:
        model: The neural network model to be wrapped.
        magnitude: Magnitude of perturbation for noisy Mahalanobis score calculation.
        weight_path: Path to the model weights.
        num_hooks: Number of hooks to use for feature extraction.
        module_type: Type of module to hook.
        module_name_regex: Regex pattern for module names to hook.
    """

    def __init__(
        self,
        model: nn.Module,
        magnitude: float,
        weight_path: str,
        num_hooks: int | None = None,
        module_type: type | None = None,
        module_name_regex: str | None = None,
    ) -> None:
        super().__init__(model)

        self._magnitude = magnitude

        # Register hooks to extract intermediate features
        self._feature_list = []
        self._hook_handles = []
        self._hook_layer_names = []
        self._is_logistic_regressor_constructed = False
        self.register_buffer("_logistic_regressor_coef", None)
        self.register_buffer("_logistic_regressor_intercept", None)

        layer_candidates = self._get_layer_candidates(
            model=self.model,
            module_type=module_type,
            module_name_regex=module_name_regex,
        )
        chosen_layers = self._filter_layer_candidates(
            layer_candidates=layer_candidates, num_hooks=num_hooks
        )
        self._num_layers = len(chosen_layers)
        self._attach_hooks(chosen_layers=chosen_layers)

        for i in range(self._num_layers):
            self.register_buffer(f"_class_means_{i}", None)
            self.register_buffer(f"_precisions_{i}", None)

        # Store original requires_grad states
        for param in self.model.parameters():
            param.requires_grad_(False)

        self._load_model(weight_path)

    def forward_head(
        self, input: Tensor, *, pre_logits: bool = False
    ) -> Tensor | dict[str, Tensor]:
        """Raises an error as the head cannot be called separately.

        Args:
            input: Input tensor.
            pre_logits: Whether to return pre-logits instead of logits.

        Raises:
            ValueError: Always raised when this method is called.
        """
        del self, input, pre_logits
        msg = "Head cannot be called separately"
        raise ValueError(msg)

    def forward(self, input: Tensor) -> dict[str, Tensor]:
        """Performs forward pass and calculates Mahalanobis scores.

        Args:
            input: Input tensor.

        Returns:
            Dictionary containing logits and Mahalanobis values.
        """
        # This is the only way to interact with the model.
        if not self._is_logistic_regressor_constructed:
            self._reconstruct_logistic_regressor()

        self._feature_list.clear()
        features = self.model.forward_head(
            self.model.forward_features(input), pre_logits=True
        )
        ret_logits = self.model.get_classifier()(features)

        noisy_mahalanobis_scores = self._calculate_noisy_mahalanobis_scores(
            input
        )  # [B, L]

        # Clamp the values because they can sometimes become inf
        noisy_mahalanobis_scores = torch.clamp(
            noisy_mahalanobis_scores,
            min=torch.finfo(torch.float32).min,
            max=torch.finfo(torch.float32).max,
        )

        ret_uncertainties = torch.from_numpy(
            self._logistic_regressor.predict_proba(noisy_mahalanobis_scores.numpy())[
                :, 1
            ]
        )  # [B]

        return {
            "logit": ret_logits,
            "mahalanobis_value": ret_uncertainties,
        }

    def train_logistic_regressor(
        self,
        train_loader: DataLoader | PrefetchLoader,
        id_loader: DataLoader | PrefetchLoader,
        ood_loader: DataLoader | PrefetchLoader,
        max_num_training_samples: int,
        max_num_id_ood_samples: int | None,
        channels_last: bool,
    ) -> None:
        """Trains the logistic regressor for Mahalanobis score classification.

        Args:
            train_loader: DataLoader for training data.
            id_loader: DataLoader for in-distribution data.
            ood_loader: DataLoader for out-of-distribution data.
            max_num_training_samples: Maximum number of training samples.
            max_num_id_ood_samples: Maximum number of ID/OOD samples.
            channels_last: Whether a channels_last memory layout should be used.
        """
        self._calculate_gaussian_parameters(
            train_loader=train_loader,
            max_num_training_samples=max_num_training_samples,
            channels_last=channels_last,
        )

        num_id_samples = len(id_loader.dataset)
        num_ood_samples = len(ood_loader.dataset)

        if max_num_id_ood_samples is None:
            max_num_id_ood_samples = min(num_id_samples, num_ood_samples)
        else:
            max_num_id_ood_samples = min(
                num_id_samples, num_ood_samples, max_num_id_ood_samples
            )

        # Get Mahalanobis scores for in-distribution data
        mahalanobis_scores_id = self._calculate_noisy_mahalanobis_scores_loader(
            loader=id_loader,
            max_num_samples=max_num_id_ood_samples,
            channels_last=channels_last,
        )
        labels_id = torch.zeros(mahalanobis_scores_id.shape[0])

        # Get Mahalanobis scores for out-of-distribution data
        mahalanobis_scores_ood = self._calculate_noisy_mahalanobis_scores_loader(
            loader=ood_loader,
            max_num_samples=max_num_id_ood_samples,
            channels_last=channels_last,
        )
        labels_ood = torch.ones(mahalanobis_scores_ood.shape[0])

        # Concatenate scores and labels
        X_train = torch.cat(
            [mahalanobis_scores_id, mahalanobis_scores_ood], dim=0
        ).numpy()
        y_train = torch.cat([labels_id, labels_ood], dim=0).numpy()

        # Train logistic regression model
        logistic_regressor = LogisticRegressionCV(n_jobs=-1).fit(X_train, y_train)

        self._logistic_regressor = logistic_regressor
        self._logistic_regressor_coef = torch.from_numpy(self._logistic_regressor.coef_)
        self._logistic_regressor_intercept = torch.from_numpy(
            self._logistic_regressor.intercept_
        )
        self._is_logistic_regressor_constructed = True

    @staticmethod
    def _pool_feature(feature: Tensor) -> Tensor:
        """Pools the feature tensor based on its shape.

        Args:
            feature: Input feature tensor.

        Returns:
            Pooled feature tensor.

        Raises:
            ValueError: If the network structure is invalid.
        """
        shape = feature.shape[1:]
        if len(shape) == 1:
            return feature
        if len(shape) == 2:
            return feature.mean(dim=1)  # collapse dimension 1
        if len(shape) == 3:
            return feature.mean(dim=(2, 3))
        msg = "Invalid network structure"
        raise ValueError(msg)

    def _attach_hooks(self, chosen_layers: list[nn.Module]) -> None:
        """Attaches hooks to the chosen layers for feature extraction.

        Args:
            chosen_layers: List of layers to attach hooks to.
        """

        def hook(model: nn.Module, input: Tensor, output: Tensor) -> None:
            del model, input
            self._feature_list.append(self._pool_feature(output))

        # Attach hooks to all children layers
        for layer in chosen_layers:
            handle = layer.register_forward_hook(hook)
            self._hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        """Removes all attached hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _calculate_noisy_mahalanobis_scores(self, inputs: Tensor) -> Tensor:
        """Calculates noisy Mahalanobis scores for the input.

        Args:
            inputs: Input tensor.

        Returns:
            Tensor of noisy Mahalanobis scores.
        """
        noisy_mahalanobis_scores = torch.empty(0).to(
            next(self.model.parameters()).device
        )

        for layer_idx in range(self._num_layers):
            with torch.enable_grad():
                inputs.requires_grad_(True)
                self._feature_list.clear()
                self.model(inputs)

                gradients = self._compute_gradients(
                    inputs=inputs,
                    features=self._feature_list[layer_idx],
                    num_classes=self.model.num_classes,
                    class_means=getattr(self, f"_class_means_{layer_idx}"),
                    precision_matrix=getattr(self, f"_precisions_{layer_idx}"),
                )

                inputs.grad = None
                inputs.requires_grad_(False)

            temp_inputs = inputs - self._magnitude * gradients

            # Populate feature_list
            self._feature_list.clear()
            self.model(temp_inputs)

            noisy_mahalanobis_scores_layer = self._compute_gaussian_scores(
                features=self._feature_list[layer_idx],
                num_classes=self.model.num_classes,
                class_means=getattr(self, f"_class_means_{layer_idx}"),
                precision_matrix=getattr(self, f"_precisions_{layer_idx}"),
            ).max(dim=1, keepdim=True)[0]  # [B, 1]

            noisy_mahalanobis_scores = torch.cat(
                [noisy_mahalanobis_scores, noisy_mahalanobis_scores_layer], dim=1
            )  # [B, L]

        return noisy_mahalanobis_scores.cpu()  # [B, L]

    def _calculate_noisy_mahalanobis_scores_loader(
        self,
        loader: DataLoader | PrefetchLoader,
        max_num_samples: int,
        channels_last: bool,
    ) -> Tensor:
        """Computes Mahalanobis confidence scores on the input loader.

        Args:
            loader: DataLoader or PrefetchLoader for input data.
            max_num_samples: Maximum number of samples to process.
            channels_last: Whether a channels_last memory layout should be used.

        Returns:
            Tensor of Mahalanobis scores.
        """
        mahalanobis_scores = torch.empty(0)

        num_samples = 0
        device = next(self.model.parameters()).device
        for inputs, _ in loader:
            if num_samples + inputs.shape[0] > max_num_samples:
                overhead = num_samples + inputs.shape[0] - max_num_samples
                modified_batch_size = inputs.shape[0] - overhead
                inputs = inputs[:modified_batch_size]

            if not isinstance(loader, PrefetchLoader):
                inputs = inputs.to(device)

            if channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            noisy_mahalanobis_scores = self._calculate_noisy_mahalanobis_scores(
                inputs
            )  # [B, L]

            mahalanobis_scores = torch.cat(
                [mahalanobis_scores, noisy_mahalanobis_scores], dim=0
            )  # [N, L]

            num_samples += inputs.shape[0]
            if num_samples > max_num_samples:
                break

        return mahalanobis_scores  # [N, L]

    @staticmethod
    def _get_layer_candidates(
        model: nn.Module, module_type: type | None, module_name_regex: str | None
    ) -> list[nn.Module]:
        """Retrieves layer candidates based on type and name pattern.

        Args:
            model: The model to extract layers from.
            module_type: Type of modules to consider.
            module_name_regex: Regex pattern for module names.

        Returns:
            List of candidate layers.
        """
        layer_candidates = []

        compiled_module_name_regex = (
            re.compile(module_name_regex) if module_name_regex is not None else None
        )

        for name, module in model.named_modules():
            if (
                compiled_module_name_regex is not None
                and compiled_module_name_regex.match(name)
            ) or (module_type is not None and isinstance(module, module_type)):
                layer_candidates.append(module)

        return layer_candidates

    @staticmethod
    def _filter_layer_candidates(
        layer_candidates: list[nn.Module], num_hooks: int | None
    ) -> list[nn.Module]:
        """Filters layer candidates based on the number of hooks.

        Args:
            layer_candidates: List of candidate layers.
            num_hooks: Number of hooks to use.

        Returns:
            Filtered list of layers.
        """
        if num_hooks is None or num_hooks >= len(layer_candidates):
            return layer_candidates

        chosen_indices = torch.linspace(
            start=0, end=len(layer_candidates) - 1, steps=num_hooks
        ).long()

        chosen_layers = [layer_candidates[i] for i in chosen_indices]

        return chosen_layers

    def _reconstruct_logistic_regressor(self) -> None:
        """Reconstructs the logistic regressor from stored coefficients and intercept.

        Raises:
            ValueError: If logistic regressor weights are not set.
        """
        if (
            self._logistic_regressor_coef is None
            or self._logistic_regressor_intercept is None
        ):
            msg = "Logistic regressor weights are not set, nothing to reconstruct"
            raise ValueError(msg)

        self._logistic_regressor = LogisticRegression()
        self._logistic_regressor.coef_ = self._logistic_regressor_coef.numpy()
        self._logistic_regressor.intercept_ = self._logistic_regressor_intercept.numpy()
        self._is_logistic_regressor_constructed = True

    def _calculate_gaussian_parameters(
        self,
        train_loader: DataLoader,
        max_num_training_samples: int | None,
        channels_last: bool,
    ) -> None:
        """Calculates Gaussian parameters for each layer and class.

        Args:
            train_loader: DataLoader for training data.
            max_num_training_samples: Maximum number of training samples to use.
            channels_last: Whether a channels_last memory layout should be used.
        """
        if max_num_training_samples is None:
            max_num_training_samples = len(train_loader.dataset)

        num_classes = self.model.num_classes
        num_layers = self._num_layers
        # Initialize tensors for storing features
        features_per_class_per_layer = [
            [[] for _ in range(num_classes)] for _ in range(num_layers)
        ]  # [L, C, *]

        # Process each batch
        num_training_samples = 0
        device = next(self.model.parameters()).device
        for inputs, targets in train_loader:
            # Truncate last batch to have exactly the maximum number of training samples
            if num_training_samples + inputs.shape[0] > max_num_training_samples:
                overhead = (
                    num_training_samples + inputs.shape[0] - max_num_training_samples
                )
                modified_batch_size = inputs.shape[0] - overhead
                inputs = inputs[:modified_batch_size]
                targets = targets[:modified_batch_size]

            if not isinstance(train_loader, PrefetchLoader):
                inputs = inputs.to(device)
            else:
                targets = targets.cpu()

            if channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            self._feature_list.clear()
            self.model(inputs)
            feature_list = self._feature_list

            # Process each layer's output
            for layer_idx, feature in enumerate(feature_list):
                feature = feature.detach().cpu()
                for class_idx in range(num_classes):
                    class_features = feature[targets == class_idx]  # [N_{LC}, D_L]
                    if class_features.shape[0] > 0:
                        features_per_class_per_layer[layer_idx][class_idx].append(
                            class_features
                        )

            num_training_samples += inputs.shape[0]
            if num_training_samples >= max_num_training_samples:
                break

        # Aggregate and compute means and precision
        class_means = []  # [L, C, D_L]
        precisions = []  # [L, D_L, D_L]
        for i in range(num_layers):
            num_features = features_per_class_per_layer[i][0][0].shape[-1]
            class_means.append(
                torch.empty(
                    (num_classes, num_features),
                    device=device,
                )
            )
            precisions.append(torch.empty((num_features, num_features), device=device))

        for layer_idx in range(num_layers):
            layer_features = torch.empty(0)
            for class_idx in range(num_classes):
                # Concatenate all features for this class across batches
                class_features = torch.cat(
                    features_per_class_per_layer[layer_idx][class_idx], dim=0
                )  # [N_L, D_L]
                class_mean = class_features.mean(dim=0)  # [D_L]
                class_means[layer_idx][class_idx] = class_mean.to(device)

                centered_class_features = class_features - class_mean  # [N_L, D_L]

                # Aggregate features for precision calculation
                layer_features = torch.cat(
                    [layer_features, centered_class_features], dim=0
                )

            # Compute precision
            covariance = layer_features.T @ layer_features / layer_features.shape[0]
            precision = torch.linalg.pinv(covariance, hermitian=True)
            precisions[layer_idx] = precision.to(device)

        for i in range(self._num_layers):
            setattr(self, f"_class_means_{i}", class_means[i])
            setattr(self, f"_precisions_{i}", precisions[i])

    @staticmethod
    def _compute_gaussian_scores(
        features: Tensor,
        num_classes: int,
        class_means: Tensor,
        precision_matrix: Tensor,
    ) -> Tensor:
        """Computes Gaussian scores for the given features.

        Args:
            features: Input feature tensor.
            num_classes: Number of classes.
            class_means: Tensor of class means.
            precision_matrix: Precision matrix.

        Returns:
            Tensor of Gaussian scores.
        """
        scores = []
        for class_idx in range(num_classes):
            difference = features.detach() - class_means[class_idx]  # [B, D_L]
            term = -0.5 * (difference @ precision_matrix @ difference.T).diag()  # [B]
            scores.append(term.unsqueeze(dim=1))  # [B, 1]

        return torch.cat(scores, dim=1)  # [B, C]

    @staticmethod
    def _compute_gradients(
        inputs: Tensor,
        features: Tensor,
        num_classes: int,
        class_means: Tensor,
        precision_matrix: Tensor,
    ) -> Tensor:
        """Computes gradients for the Mahalanobis score calculation.

        Args:
            inputs: Input tensor.
            features: Feature tensor.
            num_classes: Number of classes.
            class_means: Tensor of class means.
            precision_matrix: Precision matrix.

        Returns:
            Tensor of gradient signs.
        """
        gaussian_scores = MahalanobisWrapper._compute_gaussian_scores(
            features, num_classes, class_means, precision_matrix
        )

        max_score_indices = gaussian_scores.max(dim=1)[1]  # [B]
        max_means = class_means.index_select(dim=0, index=max_score_indices)  # [B, D_L]
        difference = features - max_means  # [B, D_L]
        term = -0.5 * (difference @ precision_matrix @ difference.T).diag()  # [B]
        loss = -term.mean()  # []

        gradient_signs = torch.autograd.grad(outputs=loss, inputs=inputs)[
            0
        ].sign()  # [B, C, H, W]

        return gradient_signs
