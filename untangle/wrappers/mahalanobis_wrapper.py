"""Mahalanobis implementation as a wrapper class.

Latent density estimation based on https://github.com/pokaxpoka/deep_Mahalanobis_detector.
"""

import re

import torch
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from torch import nn

from untangle.wrappers.model_wrapper import SpecialWrapper


class MahalanobisWrapper(SpecialWrapper):
    """This module takes a model as input and creates a Mahalanobis model from it."""

    def __init__(
        self,
        model: nn.Module,
        magnitude: float,
        weight_path: str,
        num_hooks=None,
        module_type=None,
        module_name_regex=None,
    ):
        super().__init__(model)

        self._magnitude = magnitude
        self._weight_path = weight_path
        self._num_hooks = num_hooks
        self._module_type = module_type
        self._module_name_regex = module_name_regex

        # Register hooks to extract intermediate features
        self._feature_list = []
        self._hook_handles = []
        self._hook_layer_names = []
        self._logistic_regressor = None
        self.register_buffer("_logistic_regressor_coef", None)
        self.register_buffer("_logistic_regressor_intercept", None)

        layer_candidates = self._get_layer_candidates(
            model=self.model,
            module_type=self._module_type,
            module_name_regex=self._module_name_regex,
        )
        chosen_layers = self._filter_layer_candidates(
            layer_candidates=layer_candidates, num_hooks=self._num_hooks
        )
        self._num_layers = len(chosen_layers)
        self._attach_hooks(chosen_layers=chosen_layers)

        for i in range(self._num_layers):
            self.register_buffer(f"_class_means_{i}", None)
            self.register_buffer(f"_precisions_{i}", None)

        self._load_model()

    @staticmethod
    def forward_head(x, *, pre_logits: bool = False):
        del x, pre_logits
        msg = "Head cannot be called separately"
        raise ValueError(msg)

    def forward(self, inputs):
        # This is the only way to interact with the model.
        if self._logistic_regressor is None:
            self._reconstruct_logistic_regressor()

        self._feature_list.clear()
        features = self.model.forward_head(
            self.model.forward_features(inputs), pre_logits=True
        )
        ret_logits = self.model.get_classifier()(features)

        noisy_mahalanobis_scores = self._calculate_noisy_mahalanobis_scores(
            inputs
        )  # [B, L]

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
        train_loader,
        id_loader,
        ood_loader,
        max_num_training_samples,
        max_num_id_ood_samples,
        args,
    ):
        self._calculate_gaussian_parameters(
            train_loader=train_loader,
            max_num_training_samples=max_num_training_samples,
            args=args,
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
            loader=id_loader, max_num_samples=max_num_id_ood_samples, args=args
        )
        labels_id = torch.zeros(mahalanobis_scores_id.shape[0])

        # Get Mahalanobis scores for out-of-distribution data
        mahalanobis_scores_ood = self._calculate_noisy_mahalanobis_scores_loader(
            loader=ood_loader, max_num_samples=max_num_id_ood_samples, args=args
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

    @staticmethod
    def _pool_feature(feature):
        shape = feature.shape[1:]
        if len(shape) == 1:
            return feature
        if len(shape) == 2:
            return feature.mean(dim=1)  # collapse dimension 1
        if len(shape) == 3:
            return feature.mean(dim=(2, 3))
        msg = "Invalid network structure"
        raise ValueError(msg)

    def _attach_hooks(self, chosen_layers):
        def hook(model, input, output):
            del model, input
            self._feature_list.append(self._pool_feature(output))

        # Attach hooks to all children layers
        for layer in chosen_layers:
            handle = layer.register_forward_hook(hook)
            self._hook_handles.append(handle)

    def _remove_hooks(self):
        # Remove all hooks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _calculate_noisy_mahalanobis_scores(self, inputs):
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

                inputs.grad.zero_()
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

    def _calculate_noisy_mahalanobis_scores_loader(self, loader, max_num_samples, args):
        """Compute Mahalanobis confidence score on input loader."""
        mahalanobis_scores = torch.empty(0)

        num_samples = 0
        device = next(self.model.parameters()).device
        for inputs, _ in loader:
            if num_samples + inputs.shape[0] > max_num_samples:
                overhead = num_samples + inputs.shape[0] - max_num_samples
                modified_batch_size = inputs.shape[0] - overhead
                inputs = inputs[:modified_batch_size]

            if not args.prefetcher:
                inputs = inputs.to(device)

            if args.channels_last:
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
    def _get_layer_candidates(model, module_type, module_name_regex):
        layer_candidates = []

        if module_name_regex is not None:
            module_name_regex = re.compile(module_name_regex)

        for name, module in model.named_modules():
            if (module_name_regex is not None and module_name_regex.match(name)) or (
                module_type is not None and isinstance(module, module_type)
            ):
                layer_candidates.append(module)

        return layer_candidates

    @staticmethod
    def _filter_layer_candidates(layer_candidates, num_hooks):
        if num_hooks is None:
            return layer_candidates

        num_hooks = min(num_hooks, len(layer_candidates))

        chosen_layers = []
        chosen_indices = torch.linspace(
            start=0, end=len(layer_candidates) - 1, steps=num_hooks
        )
        chosen_layers = [
            module for i, module in enumerate(layer_candidates) if i in chosen_indices
        ]

        return chosen_layers

    def _reconstruct_logistic_regressor(self):
        if (
            self.logistic_regressor_coef is None
            or self._logistic_regressor_intercept is None
        ):
            msg = "Logistic regressor weights are not set, nothing to reconstruct"
            raise ValueError(msg)

        self._logistic_regressor = LogisticRegression()
        self._logistic_regressor.coef_ = self.logistic_regressor_coef.numpy()
        self._logistic_regressor.intercept_ = self._logistic_regressor_intercept.numpy()

    def _calculate_gaussian_parameters(
        self, train_loader, max_num_training_samples, args
    ):
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

            if not args.prefetcher:
                inputs = inputs.to(device)
            else:
                targets = targets.cpu()

            if args.channels_last:
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
    def _compute_gaussian_scores(features, num_classes, class_means, precision_matrix):
        scores = []
        for class_idx in range(num_classes):
            difference = features.detach() - class_means[class_idx]  # [B, D_L]
            term = -0.5 * (difference @ precision_matrix @ difference.T).diag()  # [B]
            scores.append(term.unsqueeze(dim=1))  # [B, 1]

        return torch.cat(scores, dim=1)  # [B, C]

    @staticmethod
    def _compute_gradients(
        inputs, features, num_classes, class_means, precision_matrix
    ):
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
