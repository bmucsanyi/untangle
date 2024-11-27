"""Model utilities."""

import logging
from typing import Any

import torch
from timm.models import create_model as create_timm_model

from untangle.models import (
    resnet_50,
    resnet_c_preact_26,
    resnet_fixup_50,
    wide_resnet_c_26_10,
    wide_resnet_c_fixup_26_10,
    wide_resnet_c_preact_26_10,
)
from untangle.wrappers import (
    CEBaselineWrapper,
    CorrectnessPredictionWrapper,
    DDUWrapper,
    DeepCorrectnessPredictionWrapper,
    DeepEnsembleWrapper,
    DeepLossPredictionWrapper,
    DUQWrapper,
    EDLWrapper,
    FastDeepEnsembleWrapper,
    HetClassNNWrapper,
    HETXLWrapper,
    LaplaceWrapper,
    LossPredictionWrapper,
    MahalanobisWrapper,
    MCDropoutWrapper,
    PostNetWrapper,
    ShallowEnsembleWrapper,
    SNGPWrapper,
    SWAGWrapper,
    TemperatureWrapper,
)

logger = logging.getLogger(__name__)

UNTANGLE_STR_TO_MODEL_CLASS = {
    "resnet_50": resnet_50,
    "resnet_fixup_50": resnet_fixup_50,
    "wide_resnet_c_26_10": wide_resnet_c_26_10,
    "wide_resnet_c_fixup_26_10": wide_resnet_c_fixup_26_10,
    "wide_resnet_c_preact_26_10": wide_resnet_c_preact_26_10,
    "resnet_c_preact_26": resnet_c_preact_26,
}


def create_model(
    model_name: str,
    pretrained: bool,
    num_classes: int,
    in_chans: int,
    model_kwargs: dict,
) -> torch.nn.Module:
    """Creates a model based on the given parameters.

    Args:
        model_name: Name of the model to create.
        pretrained: Whether to use pretrained weights.
        num_classes: Number of classes for the model.
        in_chans: Number of input channels.
        model_kwargs: Additional keyword arguments for model creation.

    Returns:
        The created model.

    Raises:
        ValueError: If an invalid prefix is provided in the model_name.
    """
    prefix, model_name = model_name.split("/")

    if prefix == "timm":
        model = create_timm_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **model_kwargs,
        )
    elif prefix == "untangle":
        kwargs = dict(
            num_classes=num_classes,
            in_chans=in_chans,
            **model_kwargs,
        )

        if model_name in {"resnet_fixup_50", "resnet_50"}:
            kwargs["pretrained"] = pretrained

        model = UNTANGLE_STR_TO_MODEL_CLASS[model_name](**kwargs)
    else:
        msg = f"Invalid prefix '{prefix}' provided."
        raise ValueError(msg)

    num_params = sum(param.numel() for param in model.parameters())
    logger.info(f"Model {model_name} created, param count: {num_params}.")

    return model


def wrap_model(  # noqa: C901
    model: torch.nn.Module,
    model_wrapper_name: str,
    reset_classifier: bool,
    weight_paths: list[str],
    num_hidden_features: int,
    mlp_depth: int,
    stopgrad: bool,
    num_hooks: int,
    module_type: str,
    module_name_regex: str,
    dropout_probability: float,
    use_filterwise_dropout: bool,
    num_mc_samples: int,
    num_mc_samples_integral: int,
    num_mc_samples_cv: int,
    rbf_length_scale: float,
    ema_momentum: float,
    matrix_rank: int,
    use_het: bool,
    temperature: float,
    pred_type: str,
    hessian_structure: str,
    use_low_rank_cov: bool,
    max_rank: int,
    magnitude: float,
    num_heads: int,
    likelihood: str,
    use_spectral_normalization: bool,
    spectral_normalization_iteration: int,
    spectral_normalization_bound: float,
    use_spectral_normalized_batch_norm: bool,
    use_tight_norm_for_pointwise_convs: bool,
    num_random_features: int,
    gp_kernel_scale: float,
    gp_output_bias: bool,
    gp_random_feature_type: str,
    use_input_normalized_gp: bool,
    gp_cov_momentum: float,
    gp_cov_ridge_penalty: float,
    gp_input_dim: int,
    latent_dim: int,
    num_density_components: int,
    use_batched_flow: bool,
    edl_activation: str,
    checkpoint_path: str,
) -> torch.nn.Module:
    """Wraps the given model with the specified wrapper.

    Args:
        model: The model to be wrapped.
        model_wrapper_name: Name of the wrapper to use.
        reset_classifier: Whether to reset the classifier.
        weight_paths: Paths to model weights.
        num_hidden_features: Number of hidden features.
        mlp_depth: Depth of the MLP.
        stopgrad: Whether to stop gradient propagation.
        num_hooks: Number of hooks to use.
        module_type: Type of module.
        module_name_regex: Regex for module names.
        dropout_probability: Probability of dropout.
        use_filterwise_dropout: Whether to use filterwise dropout.
        num_mc_samples: Number of Monte Carlo samples.
        num_mc_samples_integral: Number of Monte Carlo samples for integral.
        num_mc_samples_cv: Number of Monte Carlo samples for cross-validation.
        rbf_length_scale: Length scale for RBF kernel.
        ema_momentum: Momentum for EMA.
        matrix_rank: Rank of the matrix.
        use_het: Whether to use HET.
        temperature: Temperature for scaling.
        pred_type: Type of prediction.
        hessian_structure: Structure of the Hessian.
        use_low_rank_cov: Whether to use low-rank covariance.
        max_rank: Maximum rank.
        magnitude: Magnitude value.
        num_heads: Number of heads.
        likelihood: Likelihood function.
        use_spectral_normalization: Whether to use spectral normalization.
        spectral_normalization_iteration: Number of iterations for spectral
            normalization.
        spectral_normalization_bound: Bound for spectral normalization.
        use_spectral_normalized_batch_norm: Whether to use spectral normalized batch
            norm.
        use_tight_norm_for_pointwise_convs: Whether to use tight norm for pointwise
            convolutions.
        num_random_features: Number of random features.
        gp_kernel_scale: Scale for GP kernel.
        gp_output_bias: Whether to use output bias for GP.
        gp_random_feature_type: Type of random features for GP.
        use_input_normalized_gp: Whether to use input normalized GP.
        gp_cov_momentum: Momentum for GP covariance.
        gp_cov_ridge_penalty: Ridge penalty for GP covariance.
        gp_input_dim: Input dimension for GP.
        latent_dim: Dimension of latent space.
        num_density_components: Number of density components.
        use_batched_flow: Whether to use batched flow.
        edl_activation: Activation function for EDL.
        checkpoint_path: Path to the checkpoint.

    Returns:
        The wrapped model.

    Raises:
        ValueError: If the specified model wrapper is not implemented.
    """
    if reset_classifier:
        model.reset_classifier(model.num_classes)

    if model_wrapper_name == "correctness-prediction":
        wrapped_model = CorrectnessPredictionWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            mlp_depth=mlp_depth,
            stopgrad=stopgrad,
        )
    elif model_wrapper_name == "deep-correctness-prediction":
        wrapped_model = DeepCorrectnessPredictionWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            mlp_depth=mlp_depth,
            stopgrad=stopgrad,
            num_hooks=num_hooks,
            module_type=module_type,
            module_name_regex=module_name_regex,
        )
    elif model_wrapper_name == "temperature-scaling":
        wrapped_model = TemperatureWrapper(model=model, weight_path=weight_paths[0])
    elif model_wrapper_name == "deep-ensemble":
        return DeepEnsembleWrapper(
            model=model,
            weight_paths=weight_paths,
        )
    elif model_wrapper_name == "fast-deep-ensemble":
        return FastDeepEnsembleWrapper(
            model=model,
            weight_paths=weight_paths,
        )
    elif model_wrapper_name == "ce-baseline":
        wrapped_model = CEBaselineWrapper(model=model)
    elif model_wrapper_name == "mc-dropout":
        wrapped_model = MCDropoutWrapper(
            model=model,
            dropout_probability=dropout_probability,
            use_filterwise_dropout=use_filterwise_dropout,
            num_mc_samples=num_mc_samples,
        )
    elif model_wrapper_name == "duq":
        wrapped_model = DUQWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            rbf_length_scale=rbf_length_scale,
            ema_momentum=ema_momentum,
        )
    elif model_wrapper_name == "ddu":
        wrapped_model = DDUWrapper(
            model=model,
            use_spectral_normalization=use_spectral_normalization,
            spectral_normalization_iteration=spectral_normalization_iteration,
            spectral_normalization_bound=spectral_normalization_bound,
            use_spectral_normalized_batch_norm=use_spectral_normalized_batch_norm,
            use_tight_norm_for_pointwise_convs=use_tight_norm_for_pointwise_convs,
        )
    elif model_wrapper_name == "het-xl":
        wrapped_model = HETXLWrapper(
            model=model,
            matrix_rank=matrix_rank,
            num_mc_samples=num_mc_samples,
            temperature=temperature,
            use_het=use_het,
        )
    elif model_wrapper_name == "laplace":
        wrapped_model = LaplaceWrapper(
            model=model,
            num_mc_samples=num_mc_samples,
            num_mc_samples_cv=num_mc_samples_cv,
            weight_path=weight_paths[0],
            pred_type=pred_type,
            hessian_structure=hessian_structure,
        )
    elif model_wrapper_name == "swag":
        wrapped_model = SWAGWrapper(
            model=model,
            weight_path=weight_paths[0],
            use_low_rank_cov=use_low_rank_cov,
            max_rank=max_rank,
        )
    elif model_wrapper_name == "mahalanobis":
        wrapped_model = MahalanobisWrapper(
            model=model,
            magnitude=magnitude,
            weight_path=weight_paths[0],
            num_hooks=num_hooks,
            module_type=module_type,
            module_name_regex=module_name_regex,
        )
    elif model_wrapper_name == "hetclassnn":
        wrapped_model = HetClassNNWrapper(
            model=model,
            dropout_probability=dropout_probability,
            use_filterwise_dropout=use_filterwise_dropout,
            num_mc_samples=num_mc_samples,
            num_mc_samples_integral=num_mc_samples_integral,
        )
    elif model_wrapper_name == "loss-prediction":
        wrapped_model = LossPredictionWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            mlp_depth=mlp_depth,
            stopgrad=stopgrad,
        )
    elif model_wrapper_name == "edl":
        wrapped_model = EDLWrapper(model=model, activation=edl_activation)
    elif model_wrapper_name == "postnet":
        wrapped_model = PostNetWrapper(
            model=model,
            latent_dim=latent_dim,
            hidden_dim=num_hidden_features,
            num_density_components=num_density_components,
            use_batched_flow=use_batched_flow,
        )
    elif model_wrapper_name == "deep-loss-prediction":
        wrapped_model = DeepLossPredictionWrapper(
            model=model,
            num_hidden_features=num_hidden_features,
            mlp_depth=mlp_depth,
            stopgrad=stopgrad,
            num_hooks=num_hooks,
            module_type=module_type,
            module_name_regex=module_name_regex,
        )
    elif model_wrapper_name == "shallow-ensemble":
        wrapped_model = ShallowEnsembleWrapper(model=model, num_heads=num_heads)
    elif model_wrapper_name == "sngp":
        wrapped_model = SNGPWrapper(
            model=model,
            use_spectral_normalization=use_spectral_normalization,
            use_tight_norm_for_pointwise_convs=use_tight_norm_for_pointwise_convs,
            spectral_normalization_iteration=spectral_normalization_iteration,
            spectral_normalization_bound=spectral_normalization_bound,
            use_spectral_normalized_batch_norm=use_spectral_normalized_batch_norm,
            num_mc_samples=num_mc_samples,
            num_random_features=num_random_features,
            gp_kernel_scale=gp_kernel_scale,
            gp_output_bias=gp_output_bias,
            gp_random_feature_type=gp_random_feature_type,
            use_input_normalized_gp=use_input_normalized_gp,
            gp_cov_momentum=gp_cov_momentum,
            gp_cov_ridge_penalty=gp_cov_ridge_penalty,
            gp_input_dim=gp_input_dim,
            likelihood=likelihood,
        )
    else:
        msg = (
            f'Model wrapper "{model_wrapper_name}" is currently not implemented or you '
            f"made a typo"
        )
        raise ValueError(msg)

    if checkpoint_path:
        load_checkpoint(wrapped_model, checkpoint_path)

    num_params = sum(param.numel() for param in wrapped_model.parameters())

    logger.info(
        f"Wrapper {model_wrapper_name} created, total param count: {num_params}."
    )
    logger.info(str(wrapped_model))

    return wrapped_model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Loads a checkpoint into the given model.

    Args:
        model: The model to load the checkpoint into.
        checkpoint_path: Path to the checkpoint file.

    Returns:
        A dictionary containing information about incompatible keys.

    Raises:
        FileNotFoundError: If no checkpoint is found at the specified path.
    """
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        logger.info(f"Loaded state_dict from checkpoint '{checkpoint_path}'.")
    else:
        msg = f"No checkpoint found at '{checkpoint_path}'"
        raise FileNotFoundError(msg)

    incompatible_keys = model.load_state_dict(state_dict, strict=True)

    return incompatible_keys
