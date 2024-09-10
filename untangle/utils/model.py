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


def create_model(model_name, pretrained, num_classes, in_chans, model_kwargs):
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
    model,
    model_wrapper_name,
    reset_classifier,
    weight_paths,
    num_hidden_features,
    mlp_depth,
    stopgrad,
    num_hooks,
    module_type,
    module_name_regex,
    dropout_probability,
    use_filterwise_dropout,
    num_mc_samples,
    num_mc_samples_integral,
    num_mc_samples_cv,
    rbf_length_scale,
    ema_momentum,
    matrix_rank,
    use_het,
    temperature,
    pred_type,
    hessian_structure,
    use_low_rank_cov,
    max_rank,
    magnitude,
    num_heads,
    use_spectral_normalization,
    spectral_normalization_iteration,
    spectral_normalization_bound,
    use_spectral_normalized_batch_norm,
    use_tight_norm_for_pointwise_convs,
    num_random_features,
    gp_kernel_scale,
    gp_output_bias,
    gp_random_feature_type,
    use_input_normalized_gp,
    gp_cov_momentum,
    gp_cov_ridge_penalty,
    gp_input_dim,
    latent_dim,
    num_density_components,
    use_batched_flow,
    edl_activation,
    checkpoint_path,
):
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
        )
    else:
        msg = (
            f'Model wrapper "{model_wrapper_name}" is currently not implemented or you '
            f"made a typo"
        )
        raise ValueError(msg)

    if checkpoint_path:
        load_checkpoint(wrapped_model, checkpoint_path)

    num_params = sum(param.numel() for param in model.parameters())
    logger.info(
        f"Wrapper {model_wrapper_name} created, total param count: {num_params}."
    )
    logger.info(str(model))

    return wrapped_model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> dict[str, Any]:
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]
        logger.info(f"Loaded state_dict from checkpoint '{checkpoint_path}'.")
    else:
        msg = f"No checkpoint found at '{checkpoint_path}'"
        raise FileNotFoundError(msg)

    incompatible_keys = model.load_state_dict(state_dict, strict=True)

    return incompatible_keys
