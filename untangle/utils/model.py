"""Model utilities."""

import logging
from typing import Any

import torch
from timm.models import create_model as create_timm_model

from untangle.models import (
    resnet_50,
    resnet_c_preact_26,
    resnet_fixup_50,
    simple_convnet_3_32,
    simple_convnet_3_256,
    wide_resnet_c_26_10,
    wide_resnet_c_fixup_26_10,
    wide_resnet_c_preact_26_10,
)
from untangle.wrappers import (
    BaselineWrapper,
    CovariancePushforwardLaplaceWrapper,
    EDLWrapper,
    HETWrapper,
    LinearizedSWAGWrapper,
    PostNetWrapper,
    SamplePushforwardLaplaceWrapper,
    SNGPWrapper,
    SWAGWrapper,
)

logger = logging.getLogger(__name__)

UNTANGLE_STR_TO_MODEL_CLASS = {
    "resnet_50": resnet_50,
    "resnet_fixup_50": resnet_fixup_50,
    "wide_resnet_c_26_10": wide_resnet_c_26_10,
    "wide_resnet_c_fixup_26_10": wide_resnet_c_fixup_26_10,
    "wide_resnet_c_preact_26_10": wide_resnet_c_preact_26_10,
    "resnet_c_preact_26": resnet_c_preact_26,
    "simple_convnet_3_256": simple_convnet_3_256,
    "simple_convnet_3_32": simple_convnet_3_32,
}


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    verbose: bool,
) -> dict[str, Any]:
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]

        tmp_state_dict = {}
        for k, v in state_dict.items():
            # TODO(bmucsanyi): Fix distributed saving
            if k.startswith("module."):
                k = k[7:]

            if k.startswith("model."):
                k = k[6:]

            tmp_state_dict[k] = v

        state_dict = tmp_state_dict

        if verbose:
            logger.info(f"Loaded state_dict from checkpoint '{checkpoint_path}'.")
    else:
        msg = f"No checkpoint found at '{checkpoint_path}'"
        raise FileNotFoundError(msg)

    incompatible_keys = model.load_state_dict(state_dict, strict=True)

    return incompatible_keys


def create_model(
    model_name,
    pretrained,
    num_classes,
    in_chans,
    model_kwargs,
    verbose,
    model_checkpoint_path,
):
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

        if model_name == "resnet_50":
            kwargs["pretrained"] = pretrained

        model = UNTANGLE_STR_TO_MODEL_CLASS[model_name](**kwargs)
    else:
        msg = f"Invalid prefix '{prefix}' provided."
        raise ValueError(msg)

    if model_checkpoint_path:
        load_model_checkpoint(model, model_checkpoint_path, verbose=verbose)

    num_params = sum(param.numel() for param in model.parameters())

    if verbose:
        logger.info(f"Model {model_name} created, param count: {num_params}.")

    return model


def wrap_model(
    model,
    model_wrapper_name,
    reset_classifier,
    weight_paths,
    num_hidden_features,
    num_mc_samples,
    matrix_rank,
    mask_regex,
    use_sampling,
    temperature,
    use_low_rank_cov,
    max_rank,
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
    loss_function,
    predictive_fn,
    use_eigval_prior,
    gp_likelihood,
    verbose,
):
    if reset_classifier:
        model.reset_classifier(model.num_classes)

    if model_wrapper_name == "baseline":
        wrapped_model = BaselineWrapper(model=model)
    elif model_wrapper_name == "het":
        wrapped_model = HETWrapper(
            model=model,
            matrix_rank=matrix_rank,
            num_mc_samples=num_mc_samples,
            temperature=temperature,
            use_sampling=use_sampling,
        )
    elif model_wrapper_name == "laplace":
        kwargs = {
            "model": model,
            "loss_function": loss_function,
            "rank": matrix_rank,
            "predictive_fn": predictive_fn,
            "use_eigval_prior": use_eigval_prior,
            "mask_regex": mask_regex,
        }
        if use_sampling:
            kwargs["num_mc_samples"] = num_mc_samples
            wrapped_model = SamplePushforwardLaplaceWrapper(**kwargs)
        else:
            wrapped_model = CovariancePushforwardLaplaceWrapper(**kwargs)
    elif model_wrapper_name == "swag":
        kwargs = {
            "model": model,
            "weight_path": weight_paths[0],
            "use_low_rank_cov": use_low_rank_cov,
            "max_rank": max_rank,
        }
        if use_sampling:
            wrapped_model = SWAGWrapper(**kwargs)
        else:
            wrapped_model = LinearizedSWAGWrapper(**kwargs)
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
            likelihood=gp_likelihood,
        )
    else:
        msg = (
            f'Model wrapper "{model_wrapper_name}" is currently not implemented or you '
            f"made a typo"
        )
        raise ValueError(msg)

    if checkpoint_path:
        load_checkpoint(wrapped_model, checkpoint_path, verbose=verbose)

    num_params = sum(param.numel() for param in wrapped_model.parameters())

    if verbose:
        logger.info(
            f"Wrapper {model_wrapper_name} created, total param count: {num_params}."
        )
        logger.info(str(wrapped_model))

    return wrapped_model


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    verbose: bool,
) -> dict[str, Any]:
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["state_dict"]

        tmp_state_dict = {}
        for k, v in state_dict.items():
            # TODO(bmucsanyi): Fix distributed saving
            if k.startswith("module."):
                tmp_state_dict[k[7:]] = v
            else:
                tmp_state_dict[k] = v

        state_dict = tmp_state_dict

        if verbose:
            logger.info(f"Loaded state_dict from checkpoint '{checkpoint_path}'.")
    else:
        msg = f"No checkpoint found at '{checkpoint_path}'"
        raise FileNotFoundError(msg)

    incompatible_keys = model.load_state_dict(state_dict, strict=True)

    return incompatible_keys
