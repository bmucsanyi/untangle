"""Collection of utilities for the untangle package."""

from .checkpoint_saver import CheckpointSaver
from .context import DefaultContext
from .convolution import calculate_output_padding, calculate_same_padding
from .dataset import create_dataset
from .loader import PrefetchLoader, create_loader
from .logging import log_wandb, setup_logging
from .loss import create_loss_fn
from .metric import (
    AverageMeter,
    accuracy,
    area_under_lift_curve,
    area_under_risk_coverage_curve,
    auroc,
    binary_brier,
    binary_log_probability,
    calculate_bin_metrics,
    calibration_error,
    centered_cov,
    coverage_for_accuracy,
    cross_entropy,
    dempster_shafer_metric,
    entropy,
    excess_area_under_risk_coverage_curve,
    get_ranks,
    is_correct_pred,
    kl_divergence,
    multiclass_brier,
    multiclass_log_probability,
    pearsonr,
    relative_area_under_lift_curve,
    spearmanr,
)
from .model import create_model, wrap_model
from .parsing import (
    float_tuple,
    int_tuple,
    kwargs,
    parse_args,
    resolve_data_config,
    string_tuple,
)
from .random import set_random_seed
from .replace import (
    ModuleData,
    deep_setattr,
    register,
    register_cond,
    replace,
    replace_cond,
)
from .scaler import NativeScaler
from .timm import optimizer_kwargs, scheduler_kwargs
from .transform import create_transform, hard_target_transform

__all__ = [
    "AverageMeter",
    "CheckpointSaver",
    "DefaultContext",
    "ModuleData",
    "NativeScaler",
    "PrefetchLoader",
    "accuracy",
    "area_under_lift_curve",
    "area_under_risk_coverage_curve",
    "auroc",
    "binary_brier",
    "binary_log_probability",
    "calculate_bin_metrics",
    "calculate_output_padding",
    "calculate_same_padding",
    "calibration_error",
    "centered_cov",
    "coverage_for_accuracy",
    "create_dataset",
    "create_loader",
    "create_loss_fn",
    "create_model",
    "create_transform",
    "cross_entropy",
    "deep_setattr",
    "dempster_shafer_metric",
    "entropy",
    "excess_area_under_risk_coverage_curve",
    "float_tuple",
    "get_ranks",
    "hard_target_transform",
    "int_tuple",
    "is_correct_pred",
    "kl_divergence",
    "kwargs",
    "kwargs",
    "log_wandb",
    "multiclass_brier",
    "multiclass_log_probability",
    "optimizer_kwargs",
    "parse_args",
    "pearsonr",
    "register",
    "register_cond",
    "relative_area_under_lift_curve",
    "replace",
    "replace_cond",
    "resolve_data_config",
    "scheduler_kwargs",
    "set_random_seed",
    "setup_logging",
    "spearmanr",
    "string_tuple",
    "wrap_model",
]
