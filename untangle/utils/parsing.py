"""Argparse utilities."""

import argparse
import ast
import logging
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from torch import nn

logger = logging.getLogger(__name__)


def float_tuple(string: str) -> tuple[float, ...]:
    """Converts a comma-separated string of floats to a tuple of floats."""
    if not string:
        return ()

    return tuple(map(float, string.split(",")))


def int_tuple(string: str) -> tuple[int, ...]:
    """Converts a comma-separated string of integers to a tuple of integers."""
    if not string:
        return ()

    return tuple(map(int, string.split(",")))


def string_tuple(string: str) -> tuple[str, ...]:
    """Splits a comma-separated string into a tuple of strings."""
    if not string:
        return ()

    return tuple(string.split(","))


def path_tuple(string: str) -> tuple[Path, ...]:
    """Converts a comma-separated string of paths to a tuple of Path objects."""
    if not string:
        return ()

    return tuple(map(Path, string.split(",")))


def module(string: str) -> nn.Module:
    """Retrieves a PyTorch nn module class based on the input string."""
    if not string.startswith("nn."):
        msg = f"Invalid module name: {string}. Must start with 'nn.'"
        raise ValueError(msg)

    class_name = string[3:]  # Remove the "nn." prefix
    if not hasattr(nn, class_name):
        msg = f"No such module: {string}"
        raise ValueError(msg)

    return getattr(nn, class_name)


def kwargs(string: str) -> dict[str, Any]:
    """Parses a string of key-value pairs into a dictionary."""

    def parse_value(value: str, parsers: list[Callable[[str], Any]]) -> Any:
        for parser in parsers:
            try:
                return parser(value)
            except ValueError:
                continue
        return value

    parsers = [ast.literal_eval, module]
    parse = partial(parse_value, parsers=parsers)

    return {
        key: parse(value)
        for key, value in (pair.split("=", 1) for pair in string.split())
    }


parser = argparse.ArgumentParser(description="PyTorch training with uncertainty")

# Dataset parameters
group = parser.add_argument_group("Dataset parameters")
group.add_argument(
    "--data-dir", type=Path, default=Path(), help="Path to training dataset root dir"
)
group.add_argument(
    "--dataset",
    type=str,
    default="hard/imagenet",
    help='Dataset type + name ("<type>/<name>")',
)
group.add_argument(
    "--soft-imagenet-label-dir",
    type=Path,
    default=Path(),
    help="Path to raters.npz and real.json soft ImageNet labels",
)
group.add_argument(
    "--data-dir-id", type=Path, default=None, help="Path to ID eval dataset root dir"
)
group.add_argument(
    "--dataset-id",
    type=str,
    default="soft/imagenet",
    help=(
        'ID eval + test dataset type + name ("<type>/<name>"), usually the same or a '
        "soft label variant of --dataset"
    ),
)
group.add_argument(
    "--ood-transforms-eval",
    type=string_tuple,
    default=(
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "frosted_glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic",
        "pixelate",
        "jpeg",
    ),
    help="List of dataset transforms to be used on the ood eval dataset",
)
group.add_argument(
    "--ood-transforms-test",
    type=string_tuple,
    default=(),
    help="List of dataset transforms to be used on the ood test dataset",
)
group.add_argument(
    "--train-subset",
    type=float,
    default=1.0,
    help="Fraction of training set to use during training",
)
group.add_argument(
    "--max-num-id-ood-test-samples",
    type=int,
    default=100000,
    help="Maximum number of samples in concatenated ID + OOD test dataset",
)
group.add_argument(
    "--max-num-id-ood-train-samples",
    type=int,
    default=100000,
    help=(
        "Maximum number of samples in concatenated ID + OOD train dataset in the "
        "Mahalanobis method"
    ),
)
group.add_argument(
    "--max-num-id-train-samples",
    type=int,
    default=100000,
    help="Maximum number of samples in (truncated) train dataset for the DDU method",
)
group.add_argument(
    "--max-num-covariance-samples",
    type=int,
    default=100000,
    help=(
        "Number of samples to calculate the covariance matrix in the Mahalanobis method"
    ),
)
group.add_argument(
    "--train-split",
    type=str,
    default="train",
    help="Dataset train split (train/validation/test)",
)
group.add_argument(
    "--val-split",
    type=str,
    default="val",
    help="Dataset validation split (train/validation/test)",
)
group.add_argument(
    "--test-split",
    type=str,
    default="test",
    help="Dataset test split ID/OOD (train/validation/test)",
)
group.add_argument(
    "--evaluate-on-test-sets",
    action="store_true",
    help="Evaluate model on the provided test sets",
)
group.add_argument(
    "--discard-ood-test-sets",
    action="store_true",
    help="Do not evaluate model on the provided OOD test sets",
)
group.add_argument(
    "--discard-uniform-ood-test-sets",
    action="store_true",
    help="Do not evaluate model on the provided uniform OOD test sets",
)
group.add_argument(
    "--storage-device",
    type=str,
    default="cuda",
    help="Storage device during evaluation",
)
group.add_argument(
    "--severities",
    type=int_tuple,
    default=(1, 2, 3, 4, 5),
    help="OOD severities to evaluate",
)
group.add_argument(
    "--dataset-download",
    action="store_true",
    help="Allow downloading torch datasets",
)

# Uncertainty method parameters
group = parser.add_argument_group("Method parameters")
group.add_argument(
    "--method-name",
    type=str,
    default="ce-baseline",
    help="Name of uncertainty method",
)
group.add_argument(
    "--num-hidden-features",
    type=int,
    default=256,
    help="Number of hidden features in the uncertainty method",
)
group.add_argument(
    "--mlp-depth",
    type=int,
    default=3,
    help="Number of layers in the MLP used by the uncertainty method",
)
group.add_argument(
    "--stopgrad",
    action="store_true",
    help=(
        "Whether to stop gradient flow to the model backbone in direct prediction "
        "methods"
    ),
)
group.add_argument(
    "--num-hooks",
    type=int,
    default=None,
    help="Number of hooks in deep direct prediction methods and Mahalanobis",
)
group.add_argument(
    "--module-type",
    type=module,
    default=None,
    help=(
        "Module type to attach hooks to in deep direct prediction methods and "
        "Mahalanobis"
    ),
)
group.add_argument(
    "--module-name-regex",
    default="^(act1|layer[1-4])$",
    type=str,
    help=(
        "Module names to attach hooks to in deep direct prediction methods and "
        "Mahalanobis"
    ),
)
group.add_argument(
    "--dropout-probability",
    type=float,
    default=0.05,
    help="Dropout probability in the MC-Dropout method",
)
group.add_argument(
    "--use-filterwise-dropout",
    action="store_true",
    help="Whether to use filterwise dropout in the MC-Dropout method",
)
group.add_argument(
    "--num-mc-samples",
    type=int,
    default=10,
    help="Number of Monte Carlo samples in the uncertainty method",
)
group.add_argument(
    "--num-mc-samples-integral",
    default=1000,
    type=int,
    help=(
        "Number of Monte Carlo samples to integrate out the logits with the diagonal "
        "Gaussian in the heteroscedastic classification NN method"
    ),
)
group.add_argument(
    "--num-mc-samples-cv",
    default=50,
    type=int,
    help=(
        "Number of Monte Carlo samples in the prior precision CV of the Laplace method"
    ),
)
group.add_argument(
    "--rbf-length-scale",
    type=float,
    default=0.1,
    help="Length scale of the RBF kernel in the DUQ method",
)
group.add_argument(
    "--ema-momentum",
    type=float,
    default=0.999,
    help="Momentum factor for the exponential moving average in the DUQ method",
)
group.add_argument(
    "--lambda-gradient-penalty",
    type=float,
    default=0.75,
    help="Gradient penalty lambda in the DUQ method",
)
group.add_argument(
    "--matrix-rank",
    default=15,
    type=int,
    help="Rank of low-rank covariance matrix part in the HET-XL method",
)
group.add_argument(
    "--use-het",
    action="store_true",
    help="Whether to use HET instead of HET-XL",
)
group.add_argument(
    "--temperature",
    type=float,
    default=1.5,
    help="Temperature in the HET-XL method",
)
group.add_argument(
    "--pred-type",
    type=str,
    default="glm",
    help="Prediction type used by Laplace",
)
group.add_argument(
    "--hessian-structure",
    type=str,
    default="kron",
    help="Hessian structure method used by Laplace",
)
group.add_argument(
    "--use-low-rank-cov",
    action="store_true",
    help="Whether to use the low rank covariance matrix factor in the SWAG method",
)
group.add_argument(
    "--max-rank",
    type=int,
    default=20,
    help="Maximum rank of the low rank covariance matrix factor in the SWAG method",
)
group.add_argument(
    "--num-checkpoints-per-epoch",
    type=int,
    default=4,
    help="Number of checkpoints per epoch in the SWAG method",
)
group.add_argument(
    "--magnitude",
    type=float,
    default=0.001,
    help="Gradient magnitude in the Mahalanobis method",
)
group.add_argument(
    "--num-heads",
    type=int,
    default=10,
    help="Number of output heads in the Shallow Ensemble method",
)
group.add_argument(
    "--likelihood",
    type=str,
    default="gaussian",
    help="Likelihood in the SNGP method",
)
group.add_argument(
    "--use-spectral-normalization",
    action="store_true",
    help="Whether to use spectral normalization in the SNGP and DDU methods",
)
group.add_argument(
    "--spectral-normalization-iteration",
    type=int,
    default=1,
    help=(
        "Number of iterations in the spectral normalization step of the "
        "SNGP and DDU methods"
    ),
)
group.add_argument(
    "--spectral-normalization-bound",
    type=float,
    default=6,
    help="Bound of the spectral norm in the SNGP and DDU methods",
)
group.add_argument(
    "--use-spectral-normalized-batch-norm",
    action="store_true",
    help=(
        "Whether to use spectral normalization for batch norm in the "
        "SNGP and DDU methods"
    ),
)
group.add_argument(
    "--use-tight-norm-for-pointwise-convs",
    action="store_true",
    help=(
        "Whether to use fully connected spectral normalization for pointwise convs "
        "in the SNGP and DDU methods"
    ),
)
group.add_argument(
    "--num-random-features",
    type=int,
    default=1024,
    help="Number of random features in the SNGP method",
)
group.add_argument(
    "--gp-kernel-scale",
    type=float,
    default=1.0,
    help="Kernel scale in the SNGP method",
)
group.add_argument(
    "--gp-output-bias",
    type=float,
    default=0.0,
    help="Output bias in the SNGP method",
)
group.add_argument(
    "--gp-random-feature-type",
    type=str,
    default="orf",
    help="Type of random feature in the SNGP method",
)
group.add_argument(
    "--use-input-normalized-gp",
    action="store_true",
    help="Whether to normalize the GP's input in the SNGP method",
)
group.add_argument(
    "--gp-cov-momentum",
    type=float,
    default=-1,
    help=(
        "Momentum term in the SNGP method. If -1, use exact covariance matrix from the "
        "last epoch"
    ),
)
group.add_argument(
    "--gp-cov-ridge-penalty",
    type=float,
    default=1.0,
    help="Ridge penalty for the GP precision matrix before inverting it",
)
group.add_argument(
    "--gp-input-dim",
    type=int,
    default=128,
    help="Input dimension to the GP (if > 0, use random projection)",
)
group.add_argument(
    "--latent-dim",
    type=int,
    default=6,
    help="Latent dimensionality in PostNet",
)
group.add_argument(
    "--num-density-components",
    type=int,
    default=6,
    help="Number of density components in PostNet's normalizing flow",
)
group.add_argument(
    "--use-batched-flow",
    action="store_true",
    help="Whether the normalizing flow in PostNet is batched",
)
group.add_argument(
    "--reset-classifier",
    action="store_true",
    help="Whether to reset the classifier layer before training",
)
group.add_argument(
    "--use-temperature-scaling",
    action="store_true",
    help="Whether to use temperature scaling for the Temperature and DDU methods",
)
group.add_argument(
    "--scale",
    default=(0.08, 1.0),
    type=float_tuple,
    help="Random resize scale for ImageNet",
)
group.add_argument(
    "--ratio",
    default=(3 / 4, 4 / 3),
    type=float_tuple,
    help="Random resize aspect ratio for ImageNet",
)
group.add_argument(
    "--hflip",
    type=float,
    default=0.5,
    help="Horizontal flip training aug probability",
)
group.add_argument(
    "--color-jitter",
    type=float,
    default=0.0,
    help="Color jitter factor for ImageNet",
)
group.add_argument(
    "--crop-pct",
    type=float,
    default=0.875,
    help="Input image center crop percent for ImageNet eval",
)
group.add_argument(
    "--padding",
    type=int,
    default=2,
    help="Padding for CIFAR",
)

# Loss parameters
group = parser.add_argument_group("Loss parameters")
group.add_argument(
    "--loss",
    type=str,
    default="cross-entropy",
    help="Loss for training",
)
group.add_argument(
    "--lambda-uncertainty-loss",
    type=float,
    default=0.01,
    help=(
        "Multiplier of the uncertainty loss when calculating the total loss for the "
        "correctness and loss prediction methods"
    ),
)
group.add_argument(
    "--use-top5-correctness",
    action="store_true",
    help="Whether to use top-5 correctness to train the correctness prediction method",
)
group.add_argument(
    "--detach-uncertainty-target",
    action="store_true",
    help=(
        "Whether to detach the task loss before calculating the uncertainty loss in "
        "the loss prediction method"
    ),
)

# Model parameters
group = parser.add_argument_group("Model parameters")
group.add_argument(
    "--model-name",
    type=str,
    default="timm/resnet_50",
    help="Name of model to train",
)
group.add_argument(
    "--weight-paths",
    type=path_tuple,
    default=(),
    help="List of weight paths for post-hoc methods",
)
group.add_argument(
    "--pretrained",
    action="store_true",
    help="Start with pretrained version of specified network (if available)",
)
group.add_argument(
    "--initial-checkpoint-path",
    type=Path,
    default=None,
    help="Initialize model from this checkpoint",
)
group.add_argument(
    "--num-classes",
    type=int,
    default=None,
    help="Number of label classes (model default if None)",
)
group.add_argument(
    "--img-size",
    type=int,
    default=224,
    help="Image size",
)
group.add_argument(
    "--mean",
    type=float_tuple,
    default=(0.485, 0.456, 0.406),
    help="Mean pixel value of dataset",
)
group.add_argument(
    "--std",
    type=float_tuple,
    default=(0.229, 0.224, 0.225),
    help="Std of dataset",
)
group.add_argument(
    "--batch-size",
    type=int,
    default=128,
    help="Input batch size for training",
)
group.add_argument(
    "--accumulation-steps",
    type=int,
    default=16,
    help=(
        "Number of batches to accumulate before making an optimizer step "
        "(to simulate a larger batch size)"
    ),
)
group.add_argument(
    "--validation-batch-size",
    type=int,
    default=None,
    help="Validation batch size override",
)
group.add_argument("--model-kwargs", default={}, type=kwargs)

# Scripting
group = parser.add_argument_group("Scripting")
group.add_argument(
    "--compile",
    action="store_true",
    help="Whether to compile the model backbone",
)

# Optimizer parameters
group = parser.add_argument_group("Optimizer parameters")
group.add_argument(
    "--opt",
    type=str,
    default="adamw",
    help="Optimizer",
)
group.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="Optimizer momentum",
)
group.add_argument("--weight-decay", type=float, default=2e-5, help="Weight decay")
group.add_argument(
    "--opt-kwargs",
    type=kwargs,
    default={},
    help="Optimizer keyword arguments",
)

# Learning rate schedule parameters
group = parser.add_argument_group("Learning rate schedule parameters")
group.add_argument(
    "--sched-kwargs",
    type=kwargs,
    default={},
    help="LR scheduler keyword arguments",
)
group.add_argument(
    "--lr",
    type=float,
    default=None,
    help="Learning rate, overrides lr-base if set",
)
group.add_argument(
    "--lr-base",
    type=float,
    default=0.001,
    help="Base learning rate: lr = lr_base * global_batch_size / base_size",
)
group.add_argument(
    "--epochs",
    type=int,
    default=90,
    help="Number of epochs to train",
)

# Augmentation & regularization parameters
group = parser.add_argument_group("Augmentation and regularization parameters")
group.add_argument(
    "--uce-regularization-factor",
    type=float,
    default=1e-5,
    help="UCE regularization factor for PostNets",
)
group.add_argument(
    "--edl-start-epoch",
    type=int,
    default=0,
    help="Start epoch for the EDL flatness regularizer",
)
group.add_argument(
    "--edl-scaler",
    type=float,
    default=1.0,
    help="Scaler for the EDL flatness regularizer",
)
group.add_argument(
    "--edl-activation",
    type=str,
    default="exp",
    help="EDL final activation function",
)

# Misc
group = parser.add_argument_group("Miscellaneous parameters")
group.add_argument("--seed", type=int, default=42, help="Random seed")
group.add_argument(
    "--log-interval",
    type=int,
    default=50,
    help="How many batches to wait before logging training status",
)
group.add_argument(
    "--checkpoint-history",
    type=int,
    default=3,
    help="Number of checkpoints to keep",
)
group.add_argument(
    "--num-workers",
    type=int,
    default=12,
    help="How many training worker processes to use",
)
group.add_argument(
    "--num-eval-workers",
    type=int,
    default=12,
    help="How many eval worker processes to use",
)
group.add_argument(
    "--amp", action="store_true", help="Use native AMP for mixed precision training"
)
group.add_argument(
    "--amp-dtype",
    type=str,
    default="float16",
    help="Lower precision AMP dtype",
)
group.add_argument(
    "--pin-memory",
    action="store_true",
    help="Pin CPU memory in DataLoader for (sometimes) more efficient transfer to GPU",
)
group.add_argument(
    "--prefetcher",
    action="store_true",
    help="Use fast prefetcher",
)
group.add_argument(
    "--channels-last",
    action="store_true",
    help="Use channels_last memory layout",
)
group.add_argument(
    "--eval-metric",
    type=str,
    default="id_eval_one_minus_max_probs_of_bma_auroc_hard_bma_correctness_original",
    help="Metric to track for early stopping/checkpoint saving",
)
group.add_argument(
    "--best-save-start-epoch",
    type=int,
    default=0,
    help="Epoch index from which best model according to eval metric is saved",
)
group.add_argument(
    "--log-wandb",
    action="store_true",
    default=False,
    help="Log training and validation metrics to wandb",
)


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments and processes special cases."""
    args = parser.parse_args()

    if args.data_dir_id is None:
        args.data_dir_id = args.data_dir

    # Detect a special code that tells us to use the local node storage.
    SLURM_TUE_PATH = Path(
        f"/host/scratch_local/{os.environ.get('SLURM_JOB_USER')}-"
        f"{os.environ.get('SLURM_JOBID')}/datasets"
    )

    if str(args.data_dir) == "SLURM_TUE":
        args.data_dir = SLURM_TUE_PATH

    if str(args.data_dir_id) == "SLURM_TUE":
        args.data_dir_id = SLURM_TUE_PATH

    if str(args.soft_imagenet_label_dir) == "SLURM_TUE":
        args.soft_imagenet_label_dir = SLURM_TUE_PATH

    if args.discard_ood_test_sets:
        args.severities = ()
        args.ood_transforms_eval = ()
        args.ood_transforms_test = ()
        args.discard_uniform_ood_test_sets = True

    return args


def resolve_data_config(args: dict[str, Any]) -> dict[str, Any]:
    """Resolves data configuration based on input arguments."""
    data_config = {}

    # Resolve input/image size
    in_chans = 3
    img_size = args["img_size"]
    input_size = (in_chans, img_size, img_size)
    data_config["input_size"] = input_size

    # Resolve interpolation method
    data_config["interpolation"] = "bicubic"

    # Resolve dataset mean for normalization
    data_config["mean"] = args["mean"]

    # Resolve dataset std for normalization
    data_config["std"] = args["std"]

    # Resolve default inference crop
    data_config["crop_pct"] = args["crop_pct"]

    # Resolve default crop percentage
    data_config["crop_mode"] = "center"

    # Resolve padding
    data_config["padding"] = args["padding"]

    msg = "Data processing configuration for current model:"

    for n, v in data_config.items():
        if "imagenet" in args["dataset"] and n == "padding":
            continue

        if "cifar" in args["dataset"] and n in {"crop_pct", "crop_mode"}:
            continue

        msg += f"\n\t{n}: {v}"

    logger.info(msg)

    return data_config
