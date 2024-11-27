"""Copyright 2020 Ross Wightman and 2024 Bálint Mucsányi."""

import argparse
import logging
import os
import time
from collections.abc import Callable
from numbers import Number
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn.parallel
from torch import Tensor, nn
from torch.utils.data import DataLoader

from untangle.utils import (
    AverageMeter,
    PrefetchLoader,
    area_under_lift_curve,
    area_under_risk_coverage_curve,
    auroc,
    binary_brier,
    binary_log_probability,
    calibration_error,
    coverage_for_accuracy,
    dempster_shafer_metric,
    entropy,
    excess_area_under_risk_coverage_curve,
    kl_divergence,
    multiclass_brier,
    multiclass_log_probability,
    pearsonr,
    relative_area_under_lift_curve,
    spearmanr,
)
from untangle.wrappers import (
    BaseCorrectnessPredictionWrapper,
    BaseLossPredictionWrapper,
    DDUWrapper,
    DeepEnsembleWrapper,
    DirichletWrapper,
    DUQWrapper,
    FastDeepEnsembleWrapper,
    HetClassNNWrapper,
    MahalanobisWrapper,
    SWAGWrapper,
)

logger = logging.getLogger(__name__)


def evaluate_on_ood_uniform_test_loaders(
    model: nn.Module,
    loaders: dict[str, dict[str, DataLoader | PrefetchLoader]],
    device: torch.device,
    storage_device: torch.device,
    amp_autocast: Callable,
    key_prefix: str,
    output_dir: Path,
    is_soft_dataset: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluates the model on uniform out-of-distribution test loaders.

    Args:
        model: The model to evaluate.
        loaders: Dictionary of OOD test loaders.
        device: The device to use for evaluation.
        storage_device: The device to use for storing results.
        amp_autocast: Function for automatic mixed precision.
        key_prefix: Prefix for metric keys.
        output_dir: Directory to save output.
        is_soft_dataset: Whether the dataset uses soft labels.
        args: Additional arguments.

    Returns:
        A dictionary containing flattened metrics.
    """
    metrics = {}

    for name, loader_subset in loaders.items():
        metrics[name] = {}
        per_ood_transform_type_metrics = {}
        for ood_transform_type, loader in loader_subset.items():
            logger.info(f"Evaluating {name} - {ood_transform_type}...")
            time_eval_start = time.perf_counter()

            per_ood_transform_type_metrics[ood_transform_type] = evaluate(
                model=model,
                loader=loader,
                loader_name=f"{name}_{ood_transform_type}",
                device=device,
                storage_device=storage_device,
                amp_autocast=amp_autocast,
                key_prefix="",
                output_dir=output_dir,
                is_upstream_dataset=False,
                is_test_dataset=True,
                is_soft_dataset=is_soft_dataset,
                args=args,
            )

            time_eval_end = time.perf_counter()
            time_eval = time_eval_end - time_eval_start

            logger.info(
                f"Finished evaluating {name} - {ood_transform_type}. "
                f"Took {time_eval:.2f} seconds."
            )
        metrics[name]["avg"] = get_average_metric_values(per_ood_transform_type_metrics)
        metrics[name] |= get_per_transform_ood_detection_results(
            per_ood_transform_type_metrics
        )

    # Summarize results
    flattened_metrics = flatten_ood_uniform_metrics(
        results=metrics, key_prefix=key_prefix
    )

    return flattened_metrics


def evaluate_on_ood_varied_test_loaders(
    model: nn.Module,
    loaders: dict[str, DataLoader | PrefetchLoader],
    device: torch.device,
    storage_device: torch.device,
    amp_autocast: Callable,
    key_prefix: str,
    output_dir: Path,
    is_soft_dataset: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluates the model on varied out-of-distribution test loaders.

    Args:
        model: The model to evaluate.
        loaders: Dictionary of OOD test loaders.
        device: The device to use for evaluation.
        storage_device: The device to use for storing results.
        amp_autocast: Function for automatic mixed precision.
        key_prefix: Prefix for metric keys.
        output_dir: Directory to save output.
        is_soft_dataset: Whether the dataset uses soft labels.
        args: Additional arguments.

    Returns:
        A dictionary containing flattened metrics.
    """
    metrics = {}

    for name, loader in loaders.items():
        metrics[name] = {}
        logger.info(f"Evaluating {name} - varied...")
        time_eval_start = time.perf_counter()

        metrics[name] = evaluate(
            model=model,
            loader=loader,
            loader_name=f"{name}_varied",
            device=device,
            storage_device=storage_device,
            amp_autocast=amp_autocast,
            key_prefix="",
            output_dir=output_dir,
            is_upstream_dataset=False,
            is_test_dataset=True,
            is_soft_dataset=is_soft_dataset,
            args=args,
        )

        time_eval_end = time.perf_counter()
        time_eval = time_eval_end - time_eval_start

        logger.info(
            f"Finished evaluating {name} - varied. " f"Took {time_eval:.2f} seconds."
        )

    # Summarize results
    flattened_metrics = flatten_ood_varied_metrics(
        results=metrics, key_prefix=key_prefix
    )

    # Remove tmp file
    upstream_dict_path = Path(f"data/upstream_dict_{os.environ.get('SLURM_JOBID')}.pt")
    upstream_dict_path.unlink()

    return flattened_metrics


def get_average_metric_values(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Calculates average values from the metrics dictionary.

    Args:
        metrics: Dictionary of metrics to calculate average values from.

    Returns:
        Dictionary containing the per-metric average values.
    """
    # Summarize results
    avg_metrics = {}
    first_loader_metrics = metrics[next(iter(metrics.keys()))]

    for key in first_loader_metrics:
        if isinstance(first_loader_metrics[key], Number):
            metric_vector = torch.tensor([
                loader_result[key] for _, loader_result in metrics.items()
            ])
            avg_metrics[key] = metric_vector.mean().item()

    return avg_metrics


def get_per_transform_ood_detection_results(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Filters per-transform OOD detection results from the metrics dictionary.

    Args:
        metrics: Dictionary of metrics to filter.

    Returns:
        Dictionary containing the pfiltered OOD detection results.
    """
    filtered_metrics = {}

    for ood_transform_type, ood_transform_type_metrics in metrics.items():
        filtered_metrics[ood_transform_type] = {}

        for metric_name, metric_value in ood_transform_type_metrics.items():
            if metric_name.endswith("auroc_oodness"):
                filtered_metrics[ood_transform_type][metric_name] = metric_value

    return filtered_metrics


def flatten_ood_uniform_metrics(
    results: dict[str, dict[str, dict[str, float]]], key_prefix: str
) -> dict[str, float]:
    """Flattens metrics for uniform OOD evaluation.

    Args:
        results: Dictionary of results to flatten.
        key_prefix: Prefix for flattened keys.

    Returns:
        A dictionary of flattened metrics.
    """
    # Flatten output
    flattened_results = {}
    for name, results_subset in results.items():
        for ood_transform_type, results_subsubset in results_subset.items():
            for key, value in results_subsubset.items():
                flattened_results[f"{key_prefix}_{ood_transform_type}_{name}_{key}"] = (
                    value
                )

    return flattened_results


def flatten_ood_varied_metrics(
    results: dict[str, dict[str, float]], key_prefix: str
) -> dict[str, float]:
    """Flattens metrics for varied OOD evaluation.

    Args:
        results: Dictionary of results to flatten.
        key_prefix: Prefix for flattened keys.

    Returns:
        A dictionary of flattened metrics.
    """
    # Flatten output
    flattened_results = {}
    for name, results_subset in results.items():
        for key, value in results_subset.items():
            flattened_results[f"{key_prefix}_varied_{name}_{key}"] = value

    return flattened_results


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader | PrefetchLoader,
    loader_name: str,
    device: torch.device,
    storage_device: torch.device,
    amp_autocast: Callable,
    key_prefix: str,
    output_dir: Path,
    is_upstream_dataset: bool,
    is_test_dataset: bool,
    is_soft_dataset: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluates the model on a given loader.

    Args:
        model: The model to evaluate.
        loader: The data loader.
        loader_name: Name of the loader.
        device: The device to use for evaluation.
        storage_device: The device to use for storing results.
        amp_autocast: Function for automatic mixed precision.
        key_prefix: Prefix for metric keys.
        output_dir: Directory to save output.
        is_upstream_dataset: Whether it's an upstream dataset.
        is_test_dataset: Whether it's a test dataset.
        is_soft_dataset: Whether the dataset uses soft labels.
        args: Additional arguments.

    Returns:
        A dictionary of evaluation metrics.
    """
    model.eval()

    estimates, log_probs, targets, times = get_bundle(
        model=model,
        loader=loader,
        device=device,
        storage_device=storage_device,
        amp_autocast=amp_autocast,
        is_soft_dataset=is_soft_dataset,
        args=args,
    )

    metrics = times

    if is_test_dataset:
        ood_prefix = "id" if is_upstream_dataset else "ood"
        save_prefix = f"{ood_prefix}_test_{loader_name}_"

        metrics = evaluate_on_tasks(
            model=model,
            estimates=estimates,
            log_probs=log_probs,
            targets=targets,
            metrics=metrics,
            is_soft_dataset=is_soft_dataset,
            save_prefix=save_prefix,
            output_dir=output_dir,
            args=args,
        )
    else:
        metrics = evaluate_on_validation_metrics(
            estimates=estimates,
            targets=targets,
            metrics=metrics,
            args=args,
        )

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    if is_upstream_dataset and is_test_dataset:
        # Save ingredients to disk
        max_num_indices = len(targets["gt_hard_labels"])
        num_indices = min(max_num_indices, args.max_num_id_ood_test_samples // 2)
        path_indices = data_dir / f"{num_indices}_indices_out_of_{max_num_indices}.pt"

        if path_indices.exists():
            indices = torch.load(path_indices, weights_only=True)
        else:
            indices = torch.randperm(max_num_indices, device=storage_device)[
                :num_indices
            ]
            torch.save(indices, path_indices)

        upstream_dict = {
            "upstream_estimates": filter_entries(estimates, indices),
            "upstream_targets": filter_entries(targets, indices),
            "is_soft_upstream_dataset": is_soft_dataset,
        }

        upstream_dict["upstream_log_probs"] = filter_entries(log_probs, indices)

        torch.save(
            upstream_dict,
            data_dir / f"upstream_dict_{os.environ.get('SLURM_JOBID')}.pt",
        )
    elif is_test_dataset:
        # Load ingredients from disk
        upstream_dict = torch.load(
            data_dir / f"upstream_dict_{os.environ.get('SLURM_JOBID')}.pt",
            weights_only=True,
        )
        upstream_estimates = upstream_dict["upstream_estimates"]
        upstream_log_probs = upstream_dict["upstream_log_probs"]
        upstream_targets = upstream_dict["upstream_targets"]
        is_soft_upstream_dataset = upstream_dict["is_soft_upstream_dataset"]

        # Make both upstream and downstream tensors the same size to get a 50/50 split
        num_upstream_indices = len(upstream_targets["gt_hard_labels"])
        max_num_downstream_indices = len(targets["gt_hard_labels"])
        num_indices_to_keep = min(num_upstream_indices, max_num_downstream_indices)

        # For upstream, we can just use [:num_samples_keep] in the following, because
        # it's already shuffled. For downstream, let's use random indices
        path_downstream_indices = (
            data_dir
            / f"{num_indices_to_keep}_indices_out_of_{max_num_downstream_indices}.pt"
        )

        if path_downstream_indices.exists():
            downstream_indices = torch.load(path_downstream_indices, weights_only=True)
        else:
            downstream_indices = torch.randperm(
                max_num_downstream_indices, device=storage_device
            )[:num_indices_to_keep]
            torch.save(downstream_indices, path_downstream_indices)

        upstream_estimates = truncate_entries(upstream_estimates, num_indices_to_keep)
        upstream_targets = truncate_entries(upstream_targets, num_indices_to_keep)

        upstream_log_probs = truncate_entries(upstream_log_probs, num_indices_to_keep)
        downstream_log_probs = filter_entries(log_probs, downstream_indices)

        downstream_estimates = filter_entries(estimates, downstream_indices)
        downstream_targets = filter_entries(targets, downstream_indices)

        # Mix ingredients (remember, we're cooking!)
        mixed_estimates = concatenate_values(upstream_estimates, downstream_estimates)

        mixed_log_probs = concatenate_values(upstream_log_probs, downstream_log_probs)

        mixed_targets = concatenate_values(
            upstream_targets, downstream_targets, keys_to_exclude=["gt_soft_labels"]
        )

        # Update joint targets
        mixed_targets["gt_oodness"] = torch.cat([
            torch.zeros((num_indices_to_keep,), device=storage_device),
            torch.ones((num_indices_to_keep,), device=storage_device),
        ]).int()

        if is_soft_upstream_dataset and not is_soft_dataset:
            num_classes = upstream_targets["gt_soft_labels"].shape[1]
            mixed_targets["gt_soft_labels"] = torch.cat([
                upstream_targets["gt_soft_labels"],
                F.one_hot(
                    downstream_targets["gt_hard_labels"],
                    num_classes=num_classes,
                ),
            ])

            mixed_targets["gt_soft_dual_bma_correctnesses"] = torch.cat([
                upstream_targets["gt_soft_dual_bma_correctnesses"],
                downstream_targets["gt_hard_dual_bma_correctnesses"],
            ])
            mixed_targets["gt_soft_dual_bma_correctnesses_top5"] = torch.cat([
                upstream_targets["gt_soft_dual_bma_correctnesses_top5"],
                downstream_targets["gt_hard_dual_bma_correctnesses_top5"],
            ])

            mixed_targets["gt_soft_bma_correctnesses"] = torch.cat([
                upstream_targets["gt_soft_bma_correctnesses"],
                downstream_targets["gt_hard_bma_correctnesses"],
            ])
            mixed_targets["gt_soft_bma_correctnesses_top5"] = torch.cat([
                upstream_targets["gt_soft_bma_correctnesses_top5"],
                downstream_targets["gt_hard_bma_correctnesses_top5"],
            ])
        elif not is_soft_upstream_dataset and is_soft_dataset:
            num_classes = downstream_targets["gt_soft_labels"].shape[1]
            mixed_targets["gt_soft_labels"] = torch.cat([
                F.one_hot(
                    upstream_targets["gt_hard_labels"],
                    num_classes=num_classes,
                ),
                downstream_targets["gt_soft_labels"],
            ])

            mixed_targets["gt_soft_dual_bma_correctnesses"] = torch.cat([
                upstream_targets["gt_hard_dual_bma_correctnesses"],
                downstream_targets["gt_soft_dual_bma_correctnesses"],
            ])
            mixed_targets["gt_soft_dual_bma_correctnesses_top5"] = torch.cat([
                upstream_targets["gt_hard_dual_bma_correctnesses_top5"],
                downstream_targets["gt_soft_dual_bma_correctnesses_top5"],
            ])

            mixed_targets["gt_soft_bma_correctnesses"] = torch.cat([
                upstream_targets["gt_hard_bma_correctnesses"],
                downstream_targets["gt_soft_bma_correctnesses"],
            ])
            mixed_targets["gt_soft_bma_correctnesses_top5"] = torch.cat([
                upstream_targets["gt_hard_bma_correctnesses_top5"],
                downstream_targets["gt_soft_bma_correctnesses_top5"],
            ])
        elif is_soft_upstream_dataset and is_soft_dataset:
            mixed_targets["gt_soft_labels"] = torch.cat([
                upstream_targets["gt_soft_labels"],
                downstream_targets["gt_soft_labels"],
            ])

        ood_prefix = "id" if is_upstream_dataset else "ood"
        save_prefix = (
            f"{ood_prefix}_test_{loader_name}_mixed_"
            f"{args.dataset_id.replace('/', '_')}_"
        )

        metrics = evaluate_on_tasks(
            model=model,
            estimates=mixed_estimates,
            log_probs=mixed_log_probs,
            targets=mixed_targets,
            metrics=metrics,
            is_soft_dataset=is_soft_dataset,
            save_prefix=save_prefix,
            output_dir=output_dir,
            args=args,
            is_soft_upstream_dataset=is_soft_upstream_dataset,
        )

    if key_prefix:
        for metric_name in list(metrics.keys()):
            metrics[f"{key_prefix}_{metric_name}"] = metrics.pop(metric_name)

    return metrics


def filter_entries(estimates: dict[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    """Filters entries in the estimates dictionary based on given indices.

    Args:
        estimates: Dictionary of estimates.
        indices: Indices to use for filtering.

    Returns:
        A filtered dictionary of estimates.
    """
    filtered_estimates = estimates.copy()

    for estimator_name, estimate in filtered_estimates.items():
        filtered_estimates[estimator_name] = estimate[indices]

    return filtered_estimates


def truncate_entries(
    estimates: dict[str, Tensor], num_indices_to_keep: int
) -> dict[str, Tensor]:
    """Truncates entries in the estimates dictionary.

    Args:
        estimates: Dictionary of estimates.
        num_indices_to_keep: Number of indices to keep.

    Returns:
        A truncated dictionary of estimates.
    """
    truncated_estimates = estimates.copy()

    for estimator_name, estimate in truncated_estimates.items():
        truncated_estimates[estimator_name] = estimate[:num_indices_to_keep]

    return truncated_estimates


def concatenate_values(
    upstream_dict: dict[str, Tensor],
    downstream_dict: dict[str, Tensor],
    keys_to_exclude: list[str] | None = None,
) -> dict[str, Tensor]:
    """Concatenates values from upstream and downstream dictionaries.

    Args:
        upstream_dict: Dictionary of upstream values.
        downstream_dict: Dictionary of downstream values.
        keys_to_exclude: List of keys to exclude from concatenation.

    Returns:
        A dictionary with concatenated values.
    """
    if keys_to_exclude is None:
        keys_to_exclude = []

    common_keys = upstream_dict.keys() & downstream_dict.keys()
    result = {
        key: torch.cat([upstream_dict[key], downstream_dict[key]], dim=0)
        for key in common_keys
        if key not in keys_to_exclude
    }

    return result


def evaluate_on_validation_metrics(
    estimates: dict[str, Tensor],
    targets: dict[str, Tensor],
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluates the model's estimates on the metrics needed for validation.

    Args:
        estimates: Dictionary of estimates.
        targets: Dictionary of targets.
        metrics: Dictionary of metrics to update.
        args: Additional arguments.

    Returns:
        Updated metrics dictionary.
    """
    metrics["hard_bma_accuracy_original"] = (
        targets["gt_hard_bma_correctnesses_original"].float().mean().item()
    )

    for estimator_name, estimate in estimates.items():
        if estimator_name in args.eval_metric:
            estimate = -estimate

            gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses_original"]
            metrics[f"{estimator_name}_auroc_hard_bma_correctness_original"] = auroc(
                gt_hard_bma_correctnesses, estimate
            ).item()

            break

    return metrics


def evaluate_on_tasks(
    model: nn.Module,
    estimates: dict[str, Tensor],
    log_probs: dict[str, Tensor],
    targets: dict[str, Tensor],
    metrics: dict[str, float],
    is_soft_dataset: bool,
    save_prefix: str,
    output_dir: Path,
    args: argparse.Namespace,
    is_soft_upstream_dataset: bool | None = None,
) -> dict[str, float]:
    """Evaluates the model on various uncertainty quantification tasks.

    Args:
        model: The model to evaluate.
        estimates: Dictionary of estimates.
        log_probs: Dictionary of log probabilities.
        targets: Dictionary of targets.
        metrics: Dictionary of metrics to update.
        is_soft_dataset: Whether the dataset uses soft labels.
        save_prefix: Prefix for saving results.
        output_dir: Directory to save output.
        args: Additional arguments.
        is_soft_upstream_dataset: Whether the upstream dataset uses soft labels.

    Returns:
        Updated metrics dictionary.
    """
    is_mixed_eval = is_soft_upstream_dataset is not None

    if is_mixed_eval:
        metrics |= evaluate_on_ood_detection(
            estimates=estimates,
            targets=targets,
            args=args,
        )

    metrics |= evaluate_on_correctness_prediction(
        estimates=estimates,
        targets=targets,
        is_soft_dataset=is_soft_dataset,
        args=args,
        is_soft_upstream_dataset=is_soft_upstream_dataset,
    )
    metrics |= evaluate_on_abstained_prediction(
        estimates=estimates,
        targets=targets,
        is_soft_dataset=is_soft_dataset,
        args=args,
        is_soft_upstream_dataset=is_soft_upstream_dataset,
    )
    metrics |= evaluate_on_proper_scoring_and_calibration(
        model=model,
        estimates=estimates,
        log_probs=log_probs,
        targets=targets,
        is_soft_dataset=is_soft_dataset,
        args=args,
        is_soft_upstream_dataset=is_soft_upstream_dataset,
    )
    metrics |= evaluate_on_bregman(
        estimates=estimates,
        targets=targets,
        is_soft_dataset=is_soft_dataset,
        args=args,
        is_soft_upstream_dataset=is_soft_upstream_dataset,
    )
    metrics |= evaluate_on_correlation_of_estimators(
        model=model,
        estimates=estimates,
        output_dir=output_dir,
        save_prefix=save_prefix,
        args=args,
        is_soft_upstream_dataset=is_soft_upstream_dataset,
    )
    metrics |= evaluate_on_correlation_of_decompositions(
        estimates=estimates,
        targets=targets,
        is_soft_dataset=is_soft_dataset,
        output_dir=output_dir,
        save_prefix=save_prefix,
        args=args,
        is_soft_upstream_dataset=is_soft_upstream_dataset,
    )

    return metrics


def evaluate_on_correctness_prediction(
    estimates: dict[str, Tensor],
    targets: dict[str, Tensor],
    is_soft_dataset: bool,
    args: argparse.Namespace,
    is_soft_upstream_dataset: bool | None,
) -> dict[str, float]:
    """Evaluates the model on correctness prediction metrics.

    Args:
        estimates: Dictionary of estimates.
        targets: Dictionary of targets.
        is_soft_dataset: Whether the dataset uses soft labels.
        args: Additional arguments.
        is_soft_upstream_dataset: Whether the upstream dataset uses soft labels.

    Returns:
        A dictionary of correctness prediction metrics.
    """
    is_mixed_eval = is_soft_upstream_dataset is not None

    # For correctness prediction, one of the datasets being soft is enough
    if is_mixed_eval:
        is_soft_dataset = is_soft_dataset or is_soft_upstream_dataset

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id.replace('/', '_')}_" if is_mixed_eval else ""

    gt_hard_dual_bma_correctnesses_original = targets[
        "gt_hard_dual_bma_correctnesses_original"
    ]
    gt_hard_bma_correctnesses_original = targets["gt_hard_bma_correctnesses_original"]
    gt_hard_dual_bma_correctnesses = targets["gt_hard_dual_bma_correctnesses"]
    gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses"]

    gt_hard_dual_bma_correctnesses_original_top5 = targets[
        "gt_hard_dual_bma_correctnesses_original_top5"
    ]
    gt_hard_bma_correctnesses_original_top5 = targets[
        "gt_hard_bma_correctnesses_original_top5"
    ]
    gt_hard_dual_bma_correctnesses_top5 = targets["gt_hard_dual_bma_correctnesses_top5"]
    gt_hard_bma_correctnesses_top5 = targets["gt_hard_bma_correctnesses_top5"]

    for estimator_name, estimate in estimates.items():
        # In `estimates`, we have *uncertainty* estimates: higher signals more
        # uncertain. For correctness prediction, we need *certainty* estimates: the
        # AUROC is high if there exists a threshold for which all certain samples are
        # correct (1) and all others are incorrect (0).
        estimate = -estimate

        metrics[
            f"{key_prefix}{estimator_name}_auroc_hard_dual_bma_correctness_original"
        ] = auroc(gt_hard_dual_bma_correctnesses_original, estimate).item()
        metrics[f"{key_prefix}{estimator_name}_auroc_hard_dual_bma_correctness"] = (
            auroc(gt_hard_dual_bma_correctnesses, estimate).item()
        )
        metrics[f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness_original"] = (
            auroc(gt_hard_bma_correctnesses_original, estimate).item()
        )
        metrics[f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness"] = auroc(
            gt_hard_bma_correctnesses, estimate
        ).item()

        metrics[
            f"{key_prefix}{estimator_name}_auroc_hard_dual_bma_correctness_original_top5"
        ] = auroc(gt_hard_dual_bma_correctnesses_original_top5, estimate).item()
        metrics[
            f"{key_prefix}{estimator_name}_auroc_hard_dual_bma_correctness_top5"
        ] = auroc(gt_hard_dual_bma_correctnesses_top5, estimate).item()
        metrics[
            f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness_original_top5"
        ] = auroc(gt_hard_bma_correctnesses_original_top5, estimate).item()
        metrics[f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness_top5"] = (
            auroc(gt_hard_bma_correctnesses_top5, estimate).item()
        )

    # Performance metrics
    metrics[f"{key_prefix}hard_dual_bma_accuracy_original"] = (
        targets["gt_hard_dual_bma_correctnesses_original"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_dual_bma_accuracy"] = (
        targets["gt_hard_dual_bma_correctnesses"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_bma_accuracy_original"] = (
        targets["gt_hard_bma_correctnesses_original"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_bma_accuracy"] = (
        targets["gt_hard_bma_correctnesses"].float().mean().item()
    )

    metrics[f"{key_prefix}hard_dual_bma_accuracy_original_top5"] = (
        targets["gt_hard_dual_bma_correctnesses_original_top5"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_dual_bma_accuracy_top5"] = (
        targets["gt_hard_dual_bma_correctnesses_top5"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_bma_accuracy_original_top5"] = (
        targets["gt_hard_bma_correctnesses_original_top5"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_bma_accuracy_top5"] = (
        targets["gt_hard_bma_correctnesses_top5"].float().mean().item()
    )

    if is_soft_dataset:
        metrics[f"{key_prefix}soft_dual_bma_accuracy"] = (
            targets["gt_soft_dual_bma_correctnesses"].mean().item()
        )
        metrics[f"{key_prefix}soft_bma_accuracy"] = (
            targets["gt_soft_bma_correctnesses"].mean().item()
        )

        metrics[f"{key_prefix}soft_dual_bma_accuracy_top5"] = (
            targets["gt_soft_dual_bma_correctnesses_top5"].mean().item()
        )
        metrics[f"{key_prefix}soft_bma_accuracy_top5"] = (
            targets["gt_soft_bma_correctnesses_top5"].mean().item()
        )

        probs = targets["gt_soft_labels"]
        max_labels = probs.max(dim=1)[0]
        metrics[f"{key_prefix}best_soft_accuracy"] = max_labels.mean().item()

    return metrics


def evaluate_on_abstained_prediction(
    estimates: dict[str, Tensor],
    targets: dict[str, Tensor],
    is_soft_dataset: bool,
    args: argparse.Namespace,
    is_soft_upstream_dataset: bool | None,
) -> dict[str, float]:
    """Evaluates the model on abstained prediction metrics.

    Args:
        estimates: Dictionary of estimates.
        targets: Dictionary of targets.
        is_soft_dataset: Whether the dataset uses soft labels.
        args: Additional arguments.
        is_soft_upstream_dataset: Whether the upstream dataset uses soft labels.

    Returns:
        A dictionary of abstained prediction metrics.
    """
    is_mixed_eval = is_soft_upstream_dataset is not None

    # For correctness of prediction, one of the datasets being soft is enough
    if is_mixed_eval:
        is_soft_dataset = is_soft_dataset or is_soft_upstream_dataset

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id.replace('/', '_')}_" if is_mixed_eval else ""

    gt_hard_dual_bma_correctnesses_original = targets[
        "gt_hard_dual_bma_correctnesses_original"
    ]
    gt_hard_dual_bma_correctnesses = targets["gt_hard_dual_bma_correctnesses"]
    gt_hard_bma_correctnesses_original = targets["gt_hard_bma_correctnesses_original"]
    gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses"]

    gt_hard_dual_bma_correctnesses_original_top5 = targets[
        "gt_hard_dual_bma_correctnesses_original_top5"
    ]
    gt_hard_dual_bma_correctnesses_top5 = targets["gt_hard_dual_bma_correctnesses_top5"]
    gt_hard_bma_correctnesses_original_top5 = targets[
        "gt_hard_bma_correctnesses_original_top5"
    ]
    gt_hard_bma_correctnesses_top5 = targets["gt_hard_bma_correctnesses_top5"]

    if is_soft_dataset:
        gt_soft_dual_bma_correctnesses = targets["gt_soft_dual_bma_correctnesses"]
        gt_soft_bma_correctnesses = targets["gt_soft_bma_correctnesses"]

        gt_soft_dual_bma_correctnesses_top5 = targets[
            "gt_soft_dual_bma_correctnesses_top5"
        ]
        gt_soft_bma_correctnesses_top5 = targets["gt_soft_bma_correctnesses_top5"]

    for estimator_name, estimate in estimates.items():
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aurc_original"] = (
            area_under_risk_coverage_curve(
                estimate, gt_hard_dual_bma_correctnesses_original
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aurc"] = (
            area_under_risk_coverage_curve(
                estimate, gt_hard_dual_bma_correctnesses
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_cumulative_hard_dual_bma_abstinence_auc_original"
        ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aurc_original"]
        metrics[
            f"{key_prefix}{estimator_name}_cumulative_hard_dual_bma_abstinence_auc"
        ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aurc"]
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_eaurc_original"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_dual_bma_correctnesses_original
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_eaurc"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_dual_bma_correctnesses
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aulc_original"] = (
            area_under_lift_curve(
                estimate, gt_hard_dual_bma_correctnesses_original
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aulc"] = (
            area_under_lift_curve(estimate, gt_hard_dual_bma_correctnesses).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_raulc_original"] = (
            relative_area_under_lift_curve(
                estimate, gt_hard_dual_bma_correctnesses_original
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_raulc"] = (
            relative_area_under_lift_curve(
                estimate, gt_hard_dual_bma_correctnesses
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_hard_dual_bma_coverage_for_95_accuracy_original"
        ] = coverage_for_accuracy(
            estimate, gt_hard_dual_bma_correctnesses_original, accuracy=0.95
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_dual_bma_coverage_for_95_accuracy"
        ] = coverage_for_accuracy(
            estimate, gt_hard_dual_bma_correctnesses, accuracy=0.95
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_dual_bma_coverage_for_99_accuracy_original"
        ] = coverage_for_accuracy(
            estimate, gt_hard_dual_bma_correctnesses_original, accuracy=0.99
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_dual_bma_coverage_for_99_accuracy"
        ] = coverage_for_accuracy(
            estimate, gt_hard_dual_bma_correctnesses, accuracy=0.99
        ).item()

        metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc_original"] = (
            area_under_risk_coverage_curve(
                estimate, gt_hard_bma_correctnesses_original
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc"] = (
            area_under_risk_coverage_curve(estimate, gt_hard_bma_correctnesses).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_cumulative_hard_bma_abstinence_auc_original"
        ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc_original"]
        metrics[f"{key_prefix}{estimator_name}_cumulative_hard_bma_abstinence_auc"] = (
            1 - metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc"]
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_eaurc_original"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_bma_correctnesses_original
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_eaurc"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_bma_correctnesses
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_aulc_original"] = (
            area_under_lift_curve(estimate, gt_hard_bma_correctnesses_original).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_aulc"] = area_under_lift_curve(
            estimate, gt_hard_bma_correctnesses
        ).item()
        metrics[f"{key_prefix}{estimator_name}_hard_bma_raulc_original"] = (
            relative_area_under_lift_curve(
                estimate, gt_hard_bma_correctnesses_original
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_raulc"] = (
            relative_area_under_lift_curve(estimate, gt_hard_bma_correctnesses).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_hard_bma_coverage_for_95_accuracy_original"
        ] = coverage_for_accuracy(
            estimate, gt_hard_bma_correctnesses_original, accuracy=0.95
        ).item()
        metrics[f"{key_prefix}{estimator_name}_hard_bma_coverage_for_95_accuracy"] = (
            coverage_for_accuracy(
                estimate, gt_hard_bma_correctnesses, accuracy=0.95
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_hard_bma_coverage_for_99_accuracy_original"
        ] = coverage_for_accuracy(
            estimate, gt_hard_bma_correctnesses_original, accuracy=0.99
        ).item()
        metrics[f"{key_prefix}{estimator_name}_hard_bma_coverage_for_99_accuracy"] = (
            coverage_for_accuracy(
                estimate, gt_hard_bma_correctnesses, accuracy=0.99
            ).item()
        )

        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aurc_original_top5"] = (
            area_under_risk_coverage_curve(
                estimate, gt_hard_dual_bma_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aurc_top5"] = (
            area_under_risk_coverage_curve(
                estimate, gt_hard_dual_bma_correctnesses_top5
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_cumulative_hard_dual_bma_abstinence_auc_original_top5"
        ] = (
            1
            - metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aurc_original_top5"]
        )
        metrics[
            f"{key_prefix}{estimator_name}_cumulative_hard_dual_bma_abstinence_auc_top5"
        ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aurc_top5"]
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_eaurc_original_top5"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_dual_bma_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_eaurc_top5"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_dual_bma_correctnesses_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aulc_original_top5"] = (
            area_under_lift_curve(
                estimate, gt_hard_dual_bma_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_aulc_top5"] = (
            area_under_lift_curve(estimate, gt_hard_dual_bma_correctnesses_top5).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_raulc_original_top5"] = (
            relative_area_under_lift_curve(
                estimate, gt_hard_dual_bma_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_dual_bma_raulc_top5"] = (
            relative_area_under_lift_curve(
                estimate, gt_hard_dual_bma_correctnesses_top5
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_hard_dual_bma_coverage_for_95_accuracy_original_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_dual_bma_correctnesses_original_top5, accuracy=0.95
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_dual_bma_coverage_for_95_accuracy_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_dual_bma_correctnesses_top5, accuracy=0.95
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_dual_bma_coverage_for_99_accuracy_original_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_dual_bma_correctnesses_original_top5, accuracy=0.99
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_dual_bma_coverage_for_99_accuracy_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_dual_bma_correctnesses_top5, accuracy=0.99
        ).item()

        metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc_original_top5"] = (
            area_under_risk_coverage_curve(
                estimate, gt_hard_bma_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc_top5"] = (
            area_under_risk_coverage_curve(
                estimate, gt_hard_bma_correctnesses_top5
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_cumulative_hard_bma_abstinence_auc_original_top5"
        ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc_original_top5"]
        metrics[
            f"{key_prefix}{estimator_name}_cumulative_hard_bma_abstinence_auc_top5"
        ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc_top5"]
        metrics[f"{key_prefix}{estimator_name}_hard_bma_eaurc_original_top5"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_bma_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_eaurc_top5"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_bma_correctnesses_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_aulc_original_top5"] = (
            area_under_lift_curve(
                estimate, gt_hard_bma_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_aulc_top5"] = (
            area_under_lift_curve(estimate, gt_hard_bma_correctnesses_top5).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_raulc_original_top5"] = (
            relative_area_under_lift_curve(
                estimate, gt_hard_bma_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_bma_raulc_top5"] = (
            relative_area_under_lift_curve(
                estimate, gt_hard_bma_correctnesses_top5
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_hard_bma_coverage_for_95_accuracy_original_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_bma_correctnesses_original_top5, accuracy=0.95
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_bma_coverage_for_95_accuracy_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_bma_correctnesses_top5, accuracy=0.95
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_bma_coverage_for_99_accuracy_original_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_bma_correctnesses_original_top5, accuracy=0.99
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_bma_coverage_for_99_accuracy_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_bma_correctnesses_top5, accuracy=0.99
        ).item()

        if is_soft_dataset:
            metrics[f"{key_prefix}{estimator_name}_soft_dual_bma_aurc"] = (
                area_under_risk_coverage_curve(
                    estimate, gt_soft_dual_bma_correctnesses
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_soft_dual_bma_abstinence_auc"
            ] = 1 - metrics[f"{key_prefix}{estimator_name}_soft_dual_bma_aurc"]
            metrics[f"{key_prefix}{estimator_name}_soft_dual_bma_eaurc"] = (
                excess_area_under_risk_coverage_curve(
                    estimate, gt_soft_dual_bma_correctnesses
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_dual_bma_aulc"] = (
                area_under_lift_curve(estimate, gt_soft_dual_bma_correctnesses).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_dual_bma_raulc"] = (
                relative_area_under_lift_curve(
                    estimate, gt_soft_dual_bma_correctnesses
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_soft_dual_bma_coverage_for_95_accuracy"
            ] = coverage_for_accuracy(
                estimate, gt_soft_dual_bma_correctnesses, accuracy=0.95
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_soft_dual_bma_coverage_for_99_accuracy"
            ] = coverage_for_accuracy(
                estimate, gt_soft_dual_bma_correctnesses, accuracy=0.99
            ).item()

            metrics[f"{key_prefix}{estimator_name}_soft_bma_aurc"] = (
                area_under_risk_coverage_curve(
                    estimate, gt_soft_bma_correctnesses
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_soft_bma_abstinence_auc"
            ] = 1 - metrics[f"{key_prefix}{estimator_name}_soft_bma_aurc"]
            metrics[f"{key_prefix}{estimator_name}_soft_bma_eaurc"] = (
                excess_area_under_risk_coverage_curve(
                    estimate, gt_soft_bma_correctnesses
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_bma_aulc"] = (
                area_under_lift_curve(estimate, gt_soft_bma_correctnesses).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_bma_raulc"] = (
                relative_area_under_lift_curve(
                    estimate, gt_soft_bma_correctnesses
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_soft_bma_coverage_for_95_accuracy"
            ] = coverage_for_accuracy(
                estimate, gt_soft_bma_correctnesses, accuracy=0.95
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_soft_bma_coverage_for_99_accuracy"
            ] = coverage_for_accuracy(
                estimate, gt_soft_bma_correctnesses, accuracy=0.99
            ).item()

            metrics[f"{key_prefix}{estimator_name}_soft_dual_bma_aurc_top5"] = (
                area_under_risk_coverage_curve(
                    estimate, gt_soft_dual_bma_correctnesses_top5
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_soft_dual_bma_abstinence_auc_top5"
            ] = 1 - metrics[f"{key_prefix}{estimator_name}_soft_dual_bma_aurc_top5"]
            metrics[f"{key_prefix}{estimator_name}_soft_dual_bma_eaurc_top5"] = (
                excess_area_under_risk_coverage_curve(
                    estimate, gt_soft_dual_bma_correctnesses_top5
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_dual_bma_aulc_top5"] = (
                area_under_lift_curve(
                    estimate, gt_soft_dual_bma_correctnesses_top5
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_dual_bma_raulc_top5"] = (
                relative_area_under_lift_curve(
                    estimate, gt_soft_dual_bma_correctnesses_top5
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_soft_dual_bma_coverage_for_95_accuracy_top5"
            ] = coverage_for_accuracy(
                estimate, gt_soft_dual_bma_correctnesses_top5, accuracy=0.95
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_soft_dual_bma_coverage_for_99_accuracy_top5"
            ] = coverage_for_accuracy(
                estimate, gt_soft_dual_bma_correctnesses_top5, accuracy=0.99
            ).item()

            metrics[f"{key_prefix}{estimator_name}_soft_bma_aurc_top5"] = (
                area_under_risk_coverage_curve(
                    estimate, gt_soft_bma_correctnesses_top5
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_soft_bma_abstinence_auc_top5"
            ] = 1 - metrics[f"{key_prefix}{estimator_name}_soft_bma_aurc_top5"]
            metrics[f"{key_prefix}{estimator_name}_soft_bma_eaurc_top5"] = (
                excess_area_under_risk_coverage_curve(
                    estimate, gt_soft_bma_correctnesses_top5
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_bma_aulc_top5"] = (
                area_under_lift_curve(estimate, gt_soft_bma_correctnesses_top5).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_bma_raulc_top5"] = (
                relative_area_under_lift_curve(
                    estimate, gt_soft_bma_correctnesses_top5
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_soft_bma_coverage_for_95_accuracy_top5"
            ] = coverage_for_accuracy(
                estimate, gt_soft_bma_correctnesses_top5, accuracy=0.95
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_soft_bma_coverage_for_99_accuracy_top5"
            ] = coverage_for_accuracy(
                estimate, gt_soft_bma_correctnesses_top5, accuracy=0.99
            ).item()

    return metrics


def evaluate_on_ood_detection(
    estimates: dict[str, Tensor],
    targets: dict[str, Tensor],
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluates the model on OOD detection metrics.

    Args:
        estimates: Dictionary of estimates.
        targets: Dictionary of targets.
        args: Additional arguments.

    Returns:
        A dictionary of OOD detection metrics.
    """
    metrics = {}
    for estimator_name, estimate in estimates.items():
        metrics[
            f"mixed_{args.dataset_id.replace('/', '_')}_{estimator_name}_auroc_oodness"
        ] = auroc(targets["gt_oodness"], estimate).item()

    return metrics


def evaluate_on_proper_scoring_and_calibration(
    model: nn.Module,
    estimates: dict[str, Tensor],
    log_probs: dict[str, Tensor],
    targets: dict[str, Tensor],
    is_soft_dataset: bool,
    args: argparse.Namespace,
    is_soft_upstream_dataset: bool | None,
) -> dict[str, float]:
    """Evaluates the model on proper scoring and calibration metrics.

    Args:
        model: The model to evaluate.
        estimates: Dictionary of estimates.
        log_probs: Dictionary of log probabilities.
        targets: Dictionary of targets.
        is_soft_dataset: Whether the dataset uses soft labels.
        args: Additional arguments.
        is_soft_upstream_dataset: Whether the upstream dataset uses soft labels.

    Returns:
        A dictionary of proper scoring and calibration metrics.
    """
    is_mixed_eval = is_soft_upstream_dataset is not None

    # For proper scoring and calibration, one of the datasets being soft is enough
    if is_mixed_eval:
        is_soft_dataset = is_soft_dataset or is_soft_upstream_dataset

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id.replace('/', '_')}_" if is_mixed_eval else ""

    # Proper scoring and calibration for correctness of prediction
    correctness_estimator_names = [
        "one_minus_expected_max_probs",
        "one_minus_max_probs_of_dual_bma",
        "one_minus_max_probs_of_bma",
    ]

    if isinstance(model, DUQWrapper):
        correctness_estimator_names.append("duq_values")

    if isinstance(model, BaseCorrectnessPredictionWrapper):
        correctness_estimator_names.append("error_probabilities")

    gt_hard_dual_bma_correctnesses_original = targets[
        "gt_hard_dual_bma_correctnesses_original"
    ]
    gt_hard_dual_bma_correctnesses = targets["gt_hard_dual_bma_correctnesses"]
    gt_hard_bma_correctnesses_original = targets["gt_hard_bma_correctnesses_original"]
    gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses"]

    gt_hard_dual_bma_correctnesses_original_top5 = targets[
        "gt_hard_dual_bma_correctnesses_original_top5"
    ]
    gt_hard_dual_bma_correctnesses_top5 = targets["gt_hard_dual_bma_correctnesses_top5"]
    gt_hard_bma_correctnesses_original_top5 = targets[
        "gt_hard_bma_correctnesses_original_top5"
    ]
    gt_hard_bma_correctnesses_top5 = targets["gt_hard_bma_correctnesses_top5"]

    if is_soft_dataset:
        gt_soft_dual_bma_correctnesses = targets["gt_soft_dual_bma_correctnesses"]
        gt_soft_bma_correctnesses = targets["gt_soft_bma_correctnesses"]

        gt_soft_dual_bma_correctnesses_top5 = targets[
            "gt_soft_dual_bma_correctnesses_top5"
        ]
        gt_soft_bma_correctnesses_top5 = targets["gt_soft_bma_correctnesses_top5"]

    for estimator_name in correctness_estimator_names:
        estimate = estimates[estimator_name]

        estimate = 1 - estimate  # convert to correctness probability

        # {Hard, Soft}-label correctness
        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_hard_dual_bma_correctness_original"
        ] = binary_log_probability(
            estimate, gt_hard_dual_bma_correctnesses_original
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_hard_dual_bma_correctness"
        ] = binary_log_probability(estimate, gt_hard_dual_bma_correctnesses).item()
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_dual_bma_correctness_original"
        ] = binary_brier(estimate, gt_hard_dual_bma_correctnesses_original).item()
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_dual_bma_correctness"
        ] = binary_brier(estimate, gt_hard_dual_bma_correctnesses).item()
        metrics[
            f"{key_prefix}{estimator_name}_ece_hard_dual_bma_correctness_original"
        ] = calibration_error(
            confidences=estimate,
            correctnesses=gt_hard_dual_bma_correctnesses_original,
            num_bins=15,
            norm="l1",
        ).item()
        metrics[f"{key_prefix}{estimator_name}_ece_hard_dual_bma_correctness"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_dual_bma_correctnesses,
                num_bins=15,
                norm="l1",
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_mce_hard_dual_bma_correctness_original"
        ] = calibration_error(
            confidences=estimate,
            correctnesses=gt_hard_dual_bma_correctnesses_original,
            num_bins=15,
            norm="inf",
        ).item()
        metrics[f"{key_prefix}{estimator_name}_mce_hard_dual_bma_correctness"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_dual_bma_correctnesses,
                num_bins=15,
                norm="inf",
            ).item()
        )

        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_hard_bma_correctness_original"
        ] = binary_log_probability(estimate, gt_hard_bma_correctnesses_original).item()
        metrics[f"{key_prefix}{estimator_name}_log_prob_score_hard_bma_correctness"] = (
            binary_log_probability(estimate, gt_hard_bma_correctnesses).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_bma_correctness_original"
        ] = binary_brier(estimate, gt_hard_bma_correctnesses_original).item()
        metrics[f"{key_prefix}{estimator_name}_brier_score_hard_bma_correctness"] = (
            binary_brier(estimate, gt_hard_bma_correctnesses).item()
        )
        metrics[f"{key_prefix}{estimator_name}_ece_hard_bma_correctness_original"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_bma_correctnesses_original,
                num_bins=15,
                norm="l1",
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_ece_hard_bma_correctness"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_bma_correctnesses,
                num_bins=15,
                norm="l1",
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_mce_hard_bma_correctness_original"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_bma_correctnesses_original,
                num_bins=15,
                norm="inf",
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_mce_hard_bma_correctness"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_bma_correctnesses,
                num_bins=15,
                norm="inf",
            ).item()
        )

        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_hard_dual_bma_correctness_original_top5"
        ] = binary_log_probability(
            estimate, gt_hard_dual_bma_correctnesses_original_top5
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_hard_dual_bma_correctness_top5"
        ] = binary_log_probability(estimate, gt_hard_dual_bma_correctnesses_top5).item()
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_dual_bma_correctness_original_top5"
        ] = binary_brier(estimate, gt_hard_dual_bma_correctnesses_original_top5).item()
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_dual_bma_correctness_top5"
        ] = binary_brier(estimate, gt_hard_dual_bma_correctnesses_top5).item()
        metrics[
            f"{key_prefix}{estimator_name}_ece_hard_dual_bma_correctness_original_top5"
        ] = calibration_error(
            confidences=estimate,
            correctnesses=gt_hard_dual_bma_correctnesses_original_top5,
            num_bins=15,
            norm="l1",
        ).item()
        metrics[f"{key_prefix}{estimator_name}_ece_hard_dual_bma_correctness_top5"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_dual_bma_correctnesses_top5,
                num_bins=15,
                norm="l1",
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_mce_hard_dual_bma_correctness_original_top5"
        ] = calibration_error(
            confidences=estimate,
            correctnesses=gt_hard_dual_bma_correctnesses_original_top5,
            num_bins=15,
            norm="inf",
        ).item()
        metrics[f"{key_prefix}{estimator_name}_mce_hard_dual_bma_correctness_top5"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_dual_bma_correctnesses_top5,
                num_bins=15,
                norm="inf",
            ).item()
        )

        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_hard_bma_correctness_original_top5"
        ] = binary_log_probability(
            estimate, gt_hard_bma_correctnesses_original_top5
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_hard_bma_correctness_top5"
        ] = binary_log_probability(estimate, gt_hard_bma_correctnesses_top5).item()
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_bma_correctness_original_top5"
        ] = binary_brier(estimate, gt_hard_bma_correctnesses_original_top5).item()
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_bma_correctness_top5"
        ] = binary_brier(estimate, gt_hard_bma_correctnesses_top5).item()
        metrics[
            f"{key_prefix}{estimator_name}_ece_hard_bma_correctness_original_top5"
        ] = calibration_error(
            confidences=estimate,
            correctnesses=gt_hard_bma_correctnesses_original_top5,
            num_bins=15,
            norm="l1",
        ).item()
        metrics[f"{key_prefix}{estimator_name}_ece_hard_bma_correctness_top5"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_bma_correctnesses_top5,
                num_bins=15,
                norm="l1",
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_mce_hard_bma_correctness_original_top5"
        ] = calibration_error(
            confidences=estimate,
            correctnesses=gt_hard_bma_correctnesses_original_top5,
            num_bins=15,
            norm="inf",
        ).item()
        metrics[f"{key_prefix}{estimator_name}_mce_hard_bma_correctness_top5"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_bma_correctnesses_top5,
                num_bins=15,
                norm="inf",
            ).item()
        )

        if is_soft_dataset:
            metrics[
                f"{key_prefix}{estimator_name}_log_prob_score_soft_dual_bma_correctness"
            ] = binary_log_probability(estimate, gt_soft_dual_bma_correctnesses).item()
            metrics[
                f"{key_prefix}{estimator_name}_brier_score_soft_dual_bma_correctness"
            ] = binary_brier(estimate, gt_soft_dual_bma_correctnesses).item()
            metrics[f"{key_prefix}{estimator_name}_ece_soft_dual_bma_correctness"] = (
                calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_dual_bma_correctnesses,
                    num_bins=15,
                    norm="l1",
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_mce_soft_dual_bma_correctness"] = (
                calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_dual_bma_correctnesses,
                    num_bins=15,
                    norm="inf",
                ).item()
            )

            metrics[
                f"{key_prefix}{estimator_name}_log_prob_score_soft_bma_correctness"
            ] = binary_log_probability(estimate, gt_soft_bma_correctnesses).item()
            metrics[
                f"{key_prefix}{estimator_name}_brier_score_soft_bma_correctness"
            ] = binary_brier(estimate, gt_soft_bma_correctnesses).item()
            metrics[f"{key_prefix}{estimator_name}_ece_soft_bma_correctness"] = (
                calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_bma_correctnesses,
                    num_bins=15,
                    norm="l1",
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_mce_soft_bma_correctness"] = (
                calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_bma_correctnesses,
                    num_bins=15,
                    norm="inf",
                ).item()
            )

            metrics[
                f"{key_prefix}{estimator_name}_log_prob_score_soft_dual_bma_correctness_top5"
            ] = binary_log_probability(
                estimate, gt_soft_dual_bma_correctnesses_top5
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_brier_score_soft_dual_bma_correctness_top5"
            ] = binary_brier(estimate, gt_soft_dual_bma_correctnesses_top5).item()
            metrics[
                f"{key_prefix}{estimator_name}_ece_soft_dual_bma_correctness_top5"
            ] = calibration_error(
                confidences=estimate,
                correctnesses=gt_soft_dual_bma_correctnesses_top5,
                num_bins=15,
                norm="l1",
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_mce_soft_dual_bma_correctness_top5"
            ] = calibration_error(
                confidences=estimate,
                correctnesses=gt_soft_dual_bma_correctnesses_top5,
                num_bins=15,
                norm="inf",
            ).item()

            metrics[
                f"{key_prefix}{estimator_name}_log_prob_score_soft_bma_correctness_top5"
            ] = binary_log_probability(estimate, gt_soft_bma_correctnesses_top5).item()
            metrics[
                f"{key_prefix}{estimator_name}_brier_score_soft_bma_correctness_top5"
            ] = binary_brier(estimate, gt_soft_bma_correctnesses_top5).item()
            metrics[f"{key_prefix}{estimator_name}_ece_soft_bma_correctness_top5"] = (
                calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_bma_correctnesses_top5,
                    num_bins=15,
                    norm="l1",
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_mce_soft_bma_correctness_top5"] = (
                calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_bma_correctnesses_top5,
                    num_bins=15,
                    norm="inf",
                ).item()
            )

    # Proper scoring for aleatoric uncertainty
    gt_hard_labels_original = targets["gt_hard_labels_original"]
    gt_hard_labels = targets["gt_hard_labels"]

    metrics[f"{key_prefix}log_prob_score_hard_dual_bma_aleatoric_original"] = (
        multiclass_log_probability(
            log_probs["log_dual_bmas"], gt_hard_labels_original
        ).item()
    )
    metrics[f"{key_prefix}log_prob_score_hard_dual_bma_aleatoric"] = (
        multiclass_log_probability(log_probs["log_dual_bmas"], gt_hard_labels).item()
    )
    metrics[f"{key_prefix}brier_score_hard_dual_bma_aleatoric_original"] = (
        multiclass_brier(
            log_probs["log_dual_bmas"], gt_hard_labels_original, is_soft_targets=False
        ).item()
    )
    metrics[f"{key_prefix}brier_score_hard_dual_bma_aleatoric"] = multiclass_brier(
        log_probs["log_dual_bmas"], gt_hard_labels, is_soft_targets=False
    ).item()

    metrics[f"{key_prefix}log_prob_score_hard_bma_aleatoric_original"] = (
        multiclass_log_probability(
            log_probs["log_bmas"], gt_hard_labels_original
        ).item()
    )
    metrics[f"{key_prefix}log_prob_score_hard_bma_aleatoric"] = (
        multiclass_log_probability(log_probs["log_bmas"], gt_hard_labels).item()
    )
    metrics[f"{key_prefix}brier_score_hard_bma_aleatoric_original"] = multiclass_brier(
        log_probs["log_bmas"], gt_hard_labels_original, is_soft_targets=False
    ).item()
    metrics[f"{key_prefix}brier_score_hard_bma_aleatoric"] = multiclass_brier(
        log_probs["log_bmas"], gt_hard_labels, is_soft_targets=False
    ).item()

    if is_soft_dataset:
        gt_soft_labels = targets["gt_soft_labels"]

        metrics[f"{key_prefix}log_prob_score_soft_dual_bma_aleatoric"] = (
            multiclass_log_probability(
                log_probs["log_dual_bmas"], gt_soft_labels
            ).item()
        )
        metrics[f"{key_prefix}brier_score_soft_dual_bma_aleatoric"] = multiclass_brier(
            log_probs["log_dual_bmas"], gt_soft_labels, is_soft_targets=True
        ).item()

        metrics[f"{key_prefix}log_prob_score_soft_bma_aleatoric"] = (
            multiclass_log_probability(log_probs["log_bmas"], gt_soft_labels).item()
        )
        metrics[f"{key_prefix}brier_score_soft_bma_aleatoric"] = multiclass_brier(
            log_probs["log_bmas"], gt_soft_labels, is_soft_targets=True
        ).item()

    return metrics


def evaluate_on_bregman(
    estimates: dict[str, Tensor],
    targets: dict[str, Tensor],
    is_soft_dataset: bool,
    args: argparse.Namespace,
    is_soft_upstream_dataset: bool | None,
) -> dict[str, float]:
    """Evaluates the model using the Bregman decomposition's terms.

    Args:
        estimates: Dictionary of estimates.
        targets: Dictionary of targets.
        is_soft_dataset: Whether the dataset uses soft labels.
        args: Additional arguments.
        is_soft_upstream_dataset: Whether the upstream dataset uses soft labels.

    Returns:
        A dictionary of metrics using the Bregman decomposition.
    """
    is_mixed_eval = is_soft_upstream_dataset is not None

    # For Bregman, both datasets need to be soft
    if is_mixed_eval:
        is_soft_dataset = is_soft_dataset and is_soft_upstream_dataset

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id.replace('/', '_')}_" if is_mixed_eval else ""

    gt_predictives_bregman_dual_bma = targets["gt_predictives_bregman_dual_bma"]
    gt_predictives_bregman_bma = targets["gt_predictives_bregman_bma"]

    gt_total_predictives_bregman_dual_bma = targets[
        "gt_total_predictives_bregman_dual_bma"
    ]
    gt_total_predictives_bregman_bma = targets["gt_total_predictives_bregman_bma"]

    if is_soft_dataset:
        gt_biases_bregman_dual_bma = targets["gt_biases_bregman_dual_bma"]
        gt_biases_bregman_bma = targets["gt_biases_bregman_bma"]

    if is_soft_dataset:
        gt_aleatorics_bregman = targets["gt_aleatorics_bregman"]
        multi_label_indices = (gt_aleatorics_bregman > 0).int()

    gt_epistemics_bregman = targets["gt_epistemics_bregman"]

    for estimator_name, estimate in estimates.items():
        metrics[f"{key_prefix}{estimator_name}_rank_correlation_bregman_eu"] = (
            spearmanr(estimate, gt_epistemics_bregman).item()
        )
        metrics[f"{key_prefix}{estimator_name}_mse_bregman_eu"] = (
            (estimate - gt_epistemics_bregman).square().mean().item()
        )
        metrics[f"{key_prefix}{estimator_name}_mae_bregman_eu"] = (
            (estimate - gt_epistemics_bregman).abs().mean().item()
        )

        if is_soft_dataset:
            metrics[f"{key_prefix}{estimator_name}_rank_correlation_bregman_au"] = (
                spearmanr(estimate, gt_aleatorics_bregman).item()
            )
            metrics[f"{key_prefix}{estimator_name}_mse_bregman_au"] = (
                (estimate - gt_aleatorics_bregman).square().mean().item()
            )
            metrics[f"{key_prefix}{estimator_name}_mae_bregman_au"] = (
                (estimate - gt_aleatorics_bregman).abs().mean().item()
            )

            metrics[f"{key_prefix}{estimator_name}_auroc_multiple_labels"] = auroc(
                multi_label_indices, estimate
            ).item()

        metrics[
            f"{key_prefix}{estimator_name}_rank_correlation_bregman_pu_dual_bma"
        ] = spearmanr(estimate, gt_predictives_bregman_dual_bma).item()
        metrics[f"{key_prefix}{estimator_name}_mse_bregman_pu_dual_bma"] = (
            (estimate - gt_predictives_bregman_dual_bma).square().mean().item()
        )
        metrics[f"{key_prefix}{estimator_name}_mae_bregman_pu_dual_bma"] = (
            (estimate - gt_predictives_bregman_dual_bma).abs().mean().item()
        )

        metrics[
            f"{key_prefix}{estimator_name}_rank_correlation_bregman_total_pu_dual_bma"
        ] = spearmanr(estimate, gt_total_predictives_bregman_dual_bma).item()
        metrics[f"{key_prefix}{estimator_name}_mse_bregman_total_pu_dual_bma"] = (
            (estimate - gt_total_predictives_bregman_dual_bma).square().mean().item()
        )
        metrics[f"{key_prefix}{estimator_name}_mae_bregman_total_pu_dual_bma"] = (
            (estimate - gt_total_predictives_bregman_dual_bma).abs().mean().item()
        )

        metrics[f"{key_prefix}{estimator_name}_rank_correlation_bregman_pu_bma"] = (
            spearmanr(estimate, gt_predictives_bregman_bma).item()
        )
        metrics[f"{key_prefix}{estimator_name}_mse_bregman_pu_bma"] = (
            (estimate - gt_predictives_bregman_bma).square().mean().item()
        )
        metrics[f"{key_prefix}{estimator_name}_mae_bregman_pu_bma"] = (
            (estimate - gt_predictives_bregman_bma).abs().mean().item()
        )

        metrics[
            f"{key_prefix}{estimator_name}_rank_correlation_bregman_total_pu_bma"
        ] = spearmanr(estimate, gt_total_predictives_bregman_bma).item()
        metrics[f"{key_prefix}{estimator_name}_mse_bregman_total_pu_bma"] = (
            (estimate - gt_total_predictives_bregman_bma).square().mean().item()
        )
        metrics[f"{key_prefix}{estimator_name}_mae_bregman_total_pu_bma"] = (
            (estimate - gt_total_predictives_bregman_bma).abs().mean().item()
        )

        if is_soft_dataset:
            metrics[
                f"{key_prefix}{estimator_name}_rank_correlation_bregman_b_dual_bma"
            ] = spearmanr(estimate, gt_biases_bregman_dual_bma).item()
            metrics[f"{key_prefix}{estimator_name}_mse_bregman_b_dual_bma"] = (
                (estimate - gt_biases_bregman_dual_bma).square().mean().item()
            )
            metrics[f"{key_prefix}{estimator_name}_mae_bregman_b_dual_bma"] = (
                (estimate - gt_biases_bregman_dual_bma).abs().mean().item()
            )

            metrics[f"{key_prefix}{estimator_name}_rank_correlation_bregman_b_bma"] = (
                spearmanr(estimate, gt_biases_bregman_bma).item()
            )
            metrics[f"{key_prefix}{estimator_name}_mse_bregman_b_bma"] = (
                (estimate - gt_biases_bregman_bma).square().mean().item()
            )
            metrics[f"{key_prefix}{estimator_name}_mae_bregman_b_bma"] = (
                (estimate - gt_biases_bregman_bma).abs().mean().item()
            )

    return metrics


def evaluate_on_correlation_of_estimators(
    model: nn.Module,
    estimates: dict[str, Tensor],
    output_dir: Path,
    save_prefix: str,
    args: argparse.Namespace,
    is_soft_upstream_dataset: bool | None,
) -> dict[str, float]:
    """Evaluates the correlation of estimators.

    Args:
        model: The model to evaluate.
        estimates: Dictionary of estimates.
        output_dir: Directory to save output.
        save_prefix: Prefix for saving results.
        args: Additional arguments.
        is_soft_upstream_dataset: Whether the upstream dataset uses soft labels.

    Returns:
        A dictionary of correlation metrics.
    """
    metrics = {}

    is_mixed_eval = is_soft_upstream_dataset is not None
    key_prefix = f"mixed_{args.dataset_id.replace('/', '_')}_" if is_mixed_eval else ""

    if isinstance(model, DDUWrapper):
        ddu_aleatoric = estimates["expected_entropies"]
        ddu_epistemic = estimates["gmm_neg_log_densities"]

        torch.save(
            (ddu_aleatoric, ddu_epistemic), f"{output_dir}/{save_prefix}ddu_au_eu.pt"
        )

        metrics[f"{key_prefix}correlation_ddu_au_eu"] = float(
            pearsonr(ddu_aleatoric, ddu_epistemic)
        )
        metrics[f"{key_prefix}rank_correlation_ddu_au_eu"] = float(
            spearmanr(ddu_aleatoric, ddu_epistemic)
        )
    elif isinstance(model, MahalanobisWrapper):
        mahalanobis_aleatoric = estimates["expected_entropies"]
        mahalanobis_epistemic = estimates["mahalanobis_values"]

        torch.save(
            (mahalanobis_aleatoric, mahalanobis_epistemic),
            f"{output_dir}/{save_prefix}mahalanobis_au_eu.pt",
        )

        metrics[f"{key_prefix}correlation_mahalanobis_au_eu"] = float(
            pearsonr(mahalanobis_aleatoric, mahalanobis_epistemic)
        )
        metrics[f"{key_prefix}rank_correlation_mahalanobis_au_eu"] = float(
            spearmanr(mahalanobis_aleatoric, mahalanobis_epistemic)
        )

    # Gaussian logit decomposition of Kendall and Gal
    kendall_gal_aleatoric = estimates["expected_entropies"]
    kendall_gal_epistemic_prob = estimates["expected_variances_of_probs"]
    kendall_gal_epistemic_logit = estimates["expected_variances_of_logits"]

    torch.save(
        (kendall_gal_aleatoric, kendall_gal_epistemic_prob),
        f"{output_dir}/{save_prefix}kendall_gal_au_eu_prob.pt",
    )

    metrics[f"{key_prefix}correlation_kendall_gal_au_eu_prob"] = float(
        pearsonr(kendall_gal_aleatoric, kendall_gal_epistemic_prob)
    )
    metrics[f"{key_prefix}rank_correlation_kendall_gal_au_eu_prob"] = float(
        spearmanr(kendall_gal_aleatoric, kendall_gal_epistemic_prob)
    )

    torch.save(
        (kendall_gal_aleatoric, kendall_gal_epistemic_logit),
        f"{output_dir}/{save_prefix}kendall_gal_au_eu_logit.pt",
    )

    metrics[f"{key_prefix}correlation_kendall_gal_au_eu_logit"] = float(
        pearsonr(kendall_gal_aleatoric, kendall_gal_epistemic_logit)
    )
    metrics[f"{key_prefix}rank_correlation_kendall_gal_au_eu_logit"] = float(
        spearmanr(kendall_gal_aleatoric, kendall_gal_epistemic_logit)
    )

    # Internal Gaussian logit decomposition of Kendall and Gal
    if isinstance(model, HetClassNNWrapper):
        kendall_gal_aleatoric = estimates["expected_entropies"]
        kendall_gal_epistemic_internal_prob = estimates[
            "expected_variances_of_internal_probs"
        ]
        kendall_gal_epistemic_internal_logit = estimates[
            "expected_variances_of_internal_logits"
        ]

        torch.save(
            (kendall_gal_aleatoric, kendall_gal_epistemic_internal_prob),
            f"{output_dir}/{save_prefix}kendall_gal_au_eu_internal_prob.pt",
        )

        metrics[f"{key_prefix}correlation_kendall_gal_au_eu_internal_prob"] = float(
            pearsonr(kendall_gal_aleatoric, kendall_gal_epistemic_internal_prob)
        )
        metrics[f"{key_prefix}rank_correlation_kendall_gal_au_eu_internal_prob"] = (
            float(spearmanr(kendall_gal_aleatoric, kendall_gal_epistemic_internal_prob))
        )

        torch.save(
            (kendall_gal_aleatoric, kendall_gal_epistemic_internal_logit),
            f"{output_dir}/{save_prefix}kendall_gal_au_eu_internal_logit.pt",
        )

        metrics[f"{key_prefix}correlation_kendall_gal_au_eu_internal_logit"] = float(
            pearsonr(kendall_gal_aleatoric, kendall_gal_epistemic_internal_logit)
        )
        metrics[f"{key_prefix}rank_correlation_kendall_gal_au_eu_internal_logit"] = (
            float(
                spearmanr(kendall_gal_aleatoric, kendall_gal_epistemic_internal_logit)
            )
        )

    return metrics


def evaluate_on_correlation_of_decompositions(
    estimates: dict[str, Tensor],
    targets: dict[str, Tensor],
    is_soft_dataset: bool,
    output_dir: Path,
    save_prefix: str,
    args: argparse.Namespace,
    is_soft_upstream_dataset: bool | None,
) -> dict[str, float]:
    """Evaluates the correlation of decompositions.

    Args:
        estimates: Dictionary of estimates.
        targets: Dictionary of targets.
        is_soft_dataset: Whether the dataset uses soft labels.
        output_dir: Directory to save output.
        save_prefix: Prefix for saving results.
        args: Additional arguments.
        is_soft_upstream_dataset: Whether the upstream dataset uses soft labels.

    Returns:
        A dictionary of correlation metrics for decompositions.
    """
    is_mixed_eval = is_soft_upstream_dataset is not None

    # For Bregman, both datasets need to be soft
    if is_mixed_eval:
        is_soft_dataset = is_soft_dataset and is_soft_upstream_dataset

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id.replace('/', '_')}_" if is_mixed_eval else ""

    # Information-theoretical decomposition
    entropies_of_bma = estimates["entropies_of_bma"]
    expected_entropies = estimates["expected_entropies"]
    jensen_shannon_divergences = estimates["jensen_shannon_divergences"]

    torch.save(
        (expected_entropies, jensen_shannon_divergences),
        f"{output_dir}/{save_prefix}it_au_eu.pt",
    )

    metrics[f"{key_prefix}rank_correlation_it_au_eu"] = float(
        spearmanr(expected_entropies, jensen_shannon_divergences)
    )
    metrics[f"{key_prefix}correlation_it_au_eu"] = float(
        pearsonr(expected_entropies, jensen_shannon_divergences)
    )

    metrics[f"{key_prefix}rank_correlation_it_au_pu"] = float(
        spearmanr(expected_entropies, entropies_of_bma)
    )
    metrics[f"{key_prefix}correlation_it_au_pu"] = float(
        pearsonr(expected_entropies, entropies_of_bma)
    )

    metrics[f"{key_prefix}rank_correlation_it_eu_pu"] = float(
        spearmanr(jensen_shannon_divergences, entropies_of_bma)
    )
    metrics[f"{key_prefix}correlation_it_eu_pu"] = float(
        pearsonr(jensen_shannon_divergences, entropies_of_bma)
    )

    # Bregman decomposition estimates
    expected_divergences = estimates["expected_divergences"]
    expected_entropies_plus_expected_divergences = estimates[
        "expected_entropies_plus_expected_divergences"
    ]

    torch.save(
        (expected_divergences, expected_entropies),
        f"{output_dir}/{save_prefix}bregman_eu_au_hat.pt",
    )

    metrics[f"{key_prefix}rank_correlation_bregman_eu_au_hat"] = float(
        spearmanr(expected_divergences, expected_entropies)
    )
    metrics[f"{key_prefix}correlation_bregman_eu_au_hat"] = float(
        pearsonr(expected_divergences, expected_entropies)
    )
    metrics[f"{key_prefix}rank_correlation_bregman_eu_pu_hat"] = float(
        spearmanr(expected_divergences, expected_entropies_plus_expected_divergences)
    )
    metrics[f"{key_prefix}correlation_bregman_eu_pu_hat"] = float(
        pearsonr(expected_divergences, expected_entropies_plus_expected_divergences)
    )

    metrics[f"{key_prefix}rank_correlation_bregman_au_hat_pu_hat"] = float(
        spearmanr(expected_entropies, expected_entropies_plus_expected_divergences)
    )
    metrics[f"{key_prefix}correlation_bregman_au_hat_pu_hat"] = float(
        pearsonr(expected_entropies, expected_entropies_plus_expected_divergences)
    )

    # Bregman decomposition GTs
    gt_predictives_bregman_dual_bma = targets["gt_predictives_bregman_dual_bma"]
    gt_predictives_bregman_bma = targets["gt_predictives_bregman_bma"]

    gt_total_predictives_bregman_dual_bma = targets[
        "gt_total_predictives_bregman_dual_bma"
    ]
    gt_total_predictives_bregman_bma = targets["gt_total_predictives_bregman_bma"]

    if is_soft_dataset:
        gt_biases_bregman_dual_bma = targets["gt_biases_bregman_dual_bma"]
        gt_biases_bregman_bma = targets["gt_biases_bregman_bma"]

    if is_soft_dataset:
        gt_aleatorics_bregman = targets["gt_aleatorics_bregman"]

    gt_epistemics_bregman = targets["gt_epistemics_bregman"]

    can_evaluate_au_eu = is_soft_dataset
    can_evaluate_au_b = can_evaluate_au_pu = can_evaluate_b_pu = is_soft_dataset
    can_evaluate_eu_b = can_evaluate_au_b

    if can_evaluate_au_eu:
        torch.save(
            (gt_aleatorics_bregman, gt_epistemics_bregman),
            f"{output_dir}/{save_prefix}bregman_au_eu.pt",
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_eu"] = float(
            spearmanr(gt_aleatorics_bregman, gt_epistemics_bregman)
        )
        metrics[f"{key_prefix}correlation_bregman_au_eu"] = float(
            pearsonr(gt_aleatorics_bregman, gt_epistemics_bregman)
        )

    if can_evaluate_au_b:
        metrics[f"{key_prefix}rank_correlation_bregman_au_b_dual_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_biases_bregman_dual_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_au_b_dual_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_biases_bregman_dual_bma)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_b_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_biases_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_au_b_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_biases_bregman_bma)
        )

    if can_evaluate_au_pu:
        metrics[f"{key_prefix}rank_correlation_bregman_au_pu_dual_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_predictives_bregman_dual_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_au_pu_dual_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_predictives_bregman_dual_bma)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_pu_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_predictives_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_au_pu_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_predictives_bregman_bma)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_total_pu_dual_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_total_predictives_bregman_dual_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_au_total_pu_dual_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_total_predictives_bregman_dual_bma)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_total_pu_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_total_predictives_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_au_total_pu_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_total_predictives_bregman_bma)
        )

    if can_evaluate_b_pu:
        metrics[f"{key_prefix}rank_correlation_bregman_b_pu_dual_bma"] = float(
            spearmanr(gt_biases_bregman_dual_bma, gt_predictives_bregman_dual_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_b_pu_dual_bma"] = float(
            pearsonr(gt_biases_bregman_dual_bma, gt_predictives_bregman_dual_bma)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_b_pu_bma"] = float(
            spearmanr(gt_biases_bregman_bma, gt_predictives_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_b_pu_bma"] = float(
            pearsonr(gt_biases_bregman_bma, gt_predictives_bregman_bma)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_b_total_pu_dual_bma"] = float(
            spearmanr(gt_biases_bregman_dual_bma, gt_total_predictives_bregman_dual_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_b_total_pu_dual_bma"] = float(
            pearsonr(gt_biases_bregman_dual_bma, gt_total_predictives_bregman_dual_bma)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_b_total_pu_bma"] = float(
            spearmanr(gt_biases_bregman_bma, gt_total_predictives_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_b_total_pu_bma"] = float(
            pearsonr(gt_biases_bregman_bma, gt_total_predictives_bregman_bma)
        )

    if can_evaluate_eu_b:
        metrics[f"{key_prefix}rank_correlation_bregman_eu_b_dual_bma"] = float(
            spearmanr(gt_epistemics_bregman, gt_biases_bregman_dual_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_eu_b_dual_bma"] = float(
            pearsonr(gt_epistemics_bregman, gt_biases_bregman_dual_bma)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_eu_b_bma"] = float(
            spearmanr(gt_epistemics_bregman, gt_biases_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_eu_b_bma"] = float(
            pearsonr(gt_epistemics_bregman, gt_biases_bregman_bma)
        )

    metrics[f"{key_prefix}rank_correlation_bregman_eu_pu_dual_bma"] = float(
        spearmanr(gt_epistemics_bregman, gt_predictives_bregman_dual_bma)
    )
    metrics[f"{key_prefix}correlation_bregman_eu_pu_dual_bma"] = float(
        pearsonr(gt_epistemics_bregman, gt_predictives_bregman_dual_bma)
    )

    metrics[f"{key_prefix}rank_correlation_bregman_eu_pu_bma"] = float(
        spearmanr(gt_epistemics_bregman, gt_predictives_bregman_bma)
    )
    metrics[f"{key_prefix}correlation_bregman_eu_pu_bma"] = float(
        pearsonr(gt_epistemics_bregman, gt_predictives_bregman_bma)
    )

    metrics[f"{key_prefix}rank_correlation_bregman_eu_total_pu_dual_bma"] = float(
        spearmanr(gt_epistemics_bregman, gt_total_predictives_bregman_dual_bma)
    )
    metrics[f"{key_prefix}correlation_bregman_eu_total_pu_dual_bma"] = float(
        pearsonr(gt_epistemics_bregman, gt_total_predictives_bregman_dual_bma)
    )

    metrics[f"{key_prefix}rank_correlation_bregman_eu_total_pu_bma"] = float(
        spearmanr(gt_epistemics_bregman, gt_total_predictives_bregman_bma)
    )
    metrics[f"{key_prefix}correlation_bregman_eu_total_pu_bma"] = float(
        pearsonr(gt_epistemics_bregman, gt_total_predictives_bregman_bma)
    )

    return metrics


def forward_general_model_on_loader(
    model: nn.Module,
    loader: DataLoader | PrefetchLoader,
    is_soft_dataset: bool,
    amp_autocast: Callable,
    device: torch.device,
    storage_device: torch.device,
    args: argparse.Namespace,
    log_dual_bmas: Tensor,
    log_bmas: Tensor,
    gt_epistemics_bregman: Tensor,
    time_forward_m: AverageMeter,
    expected_entropies: Tensor,
    expected_entropies_plus_expected_divergences: Tensor,
    one_minus_expected_max_probs: Tensor,
    entropies_of_bma: Tensor,
    entropies_of_dual_bma: Tensor,
    one_minus_max_probs_of_bma: Tensor,
    one_minus_max_probs_of_dual_bma: Tensor,
    jensen_shannon_divergences: Tensor,
    dempster_shafer_values: Tensor,
    expected_variances_of_probs: Tensor,
    expected_variances_of_logits: Tensor,
    loss_values: Tensor,
    gmm_neg_log_densities: Tensor,
    error_probabilities: Tensor,
    duq_values: Tensor,
    mahalanobis_values: Tensor,
    expected_variances_of_internal_probs: Tensor,
    expected_variances_of_internal_logits: Tensor,
    gt_aleatorics_bregman: Tensor,
    gt_biases_bregman_dual_bma: Tensor,
    gt_biases_bregman_bma: Tensor,
    gt_predictives_bregman_dual_bma: Tensor,
    gt_predictives_bregman_bma: Tensor,
    gt_total_predictives_bregman_dual_bma: Tensor,
    gt_total_predictives_bregman_bma: Tensor,
    gt_soft_labels: Tensor,
    gt_hard_labels: Tensor,
    gt_hard_labels_original: Tensor,
) -> None:
    """Performs the forward pass of a general model on a data loader.

    Args:
        model: The model to evaluate.
        loader: The data loader.
        is_soft_dataset: Whether the dataset uses soft labels.
        amp_autocast: Function for automatic mixed precision.
        device: The device to use for computation.
        storage_device: The device to use for storing results.
        args: Additional arguments.
        log_dual_bmas: Tensor to store log dual BMA values.
        log_bmas: Tensor to store log BMA values.
        gt_epistemics_bregman: Tensor to store ground truth epistemic uncertainties.
        time_forward_m: AverageMeter to track forward pass time.
        expected_entropies: Tensor to store expected entropies.
        expected_entropies_plus_expected_divergences: Tensor to store sum of expected
            entropies and expected divergences.
        one_minus_expected_max_probs: Tensor to store 1 minus expected max
            probabilities.
        entropies_of_bma: Tensor to store entropies of BMA.
        entropies_of_dual_bma: Tensor to store entropies of the dual BMA.
        one_minus_max_probs_of_bma: Tensor to store 1 minus max probabilities of BMA.
        one_minus_max_probs_of_dual_bma: Tensor to store 1 minus max probabilities of
            the dual BMA.
        jensen_shannon_divergences: Tensor to store Jensen-Shannon divergences.
        dempster_shafer_values: Tensor to store Dempster-Shafer values.
        expected_variances_of_probs: Tensor to store expected variances of
            probabilities.
        expected_variances_of_logits: Tensor to store expected variances of logits.
        loss_values: Tensor to store loss prediction values.
        gmm_neg_log_densities: Tensor to store GMM negative log densities.
        error_probabilities: Tensor to store error probabilities.
        duq_values: Tensor to store DUQ values.
        mahalanobis_values: Tensor to store Mahalanobis distance values.
        expected_variances_of_internal_probs: Tensor to store expected variances of
            internal probabilities.
        expected_variances_of_internal_logits: Tensor to store expected variances of
            internal logits.
        gt_aleatorics_bregman: Tensor to store ground truth aleatoric uncertainties.
        gt_biases_bregman_dual_bma: Tensor to store ground truth Bregman biases for the
            dual BMA.
        gt_biases_bregman_bma: Tensor to store ground truth Bregman biases for BMA.
        gt_predictives_bregman_dual_bma: Tensor to store ground truth Bregman predictive
            uncertainties for the dual BMA.
        gt_predictives_bregman_bma: Tensor to store ground truth Bregman predictive
            uncertainties for BMA.
        gt_total_predictives_bregman_dual_bma: Tensor to store ground truth Bregman
            total predictive uncertainties for the dual BMA.
        gt_total_predictives_bregman_bma: Tensor to store ground truth Bregman total
            predictive uncertainties for BMA.
        gt_soft_labels: Tensor to store ground truth soft labels.
        gt_hard_labels: Tensor to store ground truth hard labels.
        gt_hard_labels_original: Tensor to store original ground truth hard labels.
    """
    current_ind = 0

    for input, label in loader:
        indices = slice(current_ind, current_ind + input.shape[0])

        if not args.prefetcher:
            input, label = input.to(device), label.to(device)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        if is_soft_dataset:
            hard_label = label[:, -1]
            label = label[:, :-1]

        batch_size = input.shape[0]

        time_forward_start = time.perf_counter()

        if isinstance(model, SWAGWrapper | FastDeepEnsembleWrapper | DDUWrapper):
            inference_dict = model(input, amp_autocast)
        else:
            with amp_autocast():
                inference_dict = model(input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        time_forward_end = time.perf_counter()
        time_forward = time_forward_end - time_forward_start

        for key in list(inference_dict.keys()):
            inference_dict[key] = (
                inference_dict[key].detach().float().to(storage_device)
            )

        inference_dict = convert_inference_dict(
            model=model,
            inference_dict=inference_dict,
            time_forward=time_forward,
            args=args,
        )

        update_logit_based(
            inference_dict=inference_dict,
            indices=indices,
            batch_size=batch_size,
            log_dual_bmas=log_dual_bmas,
            log_bmas=log_bmas,
            gt_epistemics_bregman=gt_epistemics_bregman,
            time_forward_m=time_forward_m,
            expected_entropies=expected_entropies,
            expected_entropies_plus_expected_divergences=expected_entropies_plus_expected_divergences,
            one_minus_expected_max_probs=one_minus_expected_max_probs,
            entropies_of_bma=entropies_of_bma,
            entropies_of_dual_bma=entropies_of_dual_bma,
            one_minus_max_probs_of_bma=one_minus_max_probs_of_bma,
            one_minus_max_probs_of_dual_bma=one_minus_max_probs_of_dual_bma,
            jensen_shannon_divergences=jensen_shannon_divergences,
            dempster_shafer_values=dempster_shafer_values,
            expected_variances_of_probs=expected_variances_of_probs,
            expected_variances_of_logits=expected_variances_of_logits,
        )

        if isinstance(model, BaseLossPredictionWrapper):
            update_losspred(
                inference_dict=inference_dict,
                indices=indices,
                loss_values=loss_values,
            )
        elif isinstance(model, DDUWrapper):
            update_ddu(
                inference_dict=inference_dict,
                indices=indices,
                gmm_neg_log_densities=gmm_neg_log_densities,
            )
        elif isinstance(model, BaseCorrectnessPredictionWrapper):
            update_corrpred(
                inference_dict=inference_dict,
                indices=indices,
                error_probabilities=error_probabilities,
            )
        elif isinstance(model, DUQWrapper):
            update_duq(
                inference_dict=inference_dict,
                indices=indices,
                duq_values=duq_values,
            )
        elif isinstance(model, MahalanobisWrapper):
            update_mahalanobis(
                inference_dict=inference_dict,
                indices=indices,
                mahalanobis_values=mahalanobis_values,
            )
        elif isinstance(model, HetClassNNWrapper):
            update_hetclassnn(
                inference_dict=inference_dict,
                indices=indices,
                expected_variances_of_internal_probs=expected_variances_of_internal_probs,
                expected_variances_of_internal_logits=expected_variances_of_internal_logits,
            )

        # GT containers
        if is_soft_dataset:
            prob = label.float() / label.sum(dim=1, keepdim=True)  # Normalization
            prob = prob.to(storage_device)
            gt_aleatorics_bregman[indices] = entropy(prob)

        log_dual_bma = inference_dict["log_dual_bma"]
        log_bma = inference_dict["log_bma"]
        if is_soft_dataset:
            log_prob = prob.log()
            min_real = torch.finfo(log_prob.dtype).min
            log_prob = torch.clamp(log_prob, min=min_real)

            gt_biases_bregman_dual_bma[indices] = kl_divergence(log_prob, log_dual_bma)
            gt_biases_bregman_bma[indices] = kl_divergence(log_prob, log_bma)
            gt_predictives_bregman_dual_bma[indices] = (
                gt_aleatorics_bregman[indices] + gt_biases_bregman_dual_bma[indices]
            )
            gt_predictives_bregman_bma[indices] = (
                gt_aleatorics_bregman[indices] + gt_biases_bregman_bma[indices]
            )
            gt_total_predictives_bregman_dual_bma[indices] = (
                gt_aleatorics_bregman[indices]
                + gt_biases_bregman_dual_bma[indices]
                + gt_epistemics_bregman[indices]
            )
            gt_total_predictives_bregman_bma[indices] = (
                gt_aleatorics_bregman[indices]
                + gt_biases_bregman_bma[indices]
                + gt_epistemics_bregman[indices]
            )
            gt_soft_labels[indices] = prob
            gt_hard_labels_original[indices] = hard_label.to(storage_device)
            gt_hard_labels[indices] = prob.argmax(dim=1)
        else:
            gt_hard_labels_original[indices] = label.to(storage_device)
            gt_hard_labels[indices] = label.to(storage_device)
            gt_predictives_bregman_dual_bma[indices] = F.cross_entropy(
                log_dual_bma, label.to(storage_device)
            )
            gt_predictives_bregman_bma[indices] = F.cross_entropy(
                log_bma, label.to(storage_device)
            )
            gt_total_predictives_bregman_dual_bma[indices] = F.cross_entropy(
                log_dual_bma, label.to(storage_device)
            )
            gt_total_predictives_bregman_bma[indices] = F.cross_entropy(
                log_bma, label.to(storage_device)
            )

        current_ind += input.shape[0]


def forward_deep_ensemble_on_loader(
    model: DeepEnsembleWrapper,
    loader: DataLoader | PrefetchLoader,
    is_soft_dataset: bool,
    amp_autocast: Callable,
    device: torch.device,
    storage_device: torch.device,
    num_samples: int,
    args: argparse.Namespace,
    log_dual_bmas: Tensor,
    log_bmas: Tensor,
    gt_epistemics_bregman: Tensor,
    time_forward_m: AverageMeter,
    expected_entropies: Tensor,
    expected_entropies_plus_expected_divergences: Tensor,
    one_minus_expected_max_probs: Tensor,
    entropies_of_bma: Tensor,
    entropies_of_dual_bma: Tensor,
    one_minus_max_probs_of_bma: Tensor,
    one_minus_max_probs_of_dual_bma: Tensor,
    jensen_shannon_divergences: Tensor,
    dempster_shafer_values: Tensor,
    expected_variances_of_probs: Tensor,
    expected_variances_of_logits: Tensor,
    gt_aleatorics_bregman: Tensor,
    gt_biases_bregman_dual_bma: Tensor,
    gt_biases_bregman_bma: Tensor,
    gt_predictives_bregman_dual_bma: Tensor,
    gt_predictives_bregman_bma: Tensor,
    gt_total_predictives_bregman_dual_bma: Tensor,
    gt_total_predictives_bregman_bma: Tensor,
    gt_soft_labels: Tensor,
    gt_hard_labels: Tensor,
    gt_hard_labels_original: Tensor,
) -> None:
    """Performs a forward pass of a deep ensemble model on a data loader.

    This function processes the data through each model in the ensemble, aggregates
    the results, and updates various metrics and estimates.

    Args:
        model: The deep ensemble model to evaluate.
        loader: The data loader.
        is_soft_dataset: Whether the dataset uses soft labels.
        amp_autocast: Function for automatic mixed precision.
        device: The device to use for computation.
        storage_device: The device to use for storing results.
        num_samples: Number of samples.
        args: Additional arguments.
        log_dual_bmas: Tensor to store log dual BMA values.
        log_bmas: Tensor to store log BMA values.
        gt_epistemics_bregman: Tensor to store ground truth epistemic uncertainties.
        time_forward_m: AverageMeter to track forward pass time.
        expected_entropies: Tensor to store expected entropies.
        expected_entropies_plus_expected_divergences: Tensor to store sum of expected
            entropies and expected divergences.
        one_minus_expected_max_probs: Tensor to store 1 minus expected max
            probabilities.
        entropies_of_bma: Tensor to store entropies of BMA.
        entropies_of_dual_bma: Tensor to store entropies of the dual BMA.
        one_minus_max_probs_of_bma: Tensor to store 1 minus max probabilities of BMA.
        one_minus_max_probs_of_dual_bma: Tensor to store 1 minus max probabilities of
            the dual BMA.
        jensen_shannon_divergences: Tensor to store Jensen-Shannon divergences.
        dempster_shafer_values: Tensor to store Dempster-Shafer values.
        expected_variances_of_probs: Tensor to store expected variances of
            probabilities.
        expected_variances_of_logits: Tensor to store expected variances of logits.
        gt_aleatorics_bregman: Tensor to store ground truth aleatoric uncertainties.
        gt_biases_bregman_dual_bma: Tensor to store ground truth Bregman biases for
            the dual BMA.
        gt_biases_bregman_bma: Tensor to store ground truth Bregman biases for BMA.
        gt_predictives_bregman_dual_bma: Tensor to store ground truth Bregman predictive
            uncertainties for the dual BMA.
        gt_predictives_bregman_bma: Tensor to store ground truth Bregman predictive
            uncertainties for BMA.
        gt_total_predictives_bregman_dual_bma: Tensor to store ground truth Bregman
            total predictive uncertainties for the dual BMA.
        gt_total_predictives_bregman_bma: Tensor to store ground truth Bregman total
            predictive uncertainties for BMA.
        gt_soft_labels: Tensor to store ground truth soft labels.
        gt_hard_labels: Tensor to store ground truth hard labels.
        gt_hard_labels_original: Tensor to store original ground truth hard labels.
    """
    temp_logits = torch.empty(
        num_samples, model.num_models, model.num_classes, device=storage_device
    )
    time_forwards = torch.empty(len(loader), model.num_models, device=storage_device)

    for model_index in range(model.num_models):
        logger.info(f"Loading model {model_index + 1}/{model.num_models}.")

        model.load_model_with_index(model_index)
        current_ind = 0
        for i, (input, _) in enumerate(loader):
            batch_size = input.shape[0]
            indices = slice(current_ind, current_ind + batch_size)

            if not args.prefetcher:
                input = input.to(device)

            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            time_forward_start = time.perf_counter()
            with amp_autocast():
                inference_dict = model(input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            time_forward_end = time.perf_counter()
            time_forward = time_forward_end - time_forward_start

            temp_logits[indices, model_index, :] = inference_dict["logit"].to(
                storage_device
            )
            time_forwards[i, model_index] = time_forward

            current_ind += batch_size

    # Aggregate logits and forward times
    time_forwards_sum = time_forwards.sum(dim=-1)

    current_ind = 0
    for i, (_, label) in enumerate(loader):
        batch_size = label.shape[0]
        indices = slice(current_ind, current_ind + batch_size)

        inference_dict = {
            "logit": temp_logits[indices],
        }

        inference_dict = convert_inference_dict(
            model=model,
            inference_dict=inference_dict,
            time_forward=time_forwards_sum[i],
            args=args,
        )

        update_logit_based(
            inference_dict=inference_dict,
            indices=indices,
            batch_size=batch_size,
            log_dual_bmas=log_dual_bmas,
            log_bmas=log_bmas,
            gt_epistemics_bregman=gt_epistemics_bregman,
            time_forward_m=time_forward_m,
            expected_entropies=expected_entropies,
            expected_entropies_plus_expected_divergences=expected_entropies_plus_expected_divergences,
            one_minus_expected_max_probs=one_minus_expected_max_probs,
            entropies_of_bma=entropies_of_bma,
            entropies_of_dual_bma=entropies_of_dual_bma,
            one_minus_max_probs_of_bma=one_minus_max_probs_of_bma,
            one_minus_max_probs_of_dual_bma=one_minus_max_probs_of_dual_bma,
            jensen_shannon_divergences=jensen_shannon_divergences,
            dempster_shafer_values=dempster_shafer_values,
            expected_variances_of_probs=expected_variances_of_probs,
            expected_variances_of_logits=expected_variances_of_logits,
        )

        # GT containers
        if is_soft_dataset:
            hard_label = label[:, -1]
            label = label[:, :-1]

            prob = label.float() / label.sum(dim=1, keepdim=True)  # Normalization
            prob = prob.to(storage_device)
            gt_aleatorics_bregman[indices] = entropy(prob)

        log_dual_bma = inference_dict["log_dual_bma"]
        log_bma = inference_dict["log_bma"]
        if is_soft_dataset:
            log_prob = prob.log()
            min_real = torch.finfo(log_prob.dtype).min
            log_prob = torch.clamp(log_prob, min=min_real)
            gt_biases_bregman_dual_bma[indices] = kl_divergence(log_prob, log_dual_bma)
            gt_biases_bregman_bma[indices] = kl_divergence(log_prob, log_bma)
            gt_predictives_bregman_dual_bma[indices] = (
                gt_aleatorics_bregman[indices] + gt_biases_bregman_dual_bma[indices]
            )
            gt_predictives_bregman_bma[indices] = (
                gt_aleatorics_bregman[indices] + gt_biases_bregman_bma[indices]
            )
            gt_total_predictives_bregman_dual_bma[indices] = (
                gt_aleatorics_bregman[indices]
                + gt_biases_bregman_dual_bma[indices]
                + gt_epistemics_bregman[indices]
            )
            gt_total_predictives_bregman_bma[indices] = (
                gt_aleatorics_bregman[indices]
                + gt_biases_bregman_bma[indices]
                + gt_epistemics_bregman[indices]
            )
            gt_soft_labels[indices] = prob
            gt_hard_labels_original[indices] = hard_label.to(storage_device)
            gt_hard_labels[indices] = prob.argmax(dim=1)
        else:
            label = label.to(storage_device)
            gt_hard_labels_original[indices] = label
            gt_hard_labels[indices] = label

            gt_predictives_bregman_dual_bma[indices] = F.cross_entropy(
                log_dual_bma, label
            )
            gt_predictives_bregman_bma[indices] = F.cross_entropy(log_bma, label)
            gt_total_predictives_bregman_dual_bma[indices] = F.cross_entropy(
                log_dual_bma, label
            )
            gt_total_predictives_bregman_bma[indices] = F.cross_entropy(log_bma, label)

        current_ind += batch_size


def calc_correctnesses(
    log_probs: dict[str, Tensor], targets: dict[str, Tensor], is_soft: bool
) -> None:
    """Calculates correctness metrics for the model's predictions.

    Args:
        log_probs: Dictionary of log probabilities.
        targets: Dictionary of targets to update with correctness metrics.
        is_soft: Whether the dataset uses soft labels.
    """
    predicted_labels_dual_bma = log_probs["log_dual_bmas"].argmax(dim=1)

    targets["gt_hard_dual_bma_correctnesses_original"] = predicted_labels_dual_bma.eq(
        targets["gt_hard_labels_original"]
    ).int()
    targets["gt_hard_dual_bma_correctnesses"] = predicted_labels_dual_bma.eq(
        targets["gt_hard_labels"]
    ).int()

    _, predicted_labels_dual_bma_top5 = torch.topk(log_probs["log_dual_bmas"], 5, dim=1)
    expanded_gt_hard_labels_original = (
        targets["gt_hard_labels_original"]
        .unsqueeze(dim=1)
        .expand_as(predicted_labels_dual_bma_top5)
    )
    targets["gt_hard_dual_bma_correctnesses_original_top5"] = (
        predicted_labels_dual_bma_top5.eq(expanded_gt_hard_labels_original)
        .max(dim=1)[0]
        .int()
    )
    expanded_gt_hard_labels = (
        targets["gt_hard_labels"]
        .unsqueeze(dim=1)
        .expand_as(predicted_labels_dual_bma_top5)
    )
    targets["gt_hard_dual_bma_correctnesses_top5"] = (
        predicted_labels_dual_bma_top5.eq(expanded_gt_hard_labels).max(dim=1)[0].int()
    )

    predicted_labels_bma = log_probs["log_bmas"].argmax(dim=1)

    targets["gt_hard_bma_correctnesses_original"] = predicted_labels_bma.eq(
        targets["gt_hard_labels_original"]
    ).int()
    _, predicted_labels_bma_top5 = torch.topk(log_probs["log_bmas"], 5, dim=1)
    targets["gt_hard_bma_correctnesses_original_top5"] = (
        predicted_labels_bma_top5.eq(expanded_gt_hard_labels_original)
        .max(dim=1)[0]
        .int()
    )
    targets["gt_hard_bma_correctnesses"] = predicted_labels_bma.eq(
        targets["gt_hard_labels"]
    ).int()
    targets["gt_hard_bma_correctnesses_top5"] = (
        predicted_labels_bma_top5.eq(expanded_gt_hard_labels).max(dim=1)[0].int()
    )

    if is_soft:
        targets["gt_soft_dual_bma_correctnesses"] = (
            targets["gt_soft_labels"]
            .gather(dim=1, index=predicted_labels_dual_bma.unsqueeze(dim=1))
            .squeeze(dim=1)
        )

        indexed_gt_soft_labels_dual_bma = targets["gt_soft_labels"].gather(
            dim=1, index=predicted_labels_dual_bma_top5
        )
        targets["gt_soft_dual_bma_correctnesses_top5"] = (
            indexed_gt_soft_labels_dual_bma.max(dim=1)[0]
        )

        targets["gt_soft_bma_correctnesses"] = (
            targets["gt_soft_labels"]
            .gather(dim=1, index=predicted_labels_bma.unsqueeze(dim=1))
            .squeeze(dim=1)
        )

        indexed_gt_soft_labels_bma = targets["gt_soft_labels"].gather(
            dim=1, index=predicted_labels_bma_top5
        )
        targets["gt_soft_bma_correctnesses_top5"] = indexed_gt_soft_labels_bma.max(
            dim=1
        )[0]


def extract_averages(times: dict[str, AverageMeter]) -> None:
    """Extracts average values from AverageMeter objects.

    Args:
        times: Dictionary containing AverageMeter objects.
    """
    for key in list(times.keys()):
        times[key] = times[key].avg


def remove_faulty_indices(
    estimates: dict[str, Tensor],
    log_probs: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> None:
    """Removes entries with faulty indices from estimates, log_probs, and targets.

    Args:
        estimates: Dictionary of estimates.
        log_probs: Dictionary of log probabilities.
        targets: Dictionary of targets.
    """
    faulty_indices = targets["gt_aleatorics_bregman"].isnan()

    if faulty_indices.sum() > 0:
        for estimator_name in list(estimates.keys()):
            estimates[estimator_name] = estimates[estimator_name][~faulty_indices]

        for log_prob_name in list(log_probs.keys()):
            log_probs[log_prob_name] = log_probs[log_prob_name][~faulty_indices]

        for target_name in list(targets.keys()):
            targets[target_name] = targets[target_name][~faulty_indices]


def get_bundle(
    model: nn.Module,
    loader: DataLoader | PrefetchLoader,
    device: torch.device,
    storage_device: torch.device,
    amp_autocast: Callable,
    is_soft_dataset: bool,
    args: argparse.Namespace,
) -> tuple[
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, Tensor],
    dict[str, float],
]:
    """Processes the data loader and returns a bundle of evaluation results.

    Args:
        model: The model to evaluate.
        loader: The data loader.
        device: The device to use for computation.
        storage_device: The device to use for storing results.
        amp_autocast: Function for automatic mixed precision.
        is_soft_dataset: Whether the dataset uses soft labels.
        args: Additional arguments.

    Returns:
        A tuple containing dictionaries of estimates, log probabilities, targets, and
        times.
    """
    estimates = {}
    log_probs = {}
    targets = {}
    times = {}

    num_samples = len(loader.dataset)  # Total number of samples

    # Ground-truth containers
    gt_hard_labels = torch.empty(num_samples, dtype=torch.long, device=storage_device)
    gt_hard_labels_original = torch.empty(
        num_samples, dtype=torch.long, device=storage_device
    )
    targets["gt_hard_labels"] = gt_hard_labels
    targets["gt_hard_labels_original"] = gt_hard_labels_original

    if is_soft_dataset:
        gt_soft_labels = torch.empty(
            num_samples, model.num_classes, device=storage_device
        )
        targets["gt_soft_labels"] = gt_soft_labels

        # Bregman Aleatoric uncertainty
        gt_aleatorics_bregman = torch.empty(num_samples, device=storage_device)
        targets["gt_aleatorics_bregman"] = gt_aleatorics_bregman
        # Also interested in how well the GT solves the practical tasks
        estimates["gt_aleatorics_bregman"] = gt_aleatorics_bregman

        # Bregman Bias
        gt_biases_bregman_dual_bma = torch.empty(num_samples, device=storage_device)
        targets["gt_biases_bregman_dual_bma"] = gt_biases_bregman_dual_bma
        estimates["gt_biases_bregman_dual_bma"] = gt_biases_bregman_dual_bma

        gt_biases_bregman_bma = torch.empty(num_samples, device=storage_device)
        targets["gt_biases_bregman_bma"] = gt_biases_bregman_bma
        estimates["gt_biases_bregman_bma"] = gt_biases_bregman_bma

    # Estimate containers
    # Predictive uncertainty (Bregman)
    gt_predictives_bregman_dual_bma = torch.empty(num_samples, device=storage_device)
    targets["gt_predictives_bregman_dual_bma"] = gt_predictives_bregman_dual_bma
    estimates["gt_predictives_bregman_dual_bma"] = gt_predictives_bregman_dual_bma

    gt_total_predictives_bregman_dual_bma = torch.empty(
        num_samples, device=storage_device
    )
    targets["gt_total_predictives_bregman_dual_bma"] = (
        gt_total_predictives_bregman_dual_bma
    )
    estimates["gt_total_predictives_bregman_dual_bma"] = (
        gt_total_predictives_bregman_dual_bma
    )

    gt_predictives_bregman_bma = torch.empty(num_samples, device=storage_device)
    targets["gt_predictives_bregman_bma"] = gt_predictives_bregman_bma
    estimates["gt_predictives_bregman_bma"] = gt_predictives_bregman_bma

    gt_total_predictives_bregman_bma = torch.empty(num_samples, device=storage_device)
    targets["gt_total_predictives_bregman_bma"] = gt_total_predictives_bregman_bma
    estimates["gt_total_predictives_bregman_bma"] = gt_total_predictives_bregman_bma

    # Epistemic uncertainty (Bregman)
    gt_epistemics_bregman = torch.empty(num_samples, device=storage_device)
    targets["gt_epistemics_bregman"] = gt_epistemics_bregman

    # Time
    time_forward_m = AverageMeter()
    times["time_forward_m"] = time_forward_m

    log_dual_bmas = torch.empty(num_samples, model.num_classes, device=storage_device)
    log_probs["log_dual_bmas"] = log_dual_bmas

    log_bmas = torch.empty(num_samples, model.num_classes, device=storage_device)
    log_probs["log_bmas"] = log_bmas

    # Aleatoric Uncertainty
    expected_entropies = torch.empty(num_samples, device=storage_device)
    estimates["expected_entropies"] = expected_entropies
    one_minus_expected_max_probs = torch.empty(num_samples, device=storage_device)
    estimates["one_minus_expected_max_probs"] = one_minus_expected_max_probs

    # Predictive Uncertainty
    entropies_of_bma = torch.empty(num_samples, device=storage_device)
    estimates["entropies_of_bma"] = entropies_of_bma
    entropies_of_dual_bma = torch.empty(num_samples, device=storage_device)
    estimates["entropies_of_dual_bma"] = entropies_of_dual_bma
    one_minus_max_probs_of_bma = torch.empty(num_samples, device=storage_device)
    estimates["one_minus_max_probs_of_bma"] = one_minus_max_probs_of_bma
    one_minus_max_probs_of_dual_bma = torch.empty(num_samples, device=storage_device)
    estimates["one_minus_max_probs_of_dual_bma"] = one_minus_max_probs_of_dual_bma
    expected_entropies_plus_expected_divergences = torch.empty(
        num_samples, device=storage_device
    )
    estimates["expected_entropies_plus_expected_divergences"] = (
        expected_entropies_plus_expected_divergences
    )

    # Epistemic Uncertainty
    dempster_shafer_values = torch.empty(num_samples, device=storage_device)
    estimates["dempster_shafer_values"] = dempster_shafer_values
    estimates["expected_divergences"] = gt_epistemics_bregman
    jensen_shannon_divergences = torch.empty(num_samples, device=storage_device)
    estimates["jensen_shannon_divergences"] = jensen_shannon_divergences

    expected_variances_of_probs = torch.empty(num_samples, device=storage_device)
    estimates["expected_variances_of_probs"] = expected_variances_of_probs
    expected_variances_of_logits = torch.empty(num_samples, device=storage_device)
    estimates["expected_variances_of_logits"] = expected_variances_of_logits

    # Set up defaults that are overridden for specialized methods
    loss_values = None
    gmm_neg_log_densities = None
    error_probabilities = None
    duq_values = None
    mahalanobis_values = None
    expected_variances_of_internal_probs = None
    expected_variances_of_internal_logits = None

    if isinstance(model, BaseLossPredictionWrapper):
        # Predictive Uncertainty
        loss_values = torch.empty(num_samples, device=storage_device)
        estimates["loss_values"] = loss_values
    elif isinstance(model, DDUWrapper):
        gmm_neg_log_densities = torch.empty(num_samples, device=storage_device)
        estimates["gmm_neg_log_densities"] = gmm_neg_log_densities
    elif isinstance(model, BaseCorrectnessPredictionWrapper):
        # Predictive Uncertainty
        error_probabilities = torch.empty(num_samples, device=storage_device)
        estimates["error_probabilities"] = error_probabilities
    elif isinstance(model, DUQWrapper):
        # Epistemic Uncertainty
        duq_values = torch.empty(num_samples, device=storage_device)
        estimates["duq_values"] = duq_values
    elif isinstance(model, MahalanobisWrapper):
        # Epistemic Uncertainty
        mahalanobis_values = torch.empty(num_samples, device=storage_device)
        estimates["mahalanobis_values"] = mahalanobis_values
    elif isinstance(model, HetClassNNWrapper):
        expected_variances_of_internal_probs = torch.empty(
            num_samples, device=storage_device
        )
        estimates["expected_variances_of_internal_probs"] = (
            expected_variances_of_internal_probs
        )
        expected_variances_of_internal_logits = torch.empty(
            num_samples, device=storage_device
        )
        estimates["expected_variances_of_internal_logits"] = (
            expected_variances_of_internal_logits
        )

    if isinstance(model, DeepEnsembleWrapper):
        forward_deep_ensemble_on_loader(
            model=model,
            loader=loader,
            is_soft_dataset=is_soft_dataset,
            amp_autocast=amp_autocast,
            device=device,
            storage_device=storage_device,
            num_samples=num_samples,
            args=args,
            log_dual_bmas=log_dual_bmas,
            log_bmas=log_bmas,
            gt_epistemics_bregman=gt_epistemics_bregman,
            time_forward_m=time_forward_m,
            expected_entropies=expected_entropies,
            expected_entropies_plus_expected_divergences=expected_entropies_plus_expected_divergences,
            one_minus_expected_max_probs=one_minus_expected_max_probs,
            entropies_of_bma=entropies_of_bma,
            entropies_of_dual_bma=entropies_of_dual_bma,
            one_minus_max_probs_of_bma=one_minus_max_probs_of_bma,
            one_minus_max_probs_of_dual_bma=one_minus_max_probs_of_dual_bma,
            jensen_shannon_divergences=jensen_shannon_divergences,
            dempster_shafer_values=dempster_shafer_values,
            expected_variances_of_probs=expected_variances_of_probs,
            expected_variances_of_logits=expected_variances_of_logits,
            gt_aleatorics_bregman=gt_aleatorics_bregman,
            gt_biases_bregman_dual_bma=gt_biases_bregman_dual_bma,
            gt_biases_bregman_bma=gt_biases_bregman_bma,
            gt_predictives_bregman_dual_bma=gt_predictives_bregman_dual_bma,
            gt_predictives_bregman_bma=gt_predictives_bregman_bma,
            gt_total_predictives_bregman_dual_bma=gt_total_predictives_bregman_dual_bma,
            gt_total_predictives_bregman_bma=gt_total_predictives_bregman_bma,
            gt_soft_labels=gt_soft_labels,
            gt_hard_labels=gt_hard_labels,
            gt_hard_labels_original=gt_hard_labels_original,
        )
    else:
        forward_general_model_on_loader(
            model=model,
            loader=loader,
            is_soft_dataset=is_soft_dataset,
            amp_autocast=amp_autocast,
            device=device,
            storage_device=storage_device,
            args=args,
            log_dual_bmas=log_dual_bmas,
            log_bmas=log_bmas,
            gt_epistemics_bregman=gt_epistemics_bregman,
            time_forward_m=time_forward_m,
            expected_entropies=expected_entropies,
            expected_entropies_plus_expected_divergences=expected_entropies_plus_expected_divergences,
            one_minus_expected_max_probs=one_minus_expected_max_probs,
            entropies_of_bma=entropies_of_bma,
            entropies_of_dual_bma=entropies_of_dual_bma,
            one_minus_max_probs_of_bma=one_minus_max_probs_of_bma,
            one_minus_max_probs_of_dual_bma=one_minus_max_probs_of_dual_bma,
            jensen_shannon_divergences=jensen_shannon_divergences,
            dempster_shafer_values=dempster_shafer_values,
            expected_variances_of_probs=expected_variances_of_probs,
            expected_variances_of_logits=expected_variances_of_logits,
            loss_values=loss_values,
            gmm_neg_log_densities=gmm_neg_log_densities,
            error_probabilities=error_probabilities,
            duq_values=duq_values,
            mahalanobis_values=mahalanobis_values,
            expected_variances_of_internal_probs=expected_variances_of_internal_probs,
            expected_variances_of_internal_logits=expected_variances_of_internal_logits,
            gt_aleatorics_bregman=gt_aleatorics_bregman,
            gt_biases_bregman_dual_bma=gt_biases_bregman_dual_bma,
            gt_biases_bregman_bma=gt_biases_bregman_bma,
            gt_predictives_bregman_dual_bma=gt_predictives_bregman_dual_bma,
            gt_predictives_bregman_bma=gt_predictives_bregman_bma,
            gt_total_predictives_bregman_dual_bma=gt_total_predictives_bregman_dual_bma,
            gt_total_predictives_bregman_bma=gt_total_predictives_bregman_bma,
            gt_soft_labels=gt_soft_labels,
            gt_hard_labels=gt_hard_labels,
            gt_hard_labels_original=gt_hard_labels_original,
        )

    # Calculate correctness indicators
    calc_correctnesses(log_probs, targets, is_soft_dataset)

    # Extract averages from the AverageMeters
    extract_averages(times)

    if is_soft_dataset:
        remove_faulty_indices(estimates, log_probs, targets)

    return estimates, log_probs, targets, times


def convert_inference_dict(
    model: nn.Module,
    inference_dict: dict[str, Tensor],
    time_forward: float,
    args: argparse.Namespace,
) -> dict[str, Tensor]:
    """Converts the raw inference dict into a standardized format.

    Args:
        model: The model used for inference.
        inference_dict: Dictionary containing raw inference results.
        time_forward: Time taken for the forward pass.
        args: Additional arguments.

    Returns:
        A dictionary with converted and standardized inference results.
    """
    converted_inference_dict = {}

    converted_inference_dict["time_forward"] = time_forward

    if isinstance(model, DirichletWrapper):
        convert_inference_dict_dirichlet(
            converted_inference_dict=converted_inference_dict,
            inference_dict=inference_dict,
            args=args,
        )
    else:
        convert_inference_dict_general(
            converted_inference_dict=converted_inference_dict,
            inference_dict=inference_dict,
        )

    if isinstance(model, BaseLossPredictionWrapper):
        converted_inference_dict["loss_value"] = inference_dict["loss_value"]
    elif isinstance(model, DDUWrapper):
        converted_inference_dict["gmm_neg_log_density"] = inference_dict[
            "gmm_neg_log_density"
        ]
    elif isinstance(model, BaseCorrectnessPredictionWrapper):
        converted_inference_dict["error_probability"] = inference_dict[
            "error_probability"
        ]
    elif isinstance(model, DUQWrapper):
        converted_inference_dict["duq_value"] = inference_dict["duq_value"]
    elif isinstance(model, MahalanobisWrapper):
        converted_inference_dict["mahalanobis_value"] = inference_dict[
            "mahalanobis_value"
        ]
    elif isinstance(model, HetClassNNWrapper):
        internal_logits = inference_dict["internal_logit"]
        log_internal_probs = F.log_softmax(internal_logits, dim=-1)  # [B, S, C]

        internal_probs = log_internal_probs.exp()  # [B, S, C]

        if internal_logits.shape[1] > 1:
            converted_inference_dict["expected_variance_of_internal_logits"] = (
                torch.var(internal_logits, dim=1).mean(dim=-1)
            )  # [B]
        else:
            converted_inference_dict["expected_variance_of_internal_logits"] = 0.0

        if internal_probs.shape[1] > 1:
            converted_inference_dict["expected_variance_of_internal_probs"] = torch.var(
                internal_probs, dim=1
            ).mean(dim=-1)  # [B]
        else:
            converted_inference_dict["expected_variance_of_internal_probs"] = 0.0

    return converted_inference_dict


def convert_inference_dict_dirichlet(
    converted_inference_dict: dict[str, Tensor],
    inference_dict: dict[str, Tensor],
    args: argparse.Namespace,
) -> None:
    """Converts the raw inference dict into a standardized format for Dirichlets.

    Args:
        converted_inference_dict: Dictionary to collect the converted values into.
        inference_dict: Dictionary containing raw inference results.
        args: Additional arguments.
    """
    alphas = inference_dict["alpha"]  # [B, C]
    min_real = torch.finfo(alphas.dtype).min
    log_probs = (
        torch.distributions.Dirichlet(alphas)
        .sample((args.num_mc_samples,))
        .permute(1, 0, 2)
        .log()
        .clamp(min=min_real)
    )  # [B, S, C]

    sum_alphas = alphas.sum(dim=1)  # [B]
    mean_alphas = alphas.div(sum_alphas.unsqueeze(1))  # [B, C]

    log_bma = mean_alphas.log().clamp(min=min_real)
    converted_inference_dict["log_bma"] = log_bma

    log_dual_bma = F.log_softmax(log_probs.mean(dim=1), dim=-1)  # [B, C]
    converted_inference_dict["log_dual_bma"] = log_dual_bma

    digamma_term = torch.digamma(alphas + 1) - torch.digamma(sum_alphas + 1).unsqueeze(
        1
    )  # [B, C]
    expected_entropy = -mean_alphas.mul(digamma_term).sum(dim=1)  # [B]
    converted_inference_dict["expected_entropy"] = expected_entropy

    expected_divergence = kl_divergence(log_dual_bma, log_probs.permute(1, 0, 2)).mean(
        dim=0
    )
    converted_inference_dict["expected_divergence"] = expected_divergence

    probs = log_probs.exp()  # [B, S, C]

    if probs.shape[1] > 1:
        converted_inference_dict["expected_variance_of_probs"] = torch.var(
            probs, dim=1
        ).mean(dim=-1)  # [B]
    else:
        converted_inference_dict["expected_variance_of_probs"] = 0.0

    converted_inference_dict["expected_variance_of_logits"] = 0.0

    expected_max_prob = probs.max(dim=-1)[0].mean(dim=1)
    converted_inference_dict["expected_max_prob"] = expected_max_prob

    entropy_of_bma = entropy(mean_alphas)
    converted_inference_dict["entropy_of_bma"] = entropy_of_bma

    dual_bma = log_dual_bma.exp()

    entropy_of_dual_bma = entropy(dual_bma)
    converted_inference_dict["entropy_of_dual_bma"] = entropy_of_dual_bma

    max_prob_of_bma = mean_alphas.max(dim=-1)[0]
    converted_inference_dict["max_prob_of_bma"] = max_prob_of_bma

    max_prob_of_dual_bma = dual_bma.max(dim=-1)[0]
    converted_inference_dict["max_prob_of_dual_bma"] = max_prob_of_dual_bma

    jensen_shannon_divergence = entropy_of_bma - expected_entropy
    converted_inference_dict["jensen_shannon_divergence"] = jensen_shannon_divergence

    num_classes = alphas.shape[1]
    dempster_shafer_value = num_classes / sum_alphas  # [B]
    converted_inference_dict["dempster_shafer_value"] = dempster_shafer_value


def convert_inference_dict_general(
    converted_inference_dict: dict[str, Tensor],
    inference_dict: dict[str, Tensor],
) -> None:
    """Converts the raw inference dict into a standardized format for general models.

    Args:
        converted_inference_dict: Dictionary to collect the converted values into.
        inference_dict: Dictionary containing raw inference results.
    """
    logits = inference_dict["logit"]
    min_real = torch.finfo(logits.dtype).min
    if logits.dim() == 2:  # [B, C]
        logits = logits.unsqueeze(dim=1)  # [B, 1, C]
    log_probs = F.log_softmax(logits, dim=-1)  # [B, S, C]

    probs = log_probs.exp()  # [B, S, C]

    if logits.shape[1] > 1:
        converted_inference_dict["expected_variance_of_logits"] = torch.var(
            logits, dim=1
        ).mean(dim=-1)  # [B]
    else:
        converted_inference_dict["expected_variance_of_logits"] = 0.0

    if probs.shape[1] > 1:
        converted_inference_dict["expected_variance_of_probs"] = torch.var(
            probs, dim=1
        ).mean(dim=-1)  # [B]
    else:
        converted_inference_dict["expected_variance_of_probs"] = 0.0

    log_dual_bma = F.log_softmax(log_probs.mean(dim=1), dim=-1)  # [B, C]

    dual_bma = log_dual_bma.exp()
    converted_inference_dict["log_dual_bma"] = log_dual_bma

    bma = probs.mean(dim=1)  # [B, C]

    log_bma = bma.log()  # [B, C]
    log_bma = torch.clamp(log_bma, min=min_real)
    converted_inference_dict["log_bma"] = log_bma

    expected_entropy = entropy(probs).mean(dim=-1)
    converted_inference_dict["expected_entropy"] = expected_entropy

    expected_divergence = kl_divergence(log_dual_bma, log_probs.permute(1, 0, 2)).mean(
        dim=0
    )
    converted_inference_dict["expected_divergence"] = expected_divergence

    expected_max_prob = probs.max(dim=-1)[0].mean(dim=1)
    converted_inference_dict["expected_max_prob"] = expected_max_prob

    entropy_of_bma = entropy(bma)

    converted_inference_dict["entropy_of_bma"] = entropy_of_bma

    entropy_of_dual_bma = entropy(dual_bma)
    converted_inference_dict["entropy_of_dual_bma"] = entropy_of_dual_bma

    max_prob_of_bma = bma.max(dim=-1)[0]
    converted_inference_dict["max_prob_of_bma"] = max_prob_of_bma

    max_prob_of_dual_bma = dual_bma.max(dim=-1)[0]
    converted_inference_dict["max_prob_of_dual_bma"] = max_prob_of_dual_bma

    jensen_shannon_divergence = entropy_of_bma - expected_entropy
    converted_inference_dict["jensen_shannon_divergence"] = jensen_shannon_divergence

    dempster_shafer_value = dempster_shafer_metric(logits.mean(dim=1))
    converted_inference_dict["dempster_shafer_value"] = dempster_shafer_value


def update_logit_based(
    inference_dict: dict[str, Tensor],
    indices: slice,
    batch_size: int,
    log_dual_bmas: Tensor,
    log_bmas: Tensor,
    gt_epistemics_bregman: Tensor,
    time_forward_m: AverageMeter,
    expected_entropies: Tensor,
    expected_entropies_plus_expected_divergences: Tensor,
    one_minus_expected_max_probs: Tensor,
    entropies_of_bma: Tensor,
    entropies_of_dual_bma: Tensor,
    one_minus_max_probs_of_bma: Tensor,
    one_minus_max_probs_of_dual_bma: Tensor,
    jensen_shannon_divergences: Tensor,
    dempster_shafer_values: Tensor,
    expected_variances_of_probs: Tensor,
    expected_variances_of_logits: Tensor,
) -> None:
    """Updates logit-based metrics and estimates.

    Args:
        inference_dict: Dictionary containing inference results.
        indices: Slice object for indexing tensors.
        batch_size: Size of the current batch.
        log_dual_bmas: Tensor to store log dual BMA values.
        log_bmas: Tensor to store log BMA values.
        gt_epistemics_bregman: Tensor to store ground truth epistemic uncertainties.
        time_forward_m: AverageMeter to track forward pass time.
        expected_entropies: Tensor to store expected entropies.
        expected_entropies_plus_expected_divergences: Tensor to store sum of expected
            entropies and expected divergences.
        one_minus_expected_max_probs: Tensor to store 1 minus expected max
            probabilities.
        entropies_of_bma: Tensor to store entropies of BMA.
        entropies_of_dual_bma: Tensor to store entropies of the dual BMA.
        one_minus_max_probs_of_bma: Tensor to store 1 minus max probabilities of BMA.
        one_minus_max_probs_of_dual_bma: Tensor to store 1 minus max probabilities of
            the dual BMA.
        jensen_shannon_divergences: Tensor to store Jensen-Shannon divergences.
        dempster_shafer_values: Tensor to store Dempster-Shafer values.
        expected_variances_of_probs: Tensor to store expected variances of
            probabilities.
        expected_variances_of_logits: Tensor to store expected variances of logits.
    """
    log_dual_bmas[indices] = inference_dict["log_dual_bma"]
    log_bmas[indices] = inference_dict["log_bma"]
    gt_epistemics_bregman[indices] = inference_dict["expected_divergence"]

    time_forward_m.update(inference_dict["time_forward"], batch_size)

    expected_entropies[indices] = inference_dict["expected_entropy"]
    expected_entropies_plus_expected_divergences[indices] = (
        expected_entropies[indices] + gt_epistemics_bregman[indices]
    )
    one_minus_expected_max_probs[indices] = 1 - inference_dict["expected_max_prob"]
    entropies_of_bma[indices] = inference_dict["entropy_of_bma"]
    entropies_of_dual_bma[indices] = inference_dict["entropy_of_dual_bma"]
    one_minus_max_probs_of_bma[indices] = 1 - inference_dict["max_prob_of_bma"]
    one_minus_max_probs_of_dual_bma[indices] = (
        1 - inference_dict["max_prob_of_dual_bma"]
    )
    jensen_shannon_divergences[indices] = inference_dict["jensen_shannon_divergence"]
    dempster_shafer_values[indices] = inference_dict["dempster_shafer_value"]
    expected_variances_of_probs[indices] = inference_dict["expected_variance_of_probs"]
    expected_variances_of_logits[indices] = inference_dict[
        "expected_variance_of_logits"
    ]


def update_losspred(
    inference_dict: dict[str, Tensor], indices: slice, loss_values: Tensor
) -> None:
    """Updates loss prediction values.

    Args:
        inference_dict: Dictionary containing inference results.
        indices: Slice object for indexing tensors.
        loss_values: Tensor to store loss prediction values.
    """
    loss_values[indices] = inference_dict["loss_value"]


def update_ddu(
    inference_dict: dict[str, Tensor],
    indices: slice,
    gmm_neg_log_densities: Tensor,
) -> None:
    """Updates DDU-specific metrics.

    Args:
        inference_dict: Dictionary containing inference results.
        indices: Slice object for indexing tensors.
        gmm_neg_log_densities: Tensor to store GMM negative log densities.
    """
    gmm_neg_log_densities[indices] = inference_dict["gmm_neg_log_density"]


def update_corrpred(
    inference_dict: dict[str, Tensor],
    indices: slice,
    error_probabilities: Tensor,
) -> None:
    """Updates correctness prediction values.

    Args:
        inference_dict: Dictionary containing inference results.
        indices: Slice object for indexing tensors.
        error_probabilities: Tensor to store error probabilities.
    """
    error_probabilities[indices] = inference_dict["error_probability"]


def update_duq(
    inference_dict: dict[str, Tensor], indices: slice, duq_values: Tensor
) -> None:
    """Updates DUQ-specific values.

    Args:
        inference_dict: Dictionary containing inference results.
        indices: Slice object for indexing tensors.
        duq_values: Tensor to store DUQ values.
    """
    duq_values[indices] = inference_dict["duq_value"]


def update_mahalanobis(
    inference_dict: dict[str, Tensor],
    indices: slice,
    mahalanobis_values: Tensor,
) -> None:
    """Updates Mahalanobis distance values.

    Args:
        inference_dict: Dictionary containing inference results.
        indices: Slice object for indexing tensors.
        mahalanobis_values: Tensor to store Mahalanobis distance values.
    """
    mahalanobis_values[indices] = inference_dict["mahalanobis_value"]


def update_hetclassnn(
    inference_dict: dict[str, Tensor],
    indices: slice,
    expected_variances_of_internal_probs: Tensor,
    expected_variances_of_internal_logits: Tensor,
) -> None:
    """Updates HetClassNN-specific values.

    Args:
        inference_dict: Dictionary containing inference results.
        indices: Slice object for indexing tensors.
        expected_variances_of_internal_probs: Tensor to store expected variances of
            internal probabilities.
        expected_variances_of_internal_logits: Tensor to store expected variances of
            internal logits.
    """
    expected_variances_of_internal_probs[indices] = inference_dict[
        "expected_variance_of_internal_probs"
    ]
    expected_variances_of_internal_logits[indices] = inference_dict[
        "expected_variance_of_internal_logits"
    ]
