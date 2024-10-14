"""Copyright 2020 Ross Wightman and 2024 Bálint Mucsányi."""

import logging
import os
import time
from numbers import Number
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn.parallel

from untangle.utils import (
    AverageMeter,
    area_under_lift_curve,
    area_under_risk_coverage_curve,
    auroc,
    binary_brier,
    binary_log_probability,
    calibration_error,
    coverage_for_accuracy,
    entropy,
    excess_area_under_risk_coverage_curve,
    get_activation,
    get_dirichlet,
    get_predictive,
    multiclass_brier,
    multiclass_log_probability,
    pearsonr,
    relative_area_under_lift_curve,
    spearmanr,
)
from untangle.wrappers import (
    CovariancePushforwardLaplaceWrapper,
    DeepEnsembleWrapper,
    EDLWrapper,
    HETWrapper,
    LinearizedSWAGWrapper,
    PostNetWrapper,
    SNGPWrapper,
)

logger = logging.getLogger(__name__)


def evaluate_bulk(
    model,
    loaders,
    device,
    storage_device,
    amp_autocast,
    key_prefix,
    is_upstream_dataset,
    is_soft_dataset,
    args,
):
    metrics = {}

    for name, loader_subset in loaders.items():
        metrics[name] = {}
        for ood_transform_type, loader in loader_subset.items():
            logger.info(f"Evaluating {name} - {ood_transform_type}...")
            time_eval_start = time.perf_counter()

            metrics[name][ood_transform_type] = evaluate(
                model=model,
                loader=loader,
                device=device,
                storage_device=storage_device,
                amp_autocast=amp_autocast,
                key_prefix="",
                is_upstream_dataset=is_upstream_dataset,
                is_soft_dataset=is_soft_dataset,
                args=args,
            )

            time_eval_end = time.perf_counter()
            time_eval = time_eval_end - time_eval_start

            logger.info(
                f"Finished evaluating {name} - {ood_transform_type}. "
                f"Took {time_eval:.2f} seconds."
            )
        add_average(metrics[name])

    # Summarize results
    flattened_metrics = flatten(results=metrics, key_prefix=key_prefix)

    # Remove tmp file
    upstream_dict_path = Path(f"data/upstream_dict_{os.environ.get('SLURM_JOBID')}.pt")
    upstream_dict_path.unlink()

    return flattened_metrics


def add_average(results):
    # Summarize results
    avg_results = {}
    first_loader_results = results[next(iter(results.keys()))]
    for key in first_loader_results:
        if isinstance(first_loader_results[key], Number):
            result_vector = torch.tensor([
                loader_result[key] for _, loader_result in results.items()
            ])
            avg_results[key] = result_vector.mean().item()
    results["avg"] = avg_results


def flatten(results, key_prefix):
    # Flatten output
    flattened_results = {}
    for name, results_subset in results.items():
        for ood_transform_type, results_subsubset in results_subset.items():
            for key, value in results_subsubset.items():
                flattened_results[f"{key_prefix}_{name}_{ood_transform_type}_{key}"] = (
                    value
                )

    return flattened_results


def save_upstream_dict(
    estimates, targets, log_probs, data_dir, is_soft_dataset, storage_device, args
):
    # Save ingredients to disk
    max_num_indices = len(targets["gt_hard_labels"])
    num_indices = min(max_num_indices, args.max_num_id_ood_eval_samples // 2)
    path_indices = data_dir / f"{num_indices}_indices_out_of_{max_num_indices}.pt"

    if path_indices.exists():
        indices = torch.load(path_indices, weights_only=True)
    else:
        indices = torch.randperm(max_num_indices, device=storage_device)[:num_indices]
        torch.save(indices, path_indices)

    upstream_dict = {
        "upstream_estimates": filter_entries(estimates, indices),
        "upstream_targets": filter_entries(targets, indices),
        "upstream_log_probs": filter_entries(log_probs, indices),
        "is_soft_upstream_dataset": is_soft_dataset,
    }

    torch.save(
        upstream_dict, data_dir / f"upstream_dict_{os.environ.get('SLURM_JOBID')}.pt"
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    storage_device,
    amp_autocast,
    key_prefix,
    is_upstream_dataset,
    is_soft_dataset,
    args,
):
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

    metrics = evaluate_on_tasks(
        estimates=estimates,
        log_probs=log_probs,
        targets=targets,
        metrics=metrics,
        is_soft_dataset=is_soft_dataset,
        args=args,
    )

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if is_upstream_dataset:
        save_upstream_dict(
            estimates=estimates,
            targets=targets,
            log_probs=log_probs,
            data_dir=data_dir,
            is_soft_dataset=is_soft_dataset,
            storage_device=storage_device,
            args=args,
        )

    if not is_upstream_dataset:
        upstream_dict = torch.load(
            data_dir / f"upstream_dict_{os.environ.get('SLURM_JOBID')}.pt",
            weights_only=True,
        )
        upstream_estimates = upstream_dict["upstream_estimates"]
        upstream_log_probs = upstream_dict["upstream_log_probs"]
        upstream_targets = upstream_dict["upstream_targets"]
        is_soft_upstream_dataset = upstream_dict["is_soft_upstream_dataset"]

        # Make both upstream and downstream tensors the same size to get a 50/50
        # split
        num_upstream_indices = len(upstream_targets["gt_hard_labels"])
        max_num_downstream_indices = len(targets["gt_hard_labels"])
        num_indices_to_keep = min(num_upstream_indices, max_num_downstream_indices)

        # For upstream, we can just use [:num_samples_keep] in the following,
        # because it's already shuffled. For downstream, let's use random indices
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

        # Mix ingredients
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

            mixed_targets["gt_soft_bma_correctnesses"] = torch.cat([
                upstream_targets["gt_soft_bma_correctnesses"],
                downstream_targets["gt_hard_bma_correctnesses"],
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

            mixed_targets["gt_soft_bma_correctnesses"] = torch.cat([
                upstream_targets["gt_hard_bma_correctnesses"],
                downstream_targets["gt_soft_bma_correctnesses"],
            ])
        elif is_soft_upstream_dataset and is_soft_dataset:
            mixed_targets["gt_soft_labels"] = torch.cat([
                upstream_targets["gt_soft_labels"],
                downstream_targets["gt_soft_labels"],
            ])

        metrics = evaluate_on_tasks(
            estimates=mixed_estimates,
            log_probs=mixed_log_probs,
            targets=mixed_targets,
            metrics=metrics,
            is_soft_dataset=is_soft_dataset,
            args=args,
            is_soft_upstream_dataset=is_soft_upstream_dataset,
        )

    if key_prefix:
        for metric_name in list(metrics.keys()):
            metrics[f"{key_prefix}_{metric_name}"] = metrics.pop(metric_name)

    return metrics


def filter_entries(estimates, indices):
    filtered_estimates = estimates.copy()

    for estimator_name, estimate in filtered_estimates.items():
        filtered_estimates[estimator_name] = estimate[indices]

    return filtered_estimates


def truncate_entries(estimates, num_indices_to_keep):
    truncated_estimates = estimates.copy()

    for estimator_name, estimate in truncated_estimates.items():
        truncated_estimates[estimator_name] = estimate[:num_indices_to_keep]

    return truncated_estimates


def concatenate_values(upstream_dict, downstream_dict, keys_to_exclude=None):
    if keys_to_exclude is None:
        keys_to_exclude = []

    common_keys = upstream_dict.keys() & downstream_dict.keys()
    result = {
        key: torch.cat([upstream_dict[key], downstream_dict[key]], dim=0)
        for key in common_keys
        if key not in keys_to_exclude
    }

    return result


def evaluate_on_tasks(
    estimates,
    log_probs,
    targets,
    metrics,
    is_soft_dataset,
    args,
    is_soft_upstream_dataset=None,
):
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

    is_mixed_eval = is_soft_upstream_dataset is not None
    if is_mixed_eval:
        metrics |= evaluate_on_ood_detection(
            estimates=estimates,
            targets=targets,
            args=args,
        )

    metrics |= evaluate_on_proper_scoring_and_calibration(
        estimates=estimates,
        log_probs=log_probs,
        targets=targets,
        is_soft_dataset=is_soft_dataset,
        args=args,
        is_soft_upstream_dataset=is_soft_upstream_dataset,
    )
    metrics |= evaluate_on_correlation_of_decomposition(
        estimates=estimates,
        is_soft_dataset=is_soft_dataset,
        args=args,
        is_soft_upstream_dataset=is_soft_upstream_dataset,
    )

    return metrics


def evaluate_on_correctness_prediction(
    estimates,
    targets,
    is_soft_dataset,
    args,
    is_soft_upstream_dataset,
):
    is_mixed_eval = is_soft_upstream_dataset is not None

    # For correctness prediction, one of the datasets being soft is enough
    if is_mixed_eval:
        is_soft_dataset = is_soft_dataset or is_soft_upstream_dataset

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed_eval else ""

    prefixes = []
    if "mc_10_entropies_of_bma" in estimates:
        for i in [10, 100, 1000]:
            prefixes.append(f"mc_{i}")  # noqa: PERF401
    if "dirichlet_entropies_of_bma" in estimates:
        prefixes.append("dirichlet")
    if "link_normcdf_output_entropies_of_bma" in estimates:
        prefixes.append("link_normcdf_output")
    if "link_sigmoid_output_entropies_of_bma" in estimates:
        prefixes.append("link_sigmoid_output")
    if "laplace_bridge_entropies_of_bma" in estimates:
        prefixes.append("laplace_bridge")
    if "mean_field_entropies_of_bma" in estimates:
        prefixes.append("mean_field")

    for prefix in prefixes:
        gt_hard_bma_correctnesses_original = targets[
            f"{prefix}_gt_hard_bma_correctnesses_original"
        ]
        gt_hard_bma_correctnesses = targets[f"{prefix}_gt_hard_bma_correctnesses"]

        for estimator_name in estimates:
            if not estimator_name.startswith(prefix):
                continue

            # In `estimates`, we have *uncertainty* estimates: higher signals more
            # uncertain. For correctness prediction, we need *certainty* estimates: the
            # AUROC is high if there exists a threshold for which all certain samples
            # are correct (1) and all others are incorrect (0).

            estimate = -estimates[estimator_name]

            metrics[
                f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness_original"
            ] = auroc(gt_hard_bma_correctnesses_original, estimate).item()
            metrics[f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness"] = auroc(
                gt_hard_bma_correctnesses, estimate
            ).item()

        # Performance metrics
        metrics[f"{key_prefix}{prefix}_hard_bma_accuracy_original"] = (
            gt_hard_bma_correctnesses_original.float().mean().item()
        )
        metrics[f"{key_prefix}{prefix}_hard_bma_accuracy"] = (
            gt_hard_bma_correctnesses.float().mean().item()
        )

        if is_soft_dataset:
            gt_soft_bma_correctnesses = targets[f"{prefix}_gt_soft_bma_correctnesses"]
            metrics[f"{key_prefix}{prefix}_soft_bma_accuracy"] = (
                gt_soft_bma_correctnesses.mean().item()
            )

    if is_soft_dataset:
        probs = targets["gt_soft_labels"]
        max_labels = probs.max(dim=1)[0]
        metrics[f"{key_prefix}best_soft_accuracy"] = max_labels.mean().item()

    return metrics


def evaluate_on_abstained_prediction(
    estimates,
    targets,
    is_soft_dataset,
    args,
    is_soft_upstream_dataset,
):
    is_mixed_eval = is_soft_upstream_dataset is not None

    # For correctness of prediction, one of the datasets being soft is enough
    if is_mixed_eval:
        is_soft_dataset = is_soft_dataset or is_soft_upstream_dataset

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed_eval else ""

    prefixes = []
    if "mc_10_entropies_of_bma" in estimates:
        for i in [10, 100, 1000]:
            prefixes.append(f"mc_{i}")  # noqa: PERF401
    if "dirichlet_entropies_of_bma" in estimates:
        prefixes.append("dirichlet")
    if "link_normcdf_output_entropies_of_bma" in estimates:
        prefixes.append("link_normcdf_output")
    if "link_sigmoid_output_entropies_of_bma" in estimates:
        prefixes.append("link_sigmoid_output")
    if "laplace_bridge_entropies_of_bma" in estimates:
        prefixes.append("laplace_bridge")
    if "mean_field_entropies_of_bma" in estimates:
        prefixes.append("mean_field")

    for prefix in prefixes:
        gt_hard_bma_correctnesses_original = targets[
            f"{prefix}_gt_hard_bma_correctnesses_original"
        ]
        gt_hard_bma_correctnesses = targets[f"{prefix}_gt_hard_bma_correctnesses"]

        if is_soft_dataset:
            gt_soft_bma_correctnesses = targets[f"{prefix}_gt_soft_bma_correctnesses"]

        for estimator_name in estimates:
            if not estimator_name.startswith(prefix):
                continue

            estimate = estimates[estimator_name]

            metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc_original"] = (
                area_under_risk_coverage_curve(
                    estimate, gt_hard_bma_correctnesses_original
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc"] = (
                area_under_risk_coverage_curve(
                    estimate, gt_hard_bma_correctnesses
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_hard_bma_abstinence_auc_original"
            ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc_original"]
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_hard_bma_abstinence_auc"
            ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_bma_aurc"]
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
                area_under_lift_curve(
                    estimate, gt_hard_bma_correctnesses_original
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_hard_bma_aulc"] = (
                area_under_lift_curve(estimate, gt_hard_bma_correctnesses).item()
            )
            metrics[f"{key_prefix}{estimator_name}_hard_bma_raulc_original"] = (
                relative_area_under_lift_curve(
                    estimate, gt_hard_bma_correctnesses_original
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_hard_bma_raulc"] = (
                relative_area_under_lift_curve(
                    estimate, gt_hard_bma_correctnesses
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_hard_bma_coverage_for_95_accuracy_original"
            ] = coverage_for_accuracy(
                estimate, gt_hard_bma_correctnesses_original, accuracy=0.95
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_hard_bma_coverage_for_95_accuracy"
            ] = coverage_for_accuracy(
                estimate, gt_hard_bma_correctnesses, accuracy=0.95
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_hard_bma_coverage_for_99_accuracy_original"
            ] = coverage_for_accuracy(
                estimate, gt_hard_bma_correctnesses_original, accuracy=0.99
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_hard_bma_coverage_for_99_accuracy"
            ] = coverage_for_accuracy(
                estimate, gt_hard_bma_correctnesses, accuracy=0.99
            ).item()

            if is_soft_dataset:
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

    return metrics


def evaluate_on_ood_detection(estimates, targets, args):
    metrics = {}
    for estimator_name in estimates:
        metrics[f"mixed_{args.dataset_id}_{estimator_name}_auroc_oodness"] = auroc(
            targets["gt_oodness"], estimates[estimator_name]
        ).item()

    return metrics


def evaluate_on_proper_scoring_and_calibration(
    estimates,
    log_probs,
    targets,
    is_soft_dataset,
    args,
    is_soft_upstream_dataset,
):
    is_mixed_eval = is_soft_upstream_dataset is not None

    # For proper scoring and calibration, one of the datasets being soft is enough
    if is_mixed_eval:
        is_soft_dataset = is_soft_dataset or is_soft_upstream_dataset

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed_eval else ""

    gt_hard_labels_original = targets["gt_hard_labels_original"]
    gt_hard_labels = targets["gt_hard_labels"]

    prefixes = []
    if "mc_10_entropies_of_bma" in estimates:
        for i in [10, 100, 1000]:
            prefixes.append(f"mc_{i}")  # noqa: PERF401
    if "dirichlet_entropies_of_bma" in estimates:
        prefixes.append("dirichlet")
    if "link_normcdf_output_entropies_of_bma" in estimates:
        prefixes.append("link_normcdf_output")
    if "link_sigmoid_output_entropies_of_bma" in estimates:
        prefixes.append("link_sigmoid_output")
    if "laplace_bridge_entropies_of_bma" in estimates:
        prefixes.append("laplace_bridge")
    if "mean_field_entropies_of_bma" in estimates:
        prefixes.append("mean_field")

    for prefix in prefixes:
        # Proper scoring and calibration for correctness of prediction
        gt_hard_bma_correctnesses_original = targets[
            f"{prefix}_gt_hard_bma_correctnesses_original"
        ]
        gt_hard_bma_correctnesses = targets[f"{prefix}_gt_hard_bma_correctnesses"]

        if is_soft_dataset:
            gt_soft_bma_correctnesses = targets[f"{prefix}_gt_soft_bma_correctnesses"]

        estimator_name = f"{prefix}_one_minus_max_probs_of_bma"
        estimate = estimates[estimator_name]
        estimate = 1 - estimate  # convert to correctness probability

        # {Hard, Soft}-label correctness
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

        if is_soft_dataset:
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

        # Proper scoring for aleatoric uncertainty
        metrics[f"{key_prefix}{prefix}_log_prob_score_hard_bma_aleatoric_original"] = (
            multiclass_log_probability(
                log_probs[f"{prefix}_log_bmas"], gt_hard_labels_original
            ).item()
        )
        metrics[f"{key_prefix}{prefix}_log_prob_score_hard_bma_aleatoric"] = (
            multiclass_log_probability(
                log_probs[f"{prefix}_log_bmas"], gt_hard_labels
            ).item()
        )
        metrics[f"{key_prefix}{prefix}_brier_score_hard_bma_aleatoric_original"] = (
            multiclass_brier(
                log_probs[f"{prefix}_log_bmas"],
                gt_hard_labels_original,
                is_soft_targets=False,
            ).item()
        )
        metrics[f"{key_prefix}{prefix}_brier_score_hard_bma_aleatoric"] = (
            multiclass_brier(
                log_probs[f"{prefix}_log_bmas"], gt_hard_labels, is_soft_targets=False
            ).item()
        )

        if is_soft_dataset:
            gt_soft_labels = targets["gt_soft_labels"]

            metrics[f"{key_prefix}{prefix}_log_prob_score_soft_bma_aleatoric"] = (
                multiclass_log_probability(
                    log_probs[f"{prefix}_log_bmas"], gt_soft_labels
                ).item()
            )
            metrics[f"{key_prefix}{prefix}_brier_score_soft_bma_aleatoric"] = (
                multiclass_brier(
                    log_probs[f"{prefix}_log_bmas"],
                    gt_soft_labels,
                    is_soft_targets=True,
                ).item()
            )

    return metrics


def evaluate_on_correlation_of_decomposition(
    estimates,
    is_soft_dataset,
    args,
    is_soft_upstream_dataset,
):
    is_mixed_eval = is_soft_upstream_dataset is not None

    # For Bregman, both datasets need to be soft
    if is_mixed_eval:
        is_soft_dataset = is_soft_dataset and is_soft_upstream_dataset

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed_eval else ""

    prefixes = []
    if "mc_10_entropies_of_bma" in estimates:
        for i in [10, 100, 1000]:
            prefixes.append(f"mc_{i}")  # noqa: PERF401
    if "dirichlet_entropies_of_bma" in estimates:
        prefixes.append("dirichlet")
    if "link_normcdf_output_entropies_of_bma" in estimates:
        prefixes.append("link_normcdf_output")
    if "link_sigmoid_output_entropies_of_bma" in estimates:
        prefixes.append("link_sigmoid_output")
    if "laplace_bridge_entropies_of_bma" in estimates:
        prefixes.append("laplace_bridge")
    if "mean_field_entropies_of_bma" in estimates:
        prefixes.append("mean_field")

    for prefix in prefixes:
        # Information-theoretical decomposition
        entropies_of_bma = estimates[f"{prefix}_entropies_of_bma"]
        expected_entropies = estimates[f"{prefix}_expected_entropies"]
        jensen_shannon_divergences = estimates[f"{prefix}_jensen_shannon_divergences"]

        metrics[f"{key_prefix}{prefix}_rank_correlation_it_au_eu"] = float(
            spearmanr(expected_entropies, jensen_shannon_divergences)
        )
        metrics[f"{key_prefix}{prefix}_correlation_it_au_eu"] = float(
            pearsonr(expected_entropies, jensen_shannon_divergences)
        )

        metrics[f"{key_prefix}{prefix}_rank_correlation_it_au_pu"] = float(
            spearmanr(expected_entropies, entropies_of_bma)
        )
        metrics[f"{key_prefix}{prefix}_correlation_it_au_pu"] = float(
            pearsonr(expected_entropies, entropies_of_bma)
        )

        metrics[f"{key_prefix}{prefix}_rank_correlation_it_eu_pu"] = float(
            spearmanr(jensen_shannon_divergences, entropies_of_bma)
        )
        metrics[f"{key_prefix}{prefix}_correlation_it_eu_pu"] = float(
            pearsonr(jensen_shannon_divergences, entropies_of_bma)
        )

    return metrics


def forward_general_model_on_loader(
    model,
    loader,
    is_soft_dataset,
    amp_autocast,
    device,
    storage_device,
    args,
    log_probs,
    estimates,
    time_forward_m,
    gt_aleatorics_bregman,
    gt_soft_labels,
    gt_hard_labels,
    gt_hard_labels_original,
):
    current_ind = 0

    for input, label in loader:
        indices = slice(current_ind, current_ind + input.shape[0])

        if not args.prefetcher:
            input = input.to(device)
            label = label.to(device)

        if is_soft_dataset:
            hard_label = label[:, -1]
            label = label[:, :-1]

        batch_size = input.shape[0]

        time_forward_start = time.perf_counter()
        with amp_autocast():
            inference_res = model(input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        time_forward_end = time.perf_counter()
        time_forward = time_forward_end - time_forward_start

        inference_res = tuple(
            res.detach().float().to(storage_device) for res in inference_res
        )

        inference_res = convert_inference_res(
            inference_res=inference_res,
            time_forward=time_forward,
            args=args,
        )

        update_logit_based(
            inference_res=inference_res,
            indices=indices,
            batch_size=batch_size,
            log_probs=log_probs,
            estimates=estimates,
            time_forward_m=time_forward_m,
        )

        # GT containers
        if is_soft_dataset:
            prob = label.float() / label.sum(dim=1, keepdim=True)  # Normalization
            prob = prob.to(storage_device)
            gt_aleatorics_bregman[indices] = entropy(prob)

            log_prob = prob.log()
            min_real = torch.finfo(log_prob.dtype).min
            log_prob = torch.clamp(log_prob, min=min_real)

            gt_soft_labels[indices] = prob
            gt_hard_labels_original[indices] = hard_label.to(storage_device)
            gt_hard_labels[indices] = prob.argmax(dim=1)
        else:
            gt_hard_labels_original[indices] = label.to(storage_device)
            gt_hard_labels[indices] = label.to(storage_device)

        current_ind += input.shape[0]


def forward_deep_ensemble_on_loader(
    model,
    loader,
    is_soft_dataset,
    amp_autocast,
    device,
    storage_device,
    num_samples,
    args,
    log_probs,
    estimates,
    time_forward_m,
    gt_aleatorics_bregman,
    gt_soft_labels,
    gt_hard_labels,
    gt_hard_labels_original,
):
    temp_logits = torch.empty(
        num_samples, model.num_models, model.num_classes, device=storage_device
    )
    time_forwards = torch.empty(len(loader), model.num_models, device=storage_device)

    for model_index in range(model.num_models):
        logger.info(f"Loading model {model_index + 1}/{model.num_models}.")

        model.load_model(model_index)
        current_ind = 0
        for i, (input, _) in enumerate(loader):
            batch_size = input.shape[0]
            indices = slice(current_ind, current_ind + batch_size)

            if not args.prefetcher:
                input = input.to(device)

            time_forward_start = time.perf_counter()
            with amp_autocast():
                inference_res = model(input)

            if device.type == "cuda":
                torch.cuda.synchronize()

            time_forward_end = time.perf_counter()
            time_forward = time_forward_end - time_forward_start

            temp_logits[indices, model_index, :] = inference_res["logit"].to(
                storage_device
            )
            time_forwards[i, model_index] = time_forward

            current_ind += batch_size

    # Aggregate logits and forward times
    time_forwards_sum = time_forwards.sum(dim=-1)

    current_ind = 0
    for i, (input, label) in enumerate(loader):
        batch_size = input.shape[0]
        indices = slice(current_ind, current_ind + batch_size)

        inference_res = {
            "logit": temp_logits[indices],
        }

        inference_res = convert_inference_res(
            model=model,
            inference_res=inference_res,
            time_forward=time_forwards_sum[i],
            args=args,
        )

        update_logit_based(
            inference_res=inference_res,
            indices=indices,
            batch_size=batch_size,
            log_probs=log_probs,
            estimates=estimates,
            time_forward_m=time_forward_m,
        )

        # GT containers
        if is_soft_dataset:
            hard_label = label[:, -1]
            label = label[:, :-1]

            prob = label.float() / label.sum(dim=1, keepdim=True)  # Normalization
            prob = prob.to(storage_device)
            gt_aleatorics_bregman[indices] = entropy(prob)

        if is_soft_dataset:
            log_prob = prob.log()
            min_real = torch.finfo(log_prob.dtype).min
            log_prob = torch.clamp(log_prob, min=min_real)
            gt_soft_labels[indices] = prob
            gt_hard_labels_original[indices] = hard_label.to(storage_device)
            gt_hard_labels[indices] = prob.argmax(dim=1)
        else:
            label = label.to(storage_device)
            gt_hard_labels_original[indices] = label
            gt_hard_labels[indices] = label

        current_ind += batch_size


def calc_correctnesses(estimates, log_probs, targets, is_soft):
    prefixes = []
    if "mc_10_entropies_of_bma" in estimates:
        for i in [10, 100, 1000]:
            prefixes.append(f"mc_{i}")  # noqa: PERF401
    if "dirichlet_entropies_of_bma" in estimates:
        prefixes.append("dirichlet")
    if "link_normcdf_output_entropies_of_bma" in estimates:
        prefixes.append("link_normcdf_output")
    if "link_sigmoid_output_entropies_of_bma" in estimates:
        prefixes.append("link_sigmoid_output")
    if "laplace_bridge_entropies_of_bma" in estimates:
        prefixes.append("laplace_bridge")
    if "mean_field_entropies_of_bma" in estimates:
        prefixes.append("mean_field")

    for prefix in prefixes:
        predicted_labels_bma = log_probs[f"{prefix}_log_bmas"].argmax(dim=1)

        targets[f"{prefix}_gt_hard_bma_correctnesses_original"] = (
            predicted_labels_bma.eq(targets["gt_hard_labels_original"]).int()
        )
        targets[f"{prefix}_gt_hard_bma_correctnesses"] = predicted_labels_bma.eq(
            targets["gt_hard_labels"]
        ).int()

        if is_soft:
            targets[f"{prefix}_gt_soft_bma_correctnesses"] = (
                targets["gt_soft_labels"]
                .gather(dim=1, index=predicted_labels_bma.unsqueeze(dim=1))
                .squeeze(dim=1)
            )


def extract_averages(times):
    for key in list(times.keys()):
        times[key] = times[key].avg


def remove_faulty_indices(estimates, log_probs, targets):
    faulty_indices = targets["gt_aleatorics_bregman"].isnan()

    if faulty_indices.sum() > 0:
        for estimator_name in list(estimates.keys()):
            estimates[estimator_name] = estimates[estimator_name][~faulty_indices]

        for log_prob_name in list(log_probs.keys()):
            log_probs[log_prob_name] = log_probs[log_prob_name][~faulty_indices]

        for target_name in list(targets.keys()):
            targets[target_name] = targets[target_name][~faulty_indices]


def get_bundle(
    model,
    loader,
    device,
    storage_device,
    amp_autocast,
    is_soft_dataset,
    args,
):
    estimates = {}
    log_probs = {}
    targets = {}
    times = {}

    num_samples = len(loader.dataset)  # Total number of samples

    # Ground truth containers

    # Practical tasks

    # Abstained prediction
    gt_hard_labels = torch.empty(num_samples, dtype=torch.long, device=storage_device)
    gt_hard_labels_original = torch.empty(
        num_samples, dtype=torch.long, device=storage_device
    )
    targets["gt_hard_labels"] = gt_hard_labels
    targets["gt_hard_labels_original"] = gt_hard_labels_original

    # Correctness of prediction
    # We calculate the correctness in the CoP evaluation function. Here,
    # we only record the GT labels and our predictions (logits).
    # The reason is that there are two possible ways to treat soft labels

    # OOD detection
    # We don't record OOD-ness, as these labels are decided at a later point of the code

    # Proper scoring and calibration
    # We only need the labels and the logits to calculate these metrics

    # Theoretical tasks

    if is_soft_dataset:
        gt_soft_labels = torch.empty(
            num_samples, model.num_classes, device=storage_device
        )
        targets["gt_soft_labels"] = gt_soft_labels

        # Bregman Aleatoric uncertainty
        gt_aleatorics_bregman = torch.empty(num_samples, device=storage_device)
        targets["gt_aleatorics_bregman"] = gt_aleatorics_bregman

    # Estimate containers
    # Time
    time_forward_m = AverageMeter()
    times["time_forward_m"] = time_forward_m

    link = args.predictive.split("_")[0]
    is_distributional_het = isinstance(model, HETWrapper) and not args.use_sampling
    is_distributional = is_distributional_het or isinstance(
        model, SNGPWrapper | CovariancePushforwardLaplaceWrapper | LinearizedSWAGWrapper
    )

    if not isinstance(model, EDLWrapper | PostNetWrapper):
        for i in [10, 100, 1000]:
            log_bmas = torch.zeros(
                num_samples, model.num_classes, device=storage_device
            )
            log_probs[f"mc_{i}_log_bmas"] = log_bmas
            expected_entropies = torch.zeros(num_samples, device=storage_device)
            estimates[f"mc_{i}_expected_entropies"] = expected_entropies
            entropies_of_bma = torch.zeros(num_samples, device=storage_device)
            estimates[f"mc_{i}_entropies_of_bma"] = entropies_of_bma
            one_minus_max_probs_of_bma = torch.zeros(num_samples, device=storage_device)
            estimates[f"mc_{i}_one_minus_max_probs_of_bma"] = one_minus_max_probs_of_bma
            jensen_shannon_divergences = torch.zeros(num_samples, device=storage_device)
            estimates[f"mc_{i}_jensen_shannon_divergences"] = jensen_shannon_divergences

    if isinstance(model, EDLWrapper | PostNetWrapper) or (
        is_distributional and link != "softmax"
    ):
        if is_distributional:
            if link == "probit":
                suffixes = ["link_normcdf_output", "link_sigmoid_output", "link_mc"]
            elif link == "logit":
                suffixes = [
                    "link_normcdf_output",
                    "link_sigmoid_output",
                    "link_sigmoid_product_output",
                    "link_mc",
                ]
        else:
            suffixes = ["edl"]

        for suffix in suffixes:
            log_bmas = torch.empty(
                num_samples, model.num_classes, device=storage_device
            )
            log_probs[f"{suffix}_dirichlet_log_bmas"] = log_bmas
            expected_entropies = torch.empty(num_samples, device=storage_device)
            estimates[f"{suffix}_dirichlet_expected_entropies"] = expected_entropies
            entropies_of_bma = torch.empty(num_samples, device=storage_device)
            estimates[f"{suffix}_dirichlet_entropies_of_bma"] = entropies_of_bma
            one_minus_max_probs_of_bma = torch.empty(num_samples, device=storage_device)
            estimates[f"{suffix}_dirichlet_one_minus_max_probs_of_bma"] = (
                one_minus_max_probs_of_bma
            )
            jensen_shannon_divergences = torch.empty(num_samples, device=storage_device)
            estimates[f"{suffix}_dirichlet_jensen_shannon_divergences"] = (
                jensen_shannon_divergences
            )

    if is_distributional:
        if link == "softmax":
            suffixes = ["laplace_bridge", "mean_field", "mc"]
        elif link == "probit":
            suffixes = ["link_normcdf_output", "link_sigmoid_output", "link_mc"]
        elif link == "logit":
            suffixes = [
                "link_normcdf_output",
                "link_sigmoid_output",
                "link_sigmoid_product_output",
                "link_mc",
            ]

        for suffix in suffixes:
            log_bmas = torch.empty(
                num_samples, model.num_classes, device=storage_device
            )
            log_probs[f"{suffix}_log_bmas"] = log_bmas
            entropies_of_bma = torch.empty(num_samples, device=storage_device)
            estimates[f"{suffix}_expected_entropies"] = expected_entropies
            one_minus_max_probs_of_bma = torch.empty(num_samples, device=storage_device)
            estimates[f"{suffix}_entropies_of_bma"] = entropies_of_bma
            expected_entropies = torch.empty(num_samples, device=storage_device)
            estimates[f"{suffix}_one_minus_max_probs_of_bma"] = (
                one_minus_max_probs_of_bma
            )
            jensen_shannon_divergences = torch.empty(num_samples, device=storage_device)
            estimates[f"{suffix}_jensen_shannon_divergences"] = (
                jensen_shannon_divergences
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
            log_probs=log_probs,
            estimates=estimates,
            time_forward_m=time_forward_m,
            gt_aleatorics_bregman=gt_aleatorics_bregman,
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
            log_probs=log_probs,
            estimates=estimates,
            time_forward_m=time_forward_m,
            gt_aleatorics_bregman=gt_aleatorics_bregman,
            gt_soft_labels=gt_soft_labels,
            gt_hard_labels=gt_hard_labels,
            gt_hard_labels_original=gt_hard_labels_original,
        )

    # Calculate correctness indicators
    calc_correctnesses(estimates, log_probs, targets, is_soft_dataset)

    # Extract averages from the AverageMeters
    extract_averages(times)

    if is_soft_dataset:
        remove_faulty_indices(estimates, log_probs, targets)

    return estimates, log_probs, targets, times


def handle_samples(logits, converted_inference_res, act_fn, num_samples):
    i = num_samples
    min_real = torch.finfo(logits.dtype).min
    if logits.dim() == 2:  # [B, C]
        logits = logits.unsqueeze(dim=1)  # [B, 1, C]

    probs = act_fn(logits)  # [B, S, C]

    bmas = probs.mean(dim=1)  # [B, C]
    log_bmas = bmas.log().clamp(min=min_real)  # [B, C]
    converted_inference_res[f"mc_{i}_log_bmas"] = log_bmas

    expected_entropies = entropy(probs).mean(dim=-1)
    converted_inference_res[f"mc_{i}_expected_entropies"] = expected_entropies

    entropies_of_bma = entropy(bmas)
    converted_inference_res[f"mc_{i}_entropies_of_bma"] = entropies_of_bma

    one_minus_max_probs_of_bma = 1 - bmas.max(dim=-1)[0]
    converted_inference_res[f"mc_{i}_one_minus_max_probs_of_bma"] = (
        one_minus_max_probs_of_bma
    )

    jsds = entropies_of_bma - expected_entropies
    converted_inference_res[f"mc_{i}_jensen_shannon_divergences"] = jsds


def handle_alpha(alpha, converted_inference_res, prefix):
    min_real = torch.finfo(alpha.dtype).min

    sum_alphas = alpha.sum(dim=1)  # [B]
    mean_alphas = alpha.div(sum_alphas.unsqueeze(1))  # [B, C]

    log_bmas = mean_alphas.log().clamp(min=min_real)
    converted_inference_res[f"{prefix}_dirichlet_log_bmas"] = log_bmas

    digamma_term = torch.digamma(alpha + 1) - torch.digamma(sum_alphas + 1).unsqueeze(
        1
    )  # [B, C]
    expected_entropies = -mean_alphas.mul(digamma_term).sum(dim=1)  # [B]
    converted_inference_res[f"{prefix}_dirichlet_expected_entropies"] = (
        expected_entropies
    )

    entropies_of_bma = entropy(mean_alphas)
    converted_inference_res[f"{prefix}_dirichlet_entropies_of_bma"] = entropies_of_bma

    one_minus_max_probs_of_bma = 1 - mean_alphas.max(dim=-1)[0]
    converted_inference_res[f"{prefix}_dirichlet_one_minus_max_probs_of_bma"] = (
        one_minus_max_probs_of_bma
    )

    jsd = entropies_of_bma - expected_entropies
    converted_inference_res[f"{prefix}_dirichlet_jensen_shannon_divergences"] = jsd


def handle_bma(bma, converted_inference_res, prefix):
    min_real = torch.finfo(bma.dtype).min
    log_bma = bma.log().clamp(min=min_real)  # [B, C]
    converted_inference_res[f"{prefix}_log_bmas"] = log_bma

    entropies_of_bma = entropy(bma)
    converted_inference_res[f"{prefix}_entropies_of_bma"] = entropies_of_bma

    one_minus_max_probs_of_bma = 1 - bma.max(dim=-1)[0]
    converted_inference_res[f"{prefix}_one_minus_max_probs_of_bma"] = (
        one_minus_max_probs_of_bma
    )


def convert_inference_res(inference_res, time_forward, args):
    converted_inference_res = {}

    converted_inference_res["time_forward"] = time_forward

    if len(inference_res) == 2:
        mean, var = inference_res
        # TODO(bmucsanyi): ask Nathaël if this is the only predictive we want to use in
        # the eval or others as well (e.g., does the output_function matter a lot for
        # training HET? For others I think we should evaluate all predictives with the
        # same link function. For HET, we could also do that and just choose an
        # arbitrary predictive for training...)

        # TODO(bmucsanyi): create the containers very thoroughly, based on the following
        # if-else structure
        link = args.predictive.split("_")[0]
        if link == "softmax":
            suffixes = ["laplace_bridge", "mean_field", "mc"]
        elif link == "probit":
            suffixes = ["link_normcdf_output", "link_sigmoid_output", "link_mc"]
        elif link == "logit":
            suffixes = [
                "link_normcdf_output",
                "link_sigmoid_output",
                "link_sigmoid_product_output",
                "link_mc",
            ]

        for suffix in suffixes:
            predictive_name = f"{link}_{suffix}"
            predictive_fn = get_predictive(
                predictive_name, args.use_correction, args.num_mc_samples
            )

            if suffix.endswith("mc"):
                for i in [10, 100, 1000]:
                    mc_predictive_fn = get_predictive(
                        predictive_name, args.use_correction, i
                    )
                    bma, samples = mc_predictive_fn(
                        mean, var, return_samples=True
                    )  # [B, C]
                    act_fn = get_activation(predictive_name)
                    handle_samples(samples, converted_inference_res, act_fn, i)
            else:
                bma = predictive_fn(mean, var)
                handle_bma(bma, converted_inference_res, suffix)

        if link != "softmax":
            for suffix in suffixes:
                if suffix.endswith("mc"):
                    continue

                predictive_name = f"{link}_{suffix}"
                dirichlet_fn = get_dirichlet(predictive_name)
                alpha = dirichlet_fn(mean, var)
                handle_alpha(alpha, converted_inference_res, suffix)

    elif len(inference_res) == 1 and inference_res[0].ndim == 3:
        samples = inference_res[0]
        act_fn = get_activation(args.predictive)
        for i in [10, 100, 1000]:
            handle_samples(samples, converted_inference_res, act_fn, i)
    elif len(inference_res) == 1 and inference_res[0].ndim == 2:
        alpha = inference_res[0]
        handle_alpha(alpha, converted_inference_res, "edl")
    else:
        msg = "Invalid inference_res structure"
        raise ValueError(msg)

    return converted_inference_res


def update_logit_based(
    inference_res,
    indices,
    batch_size,
    log_probs,
    estimates,
    time_forward_m,
):
    for key in inference_res:
        if key == "time_forward":
            time_forward_m.update(inference_res[key], batch_size)
        elif key.endswith("log_bmas"):
            log_probs[key][indices] = inference_res[key]
        else:
            estimates[key][indices] = inference_res[key]
