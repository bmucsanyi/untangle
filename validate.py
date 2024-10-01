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
    HetClassNNWrapper,
    MahalanobisWrapper,
)

logger = logging.getLogger(__name__)


def evaluate_bulk(
    model,
    loaders,
    device,
    storage_device,
    amp_autocast,
    key_prefix,
    output_dir,
    is_upstream_dataset,
    is_test_dataset,
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
                loader_name=f"{name}_{ood_transform_type}",
                device=device,
                storage_device=storage_device,
                amp_autocast=amp_autocast,
                key_prefix="",
                output_dir=output_dir,
                is_upstream_dataset=is_upstream_dataset,
                is_test_dataset=is_test_dataset,
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


@torch.no_grad()
def evaluate(
    model,
    loader,
    loader_name,
    device,
    storage_device,
    amp_autocast,
    key_prefix,
    output_dir,
    is_upstream_dataset,
    is_test_dataset,
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

    if is_test_dataset:
        ood_prefix = "id" if is_upstream_dataset else "ood"
        save_prefix = f"{ood_prefix}_test_{loader_name.replace('/', '_')}_"

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
        metrics = evaluate_on_auroc_hard_bma_correctness(
            estimates=estimates,
            targets=targets,
            metrics=metrics,
            args=args,
        )

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    if is_upstream_dataset and is_test_dataset and output_dir is not None:
        # Save ingredients to disk
        max_num_indices = len(targets["gt_hard_labels"])
        num_indices = min(max_num_indices, args.max_num_id_ood_eval_samples // 2)
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
    elif is_test_dataset and output_dir is not None:
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

            mixed_targets["gt_soft_fbar_correctnesses"] = torch.cat([
                upstream_targets["gt_soft_fbar_correctnesses"],
                downstream_targets["gt_hard_fbar_correctnesses"],
            ])
            mixed_targets["gt_soft_fbar_correctnesses_top5"] = torch.cat([
                upstream_targets["gt_soft_fbar_correctnesses_top5"],
                downstream_targets["gt_hard_fbar_correctnesses_top5"],
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

            mixed_targets["gt_soft_fbar_correctnesses"] = torch.cat([
                upstream_targets["gt_hard_fbar_correctnesses"],
                downstream_targets["gt_soft_fbar_correctnesses"],
            ])
            mixed_targets["gt_soft_fbar_correctnesses_top5"] = torch.cat([
                upstream_targets["gt_hard_fbar_correctnesses_top5"],
                downstream_targets["gt_soft_fbar_correctnesses_top5"],
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
            f"{ood_prefix}_test_{loader_name.replace('/', '_')}_mixed_"
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


def evaluate_on_auroc_hard_bma_correctness(
    estimates,
    targets,
    metrics,
    args,
):
    metrics["hard_bma_accuracy_original"] = (
        targets["gt_hard_bma_correctnesses_original"].float().mean().item()
    )

    for estimator_name in estimates:
        if estimator_name in args.eval_metric:
            estimate = -estimates[estimator_name]

            gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses"]
            metrics[f"{estimator_name}_auroc_hard_bma_correctness"] = auroc(
                gt_hard_bma_correctnesses, estimate
            ).item()

            break

    return metrics


def evaluate_on_tasks(
    model,
    estimates,
    log_probs,
    targets,
    metrics,
    is_soft_dataset,
    save_prefix,
    output_dir,
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

    gt_hard_fbar_correctnesses_original = targets["gt_hard_fbar_correctnesses_original"]
    gt_hard_bma_correctnesses_original = targets["gt_hard_bma_correctnesses_original"]
    gt_hard_fbar_correctnesses = targets["gt_hard_fbar_correctnesses"]
    gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses"]

    gt_hard_fbar_correctnesses_original_top5 = targets[
        "gt_hard_fbar_correctnesses_original_top5"
    ]
    gt_hard_bma_correctnesses_original_top5 = targets[
        "gt_hard_bma_correctnesses_original_top5"
    ]
    gt_hard_fbar_correctnesses_top5 = targets["gt_hard_fbar_correctnesses_top5"]
    gt_hard_bma_correctnesses_top5 = targets["gt_hard_bma_correctnesses_top5"]

    for estimator_name in estimates:
        # In `estimates`, we have *uncertainty* estimates: higher signals more
        # uncertain. For correctness prediction, we need *certainty* estimates: the
        # AUROC is high if there exists a threshold for which all certain samples are
        # correct (1) and all others are incorrect (0).

        estimate = -estimates[estimator_name]

        metrics[
            f"{key_prefix}{estimator_name}_auroc_hard_fbar_correctness_original"
        ] = auroc(gt_hard_fbar_correctnesses_original, estimate).item()
        metrics[f"{key_prefix}{estimator_name}_auroc_hard_fbar_correctness"] = auroc(
            gt_hard_fbar_correctnesses, estimate
        ).item()
        metrics[f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness_original"] = (
            auroc(gt_hard_bma_correctnesses_original, estimate).item()
        )
        metrics[f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness"] = auroc(
            gt_hard_bma_correctnesses, estimate
        ).item()

        metrics[
            f"{key_prefix}{estimator_name}_auroc_hard_fbar_correctness_original_top5"
        ] = auroc(gt_hard_fbar_correctnesses_original_top5, estimate).item()
        metrics[f"{key_prefix}{estimator_name}_auroc_hard_fbar_correctness_top5"] = (
            auroc(gt_hard_fbar_correctnesses_top5, estimate).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness_original_top5"
        ] = auroc(gt_hard_bma_correctnesses_original_top5, estimate).item()
        metrics[f"{key_prefix}{estimator_name}_auroc_hard_bma_correctness_top5"] = (
            auroc(gt_hard_bma_correctnesses_top5, estimate).item()
        )

    # Performance metrics
    metrics[f"{key_prefix}hard_fbar_accuracy_original"] = (
        targets["gt_hard_fbar_correctnesses_original"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_fbar_accuracy"] = (
        targets["gt_hard_fbar_correctnesses"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_bma_accuracy_original"] = (
        targets["gt_hard_bma_correctnesses_original"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_bma_accuracy"] = (
        targets["gt_hard_bma_correctnesses"].float().mean().item()
    )

    metrics[f"{key_prefix}hard_fbar_accuracy_original_top5"] = (
        targets["gt_hard_fbar_correctnesses_original_top5"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_fbar_accuracy_top5"] = (
        targets["gt_hard_fbar_correctnesses_top5"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_bma_accuracy_original_top5"] = (
        targets["gt_hard_bma_correctnesses_original_top5"].float().mean().item()
    )
    metrics[f"{key_prefix}hard_bma_accuracy_top5"] = (
        targets["gt_hard_bma_correctnesses_top5"].float().mean().item()
    )

    if is_soft_dataset:
        metrics[f"{key_prefix}soft_fbar_accuracy"] = (
            targets["gt_soft_fbar_correctnesses"].mean().item()
        )
        metrics[f"{key_prefix}soft_bma_accuracy"] = (
            targets["gt_soft_bma_correctnesses"].mean().item()
        )

        metrics[f"{key_prefix}soft_fbar_accuracy_top5"] = (
            targets["gt_soft_fbar_correctnesses_top5"].mean().item()
        )
        metrics[f"{key_prefix}soft_bma_accuracy_top5"] = (
            targets["gt_soft_bma_correctnesses_top5"].mean().item()
        )

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

    gt_hard_fbar_correctnesses_original = targets["gt_hard_fbar_correctnesses_original"]
    gt_hard_fbar_correctnesses = targets["gt_hard_fbar_correctnesses"]
    gt_hard_bma_correctnesses_original = targets["gt_hard_bma_correctnesses_original"]
    gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses"]

    gt_hard_fbar_correctnesses_original_top5 = targets[
        "gt_hard_fbar_correctnesses_original_top5"
    ]
    gt_hard_fbar_correctnesses_top5 = targets["gt_hard_fbar_correctnesses_top5"]
    gt_hard_bma_correctnesses_original_top5 = targets[
        "gt_hard_bma_correctnesses_original_top5"
    ]
    gt_hard_bma_correctnesses_top5 = targets["gt_hard_bma_correctnesses_top5"]

    if is_soft_dataset:
        gt_soft_fbar_correctnesses = targets["gt_soft_fbar_correctnesses"]
        gt_soft_bma_correctnesses = targets["gt_soft_bma_correctnesses"]

        gt_soft_fbar_correctnesses_top5 = targets["gt_soft_fbar_correctnesses_top5"]
        gt_soft_bma_correctnesses_top5 = targets["gt_soft_bma_correctnesses_top5"]

    for estimator_name in estimates:
        estimate = estimates[estimator_name]

        metrics[f"{key_prefix}{estimator_name}_hard_fbar_aurc_original"] = (
            area_under_risk_coverage_curve(
                estimate, gt_hard_fbar_correctnesses_original
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_aurc"] = (
            area_under_risk_coverage_curve(estimate, gt_hard_fbar_correctnesses).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_cumulative_hard_fbar_abstinence_auc_original"
        ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_fbar_aurc_original"]
        metrics[f"{key_prefix}{estimator_name}_cumulative_hard_fbar_abstinence_auc"] = (
            1 - metrics[f"{key_prefix}{estimator_name}_hard_fbar_aurc"]
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_eaurc_original"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_fbar_correctnesses_original
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_eaurc"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_fbar_correctnesses
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_aulc_original"] = (
            area_under_lift_curve(estimate, gt_hard_fbar_correctnesses_original).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_aulc"] = area_under_lift_curve(
            estimate, gt_hard_fbar_correctnesses
        ).item()
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_raulc_original"] = (
            relative_area_under_lift_curve(
                estimate, gt_hard_fbar_correctnesses_original
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_raulc"] = (
            relative_area_under_lift_curve(estimate, gt_hard_fbar_correctnesses).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_hard_fbar_coverage_for_95_accuracy_original"
        ] = coverage_for_accuracy(
            estimate, gt_hard_fbar_correctnesses_original, accuracy=0.95
        ).item()
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_coverage_for_95_accuracy"] = (
            coverage_for_accuracy(
                estimate, gt_hard_fbar_correctnesses, accuracy=0.95
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_hard_fbar_coverage_for_99_accuracy_original"
        ] = coverage_for_accuracy(
            estimate, gt_hard_fbar_correctnesses_original, accuracy=0.99
        ).item()
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_coverage_for_99_accuracy"] = (
            coverage_for_accuracy(
                estimate, gt_hard_fbar_correctnesses, accuracy=0.99
            ).item()
        )

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

        metrics[f"{key_prefix}{estimator_name}_hard_fbar_aurc_original_top5"] = (
            area_under_risk_coverage_curve(
                estimate, gt_hard_fbar_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_aurc_top5"] = (
            area_under_risk_coverage_curve(
                estimate, gt_hard_fbar_correctnesses_top5
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_cumulative_hard_fbar_abstinence_auc_original_top5"
        ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_fbar_aurc_original_top5"]
        metrics[
            f"{key_prefix}{estimator_name}_cumulative_hard_fbar_abstinence_auc_top5"
        ] = 1 - metrics[f"{key_prefix}{estimator_name}_hard_fbar_aurc_top5"]
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_eaurc_original_top5"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_fbar_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_eaurc_top5"] = (
            excess_area_under_risk_coverage_curve(
                estimate, gt_hard_fbar_correctnesses_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_aulc_original_top5"] = (
            area_under_lift_curve(
                estimate, gt_hard_fbar_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_aulc_top5"] = (
            area_under_lift_curve(estimate, gt_hard_fbar_correctnesses_top5).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_raulc_original_top5"] = (
            relative_area_under_lift_curve(
                estimate, gt_hard_fbar_correctnesses_original_top5
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_hard_fbar_raulc_top5"] = (
            relative_area_under_lift_curve(
                estimate, gt_hard_fbar_correctnesses_top5
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_hard_fbar_coverage_for_95_accuracy_original_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_fbar_correctnesses_original_top5, accuracy=0.95
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_fbar_coverage_for_95_accuracy_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_fbar_correctnesses_top5, accuracy=0.95
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_fbar_coverage_for_99_accuracy_original_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_fbar_correctnesses_original_top5, accuracy=0.99
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_hard_fbar_coverage_for_99_accuracy_top5"
        ] = coverage_for_accuracy(
            estimate, gt_hard_fbar_correctnesses_top5, accuracy=0.99
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
            metrics[f"{key_prefix}{estimator_name}_soft_fbar_aurc"] = (
                area_under_risk_coverage_curve(
                    estimate, gt_soft_fbar_correctnesses
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_soft_fbar_abstinence_auc"
            ] = 1 - metrics[f"{key_prefix}{estimator_name}_soft_fbar_aurc"]
            metrics[f"{key_prefix}{estimator_name}_soft_fbar_eaurc"] = (
                excess_area_under_risk_coverage_curve(
                    estimate, gt_soft_fbar_correctnesses
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_fbar_aulc"] = (
                area_under_lift_curve(estimate, gt_soft_fbar_correctnesses).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_fbar_raulc"] = (
                relative_area_under_lift_curve(
                    estimate, gt_soft_fbar_correctnesses
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_soft_fbar_coverage_for_95_accuracy"
            ] = coverage_for_accuracy(
                estimate, gt_soft_fbar_correctnesses, accuracy=0.95
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_soft_fbar_coverage_for_99_accuracy"
            ] = coverage_for_accuracy(
                estimate, gt_soft_fbar_correctnesses, accuracy=0.99
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

            metrics[f"{key_prefix}{estimator_name}_soft_fbar_aurc_top5"] = (
                area_under_risk_coverage_curve(
                    estimate, gt_soft_fbar_correctnesses_top5
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_cumulative_soft_fbar_abstinence_auc_top5"
            ] = 1 - metrics[f"{key_prefix}{estimator_name}_soft_fbar_aurc_top5"]
            metrics[f"{key_prefix}{estimator_name}_soft_fbar_eaurc_top5"] = (
                excess_area_under_risk_coverage_curve(
                    estimate, gt_soft_fbar_correctnesses_top5
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_fbar_aulc_top5"] = (
                area_under_lift_curve(estimate, gt_soft_fbar_correctnesses_top5).item()
            )
            metrics[f"{key_prefix}{estimator_name}_soft_fbar_raulc_top5"] = (
                relative_area_under_lift_curve(
                    estimate, gt_soft_fbar_correctnesses_top5
                ).item()
            )
            metrics[
                f"{key_prefix}{estimator_name}_soft_fbar_coverage_for_95_accuracy_top5"
            ] = coverage_for_accuracy(
                estimate, gt_soft_fbar_correctnesses_top5, accuracy=0.95
            ).item()
            metrics[
                f"{key_prefix}{estimator_name}_soft_fbar_coverage_for_99_accuracy_top5"
            ] = coverage_for_accuracy(
                estimate, gt_soft_fbar_correctnesses_top5, accuracy=0.99
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


def evaluate_on_ood_detection(estimates, targets, args):
    metrics = {}
    for estimator_name in estimates:
        metrics[f"mixed_{args.dataset_id}_{estimator_name}_auroc_oodness"] = auroc(
            targets["gt_oodness"], estimates[estimator_name]
        ).item()

    return metrics


def evaluate_on_proper_scoring_and_calibration(
    model,
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

    # Proper scoring and calibration for correctness of prediction
    correctness_estimator_names = [
        "one_minus_expected_max_probs",
        "one_minus_max_probs_of_fbar",
        "one_minus_max_probs_of_bma",
    ]

    if isinstance(model, DUQWrapper):
        correctness_estimator_names.append("duq_values")

    if isinstance(model, BaseCorrectnessPredictionWrapper):
        correctness_estimator_names.append("error_probabilities")

    gt_hard_fbar_correctnesses_original = targets["gt_hard_fbar_correctnesses_original"]
    gt_hard_fbar_correctnesses = targets["gt_hard_fbar_correctnesses"]
    gt_hard_bma_correctnesses_original = targets["gt_hard_bma_correctnesses_original"]
    gt_hard_bma_correctnesses = targets["gt_hard_bma_correctnesses"]

    gt_hard_fbar_correctnesses_original_top5 = targets[
        "gt_hard_fbar_correctnesses_original_top5"
    ]
    gt_hard_fbar_correctnesses_top5 = targets["gt_hard_fbar_correctnesses_top5"]
    gt_hard_bma_correctnesses_original_top5 = targets[
        "gt_hard_bma_correctnesses_original_top5"
    ]
    gt_hard_bma_correctnesses_top5 = targets["gt_hard_bma_correctnesses_top5"]

    if is_soft_dataset:
        gt_soft_fbar_correctnesses = targets["gt_soft_fbar_correctnesses"]
        gt_soft_bma_correctnesses = targets["gt_soft_bma_correctnesses"]

        gt_soft_fbar_correctnesses_top5 = targets["gt_soft_fbar_correctnesses_top5"]
        gt_soft_bma_correctnesses_top5 = targets["gt_soft_bma_correctnesses_top5"]

    for estimator_name in correctness_estimator_names:
        estimate = estimates[estimator_name]

        estimate = 1 - estimate  # convert to correctness probability

        # {Hard, Soft}-label correctness
        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_hard_fbar_correctness_original"
        ] = binary_log_probability(estimate, gt_hard_fbar_correctnesses_original).item()
        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_hard_fbar_correctness"
        ] = binary_log_probability(estimate, gt_hard_fbar_correctnesses).item()
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_fbar_correctness_original"
        ] = binary_brier(estimate, gt_hard_fbar_correctnesses_original).item()
        metrics[f"{key_prefix}{estimator_name}_brier_score_hard_fbar_correctness"] = (
            binary_brier(estimate, gt_hard_fbar_correctnesses).item()
        )
        metrics[f"{key_prefix}{estimator_name}_ece_hard_fbar_correctness_original"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_fbar_correctnesses_original,
                num_bins=15,
                norm="l1",
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_ece_hard_fbar_correctness"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_fbar_correctnesses,
                num_bins=15,
                norm="l1",
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_mce_hard_fbar_correctness_original"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_fbar_correctnesses_original,
                num_bins=15,
                norm="inf",
            ).item()
        )
        metrics[f"{key_prefix}{estimator_name}_mce_hard_fbar_correctness"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_fbar_correctnesses,
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
            f"{key_prefix}{estimator_name}_log_prob_score_hard_fbar_correctness_original_top5"
        ] = binary_log_probability(
            estimate, gt_hard_fbar_correctnesses_original_top5
        ).item()
        metrics[
            f"{key_prefix}{estimator_name}_log_prob_score_hard_fbar_correctness_top5"
        ] = binary_log_probability(estimate, gt_hard_fbar_correctnesses_top5).item()
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_fbar_correctness_original_top5"
        ] = binary_brier(estimate, gt_hard_fbar_correctnesses_original_top5).item()
        metrics[
            f"{key_prefix}{estimator_name}_brier_score_hard_fbar_correctness_top5"
        ] = binary_brier(estimate, gt_hard_fbar_correctnesses_top5).item()
        metrics[
            f"{key_prefix}{estimator_name}_ece_hard_fbar_correctness_original_top5"
        ] = calibration_error(
            confidences=estimate,
            correctnesses=gt_hard_fbar_correctnesses_original_top5,
            num_bins=15,
            norm="l1",
        ).item()
        metrics[f"{key_prefix}{estimator_name}_ece_hard_fbar_correctness_top5"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_fbar_correctnesses_top5,
                num_bins=15,
                norm="l1",
            ).item()
        )
        metrics[
            f"{key_prefix}{estimator_name}_mce_hard_fbar_correctness_original_top5"
        ] = calibration_error(
            confidences=estimate,
            correctnesses=gt_hard_fbar_correctnesses_original_top5,
            num_bins=15,
            norm="inf",
        ).item()
        metrics[f"{key_prefix}{estimator_name}_mce_hard_fbar_correctness_top5"] = (
            calibration_error(
                confidences=estimate,
                correctnesses=gt_hard_fbar_correctnesses_top5,
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
                f"{key_prefix}{estimator_name}_log_prob_score_soft_fbar_correctness"
            ] = binary_log_probability(estimate, gt_soft_fbar_correctnesses).item()
            metrics[
                f"{key_prefix}{estimator_name}_brier_score_soft_fbar_correctness"
            ] = binary_brier(estimate, gt_soft_fbar_correctnesses).item()
            metrics[f"{key_prefix}{estimator_name}_ece_soft_fbar_correctness"] = (
                calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_fbar_correctnesses,
                    num_bins=15,
                    norm="l1",
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_mce_soft_fbar_correctness"] = (
                calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_fbar_correctnesses,
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
                f"{key_prefix}{estimator_name}_log_prob_score_soft_fbar_correctness_top5"
            ] = binary_log_probability(estimate, gt_soft_fbar_correctnesses_top5).item()
            metrics[
                f"{key_prefix}{estimator_name}_brier_score_soft_fbar_correctness_top5"
            ] = binary_brier(estimate, gt_soft_fbar_correctnesses_top5).item()
            metrics[f"{key_prefix}{estimator_name}_ece_soft_fbar_correctness_top5"] = (
                calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_fbar_correctnesses_top5,
                    num_bins=15,
                    norm="l1",
                ).item()
            )
            metrics[f"{key_prefix}{estimator_name}_mce_soft_fbar_correctness_top5"] = (
                calibration_error(
                    confidences=estimate,
                    correctnesses=gt_soft_fbar_correctnesses_top5,
                    num_bins=15,
                    norm="inf",
                ).item()
            )

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

    metrics[f"{key_prefix}log_prob_score_hard_fbar_aleatoric_original"] = (
        multiclass_log_probability(
            log_probs["log_fbars"], gt_hard_labels_original
        ).item()
    )
    metrics[f"{key_prefix}log_prob_score_hard_fbar_aleatoric"] = (
        multiclass_log_probability(log_probs["log_fbars"], gt_hard_labels).item()
    )
    metrics[f"{key_prefix}brier_score_hard_fbar_aleatoric_original"] = multiclass_brier(
        log_probs["log_fbars"], gt_hard_labels_original, is_soft_targets=False
    ).item()
    metrics[f"{key_prefix}brier_score_hard_fbar_aleatoric"] = multiclass_brier(
        log_probs["log_fbars"], gt_hard_labels, is_soft_targets=False
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

        metrics[f"{key_prefix}log_prob_score_soft_fbar_aleatoric"] = (
            multiclass_log_probability(log_probs["log_fbars"], gt_soft_labels).item()
        )
        metrics[f"{key_prefix}brier_score_soft_fbar_aleatoric"] = multiclass_brier(
            log_probs["log_fbars"], gt_soft_labels, is_soft_targets=True
        ).item()

        metrics[f"{key_prefix}log_prob_score_soft_bma_aleatoric"] = (
            multiclass_log_probability(log_probs["log_bmas"], gt_soft_labels).item()
        )
        metrics[f"{key_prefix}brier_score_soft_bma_aleatoric"] = multiclass_brier(
            log_probs["log_bmas"], gt_soft_labels, is_soft_targets=True
        ).item()

    return metrics


def evaluate_on_bregman(
    estimates,
    targets,
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

    gt_predictives_bregman_fbar = targets["gt_predictives_bregman_fbar"]
    gt_predictives_bregman_bma = targets["gt_predictives_bregman_bma"]

    gt_total_predictives_bregman_fbar = targets["gt_total_predictives_bregman_fbar"]
    gt_total_predictives_bregman_bma = targets["gt_total_predictives_bregman_bma"]

    if is_soft_dataset:
        gt_biases_bregman_fbar = targets["gt_biases_bregman_fbar"]
        gt_biases_bregman_bma = targets["gt_biases_bregman_bma"]

    if is_soft_dataset:
        gt_aleatorics_bregman = targets["gt_aleatorics_bregman"]
        multi_label_indices = (gt_aleatorics_bregman > 0).int()

    gt_epistemics_bregman = targets["gt_epistemics_bregman"]

    for estimator_name in estimates:
        estimate = estimates[estimator_name]

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

        metrics[f"{key_prefix}{estimator_name}_rank_correlation_bregman_pu_fbar"] = (
            spearmanr(estimate, gt_predictives_bregman_fbar).item()
        )
        metrics[f"{key_prefix}{estimator_name}_mse_bregman_pu_fbar"] = (
            (estimate - gt_predictives_bregman_fbar).square().mean().item()
        )
        metrics[f"{key_prefix}{estimator_name}_mae_bregman_pu_fbar"] = (
            (estimate - gt_predictives_bregman_fbar).abs().mean().item()
        )

        metrics[
            f"{key_prefix}{estimator_name}_rank_correlation_bregman_total_pu_fbar"
        ] = spearmanr(estimate, gt_total_predictives_bregman_fbar).item()
        metrics[f"{key_prefix}{estimator_name}_mse_bregman_total_pu_fbar"] = (
            (estimate - gt_total_predictives_bregman_fbar).square().mean().item()
        )
        metrics[f"{key_prefix}{estimator_name}_mae_bregman_total_pu_fbar"] = (
            (estimate - gt_total_predictives_bregman_fbar).abs().mean().item()
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
            metrics[f"{key_prefix}{estimator_name}_rank_correlation_bregman_b_fbar"] = (
                spearmanr(estimate, gt_biases_bregman_fbar).item()
            )
            metrics[f"{key_prefix}{estimator_name}_mse_bregman_b_fbar"] = (
                (estimate - gt_biases_bregman_fbar).square().mean().item()
            )
            metrics[f"{key_prefix}{estimator_name}_mae_bregman_b_fbar"] = (
                (estimate - gt_biases_bregman_fbar).abs().mean().item()
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
    model,
    estimates,
    output_dir,
    save_prefix,
    args,
    is_soft_upstream_dataset,
):
    metrics = {}

    is_mixed_eval = is_soft_upstream_dataset is not None
    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed_eval else ""

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
    estimates,
    targets,
    is_soft_dataset,
    output_dir,
    save_prefix,
    args,
    is_soft_upstream_dataset,
):
    is_mixed_eval = is_soft_upstream_dataset is not None

    # For Bregman, both datasets need to be soft
    if is_mixed_eval:
        is_soft_dataset = is_soft_dataset and is_soft_upstream_dataset

    metrics = {}

    key_prefix = f"mixed_{args.dataset_id}_" if is_mixed_eval else ""

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
    gt_predictives_bregman_fbar = targets["gt_predictives_bregman_fbar"]
    gt_predictives_bregman_bma = targets["gt_predictives_bregman_bma"]

    gt_total_predictives_bregman_fbar = targets["gt_total_predictives_bregman_fbar"]
    gt_total_predictives_bregman_bma = targets["gt_total_predictives_bregman_bma"]

    if is_soft_dataset:
        gt_biases_bregman_fbar = targets["gt_biases_bregman_fbar"]
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
        metrics[f"{key_prefix}rank_correlation_bregman_au_b_fbar"] = float(
            spearmanr(gt_aleatorics_bregman, gt_biases_bregman_fbar)
        )
        metrics[f"{key_prefix}correlation_bregman_au_b_fbar"] = float(
            pearsonr(gt_aleatorics_bregman, gt_biases_bregman_fbar)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_b_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_biases_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_au_b_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_biases_bregman_bma)
        )

    if can_evaluate_au_pu:
        metrics[f"{key_prefix}rank_correlation_bregman_au_pu_fbar"] = float(
            spearmanr(gt_aleatorics_bregman, gt_predictives_bregman_fbar)
        )
        metrics[f"{key_prefix}correlation_bregman_au_pu_fbar"] = float(
            pearsonr(gt_aleatorics_bregman, gt_predictives_bregman_fbar)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_pu_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_predictives_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_au_pu_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_predictives_bregman_bma)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_total_pu_fbar"] = float(
            spearmanr(gt_aleatorics_bregman, gt_total_predictives_bregman_fbar)
        )
        metrics[f"{key_prefix}correlation_bregman_au_total_pu_fbar"] = float(
            pearsonr(gt_aleatorics_bregman, gt_total_predictives_bregman_fbar)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_au_total_pu_bma"] = float(
            spearmanr(gt_aleatorics_bregman, gt_total_predictives_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_au_total_pu_bma"] = float(
            pearsonr(gt_aleatorics_bregman, gt_total_predictives_bregman_bma)
        )

    if can_evaluate_b_pu:
        metrics[f"{key_prefix}rank_correlation_bregman_b_pu_fbar"] = float(
            spearmanr(gt_biases_bregman_fbar, gt_predictives_bregman_fbar)
        )
        metrics[f"{key_prefix}correlation_bregman_b_pu_fbar"] = float(
            pearsonr(gt_biases_bregman_fbar, gt_predictives_bregman_fbar)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_b_pu_bma"] = float(
            spearmanr(gt_biases_bregman_bma, gt_predictives_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_b_pu_bma"] = float(
            pearsonr(gt_biases_bregman_bma, gt_predictives_bregman_bma)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_b_total_pu_fbar"] = float(
            spearmanr(gt_biases_bregman_fbar, gt_total_predictives_bregman_fbar)
        )
        metrics[f"{key_prefix}correlation_bregman_b_total_pu_fbar"] = float(
            pearsonr(gt_biases_bregman_fbar, gt_total_predictives_bregman_fbar)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_b_total_pu_bma"] = float(
            spearmanr(gt_biases_bregman_bma, gt_total_predictives_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_b_total_pu_bma"] = float(
            pearsonr(gt_biases_bregman_bma, gt_total_predictives_bregman_bma)
        )

    if can_evaluate_eu_b:
        metrics[f"{key_prefix}rank_correlation_bregman_eu_b_fbar"] = float(
            spearmanr(gt_epistemics_bregman, gt_biases_bregman_fbar)
        )
        metrics[f"{key_prefix}correlation_bregman_eu_b_fbar"] = float(
            pearsonr(gt_epistemics_bregman, gt_biases_bregman_fbar)
        )

        metrics[f"{key_prefix}rank_correlation_bregman_eu_b_bma"] = float(
            spearmanr(gt_epistemics_bregman, gt_biases_bregman_bma)
        )
        metrics[f"{key_prefix}correlation_bregman_eu_b_bma"] = float(
            pearsonr(gt_epistemics_bregman, gt_biases_bregman_bma)
        )

    metrics[f"{key_prefix}rank_correlation_bregman_eu_pu_fbar"] = float(
        spearmanr(gt_epistemics_bregman, gt_predictives_bregman_fbar)
    )
    metrics[f"{key_prefix}correlation_bregman_eu_pu_fbar"] = float(
        pearsonr(gt_epistemics_bregman, gt_predictives_bregman_fbar)
    )

    metrics[f"{key_prefix}rank_correlation_bregman_eu_pu_bma"] = float(
        spearmanr(gt_epistemics_bregman, gt_predictives_bregman_bma)
    )
    metrics[f"{key_prefix}correlation_bregman_eu_pu_bma"] = float(
        pearsonr(gt_epistemics_bregman, gt_predictives_bregman_bma)
    )

    metrics[f"{key_prefix}rank_correlation_bregman_eu_total_pu_fbar"] = float(
        spearmanr(gt_epistemics_bregman, gt_total_predictives_bregman_fbar)
    )
    metrics[f"{key_prefix}correlation_bregman_eu_total_pu_fbar"] = float(
        pearsonr(gt_epistemics_bregman, gt_total_predictives_bregman_fbar)
    )

    metrics[f"{key_prefix}rank_correlation_bregman_eu_total_pu_bma"] = float(
        spearmanr(gt_epistemics_bregman, gt_total_predictives_bregman_bma)
    )
    metrics[f"{key_prefix}correlation_bregman_eu_total_pu_bma"] = float(
        pearsonr(gt_epistemics_bregman, gt_total_predictives_bregman_bma)
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
    log_fbars,
    log_bmas,
    gt_epistemics_bregman,
    time_forward_m,
    expected_entropies,
    expected_entropies_plus_expected_divergences,
    one_minus_expected_max_probs,
    entropies_of_bma,
    entropies_of_fbar,
    one_minus_max_probs_of_bma,
    one_minus_max_probs_of_fbar,
    jensen_shannon_divergences,
    dempster_shafer_values,
    expected_variances_of_probs,
    expected_variances_of_logits,
    loss_values,
    gmm_neg_log_densities,
    error_probabilities,
    duq_values,
    mahalanobis_values,
    expected_variances_of_internal_probs,
    expected_variances_of_internal_logits,
    gt_aleatorics_bregman,
    gt_biases_bregman_fbar,
    gt_biases_bregman_bma,
    gt_predictives_bregman_fbar,
    gt_predictives_bregman_bma,
    gt_total_predictives_bregman_fbar,
    gt_total_predictives_bregman_bma,
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
            log_fbars=log_fbars,
            log_bmas=log_bmas,
            gt_epistemics_bregman=gt_epistemics_bregman,
            time_forward_m=time_forward_m,
            expected_entropies=expected_entropies,
            expected_entropies_plus_expected_divergences=expected_entropies_plus_expected_divergences,
            one_minus_expected_max_probs=one_minus_expected_max_probs,
            entropies_of_bma=entropies_of_bma,
            entropies_of_fbar=entropies_of_fbar,
            one_minus_max_probs_of_bma=one_minus_max_probs_of_bma,
            one_minus_max_probs_of_fbar=one_minus_max_probs_of_fbar,
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

        log_fbar = inference_dict["log_fbar"]
        log_bma = inference_dict["log_bma"]
        if is_soft_dataset:
            log_prob = prob.log()
            min_real = torch.finfo(log_prob.dtype).min
            log_prob = torch.clamp(log_prob, min=min_real)

            gt_biases_bregman_fbar[indices] = kl_divergence(log_prob, log_fbar)
            gt_biases_bregman_bma[indices] = kl_divergence(log_prob, log_bma)
            gt_predictives_bregman_fbar[indices] = (
                gt_aleatorics_bregman[indices] + gt_biases_bregman_fbar[indices]
            )
            gt_predictives_bregman_bma[indices] = (
                gt_aleatorics_bregman[indices] + gt_biases_bregman_bma[indices]
            )
            gt_total_predictives_bregman_fbar[indices] = (
                gt_aleatorics_bregman[indices]
                + gt_biases_bregman_fbar[indices]
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
            gt_predictives_bregman_fbar[indices] = F.cross_entropy(
                log_fbar, label.to(storage_device)
            )
            gt_predictives_bregman_bma[indices] = F.cross_entropy(
                log_bma, label.to(storage_device)
            )
            gt_total_predictives_bregman_fbar[indices] = F.cross_entropy(
                log_fbar, label.to(storage_device)
            )
            gt_total_predictives_bregman_bma[indices] = F.cross_entropy(
                log_bma, label.to(storage_device)
            )

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
    log_fbars,
    log_bmas,
    gt_epistemics_bregman,
    time_forward_m,
    expected_entropies,
    expected_entropies_plus_expected_divergences,
    one_minus_expected_max_probs,
    entropies_of_bma,
    entropies_of_fbar,
    one_minus_max_probs_of_bma,
    one_minus_max_probs_of_fbar,
    jensen_shannon_divergences,
    dempster_shafer_values,
    expected_variances_of_probs,
    expected_variances_of_logits,
    gt_aleatorics_bregman,
    gt_biases_bregman_fbar,
    gt_biases_bregman_bma,
    gt_predictives_bregman_fbar,
    gt_predictives_bregman_bma,
    gt_total_predictives_bregman_fbar,
    gt_total_predictives_bregman_bma,
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
    for i, (input, label) in enumerate(loader):
        batch_size = input.shape[0]
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
            log_fbars=log_fbars,
            log_bmas=log_bmas,
            gt_epistemics_bregman=gt_epistemics_bregman,
            time_forward_m=time_forward_m,
            expected_entropies=expected_entropies,
            expected_entropies_plus_expected_divergences=expected_entropies_plus_expected_divergences,
            one_minus_expected_max_probs=one_minus_expected_max_probs,
            entropies_of_bma=entropies_of_bma,
            entropies_of_fbar=entropies_of_fbar,
            one_minus_max_probs_of_bma=one_minus_max_probs_of_bma,
            one_minus_max_probs_of_fbar=one_minus_max_probs_of_fbar,
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

        log_fbar = inference_dict["log_fbar"]
        log_bma = inference_dict["log_bma"]
        if is_soft_dataset:
            log_prob = prob.log()
            min_real = torch.finfo(log_prob.dtype).min
            log_prob = torch.clamp(log_prob, min=min_real)
            gt_biases_bregman_fbar[indices] = kl_divergence(log_prob, log_fbar)
            gt_biases_bregman_bma[indices] = kl_divergence(log_prob, log_bma)
            gt_predictives_bregman_fbar[indices] = (
                gt_aleatorics_bregman[indices] + gt_biases_bregman_fbar[indices]
            )
            gt_predictives_bregman_bma[indices] = (
                gt_aleatorics_bregman[indices] + gt_biases_bregman_bma[indices]
            )
            gt_total_predictives_bregman_fbar[indices] = (
                gt_aleatorics_bregman[indices]
                + gt_biases_bregman_fbar[indices]
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

            gt_predictives_bregman_fbar[indices] = F.cross_entropy(log_fbar, label)
            gt_predictives_bregman_bma[indices] = F.cross_entropy(log_bma, label)
            gt_total_predictives_bregman_fbar[indices] = F.cross_entropy(
                log_fbar, label
            )
            gt_total_predictives_bregman_bma[indices] = F.cross_entropy(log_bma, label)

        current_ind += batch_size


def calc_correctnesses(log_probs, targets, is_soft):
    predicted_labels_fbar = log_probs["log_fbars"].argmax(dim=1)

    targets["gt_hard_fbar_correctnesses_original"] = predicted_labels_fbar.eq(
        targets["gt_hard_labels_original"]
    ).int()
    targets["gt_hard_fbar_correctnesses"] = predicted_labels_fbar.eq(
        targets["gt_hard_labels"]
    ).int()

    _, predicted_labels_fbar_top5 = torch.topk(log_probs["log_fbars"], 5, dim=1)
    expanded_gt_hard_labels_original = (
        targets["gt_hard_labels_original"]
        .unsqueeze(dim=1)
        .expand_as(predicted_labels_fbar_top5)
    )
    targets["gt_hard_fbar_correctnesses_original_top5"] = (
        predicted_labels_fbar_top5.eq(expanded_gt_hard_labels_original)
        .max(dim=1)[0]
        .int()
    )
    expanded_gt_hard_labels = (
        targets["gt_hard_labels"].unsqueeze(dim=1).expand_as(predicted_labels_fbar_top5)
    )
    targets["gt_hard_fbar_correctnesses_top5"] = (
        predicted_labels_fbar_top5.eq(expanded_gt_hard_labels).max(dim=1)[0].int()
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
        targets["gt_soft_fbar_correctnesses"] = (
            targets["gt_soft_labels"]
            .gather(dim=1, index=predicted_labels_fbar.unsqueeze(dim=1))
            .squeeze(dim=1)
        )

        indexed_gt_soft_labels_fbar = targets["gt_soft_labels"].gather(
            dim=1, index=predicted_labels_fbar_top5
        )
        targets["gt_soft_fbar_correctnesses_top5"] = indexed_gt_soft_labels_fbar.max(
            dim=1
        )[0]

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
        # Also interested in how well the GT solves the practical tasks
        estimates["gt_aleatorics_bregman"] = gt_aleatorics_bregman

        # Bregman Bias
        gt_biases_bregman_fbar = torch.empty(num_samples, device=storage_device)
        targets["gt_biases_bregman_fbar"] = gt_biases_bregman_fbar
        estimates["gt_biases_bregman_fbar"] = gt_biases_bregman_fbar

        gt_biases_bregman_bma = torch.empty(num_samples, device=storage_device)
        targets["gt_biases_bregman_bma"] = gt_biases_bregman_bma
        estimates["gt_biases_bregman_bma"] = gt_biases_bregman_bma

    # Estimate containers
    # Predictive uncertainty (Bregman)
    gt_predictives_bregman_fbar = torch.empty(num_samples, device=storage_device)
    targets["gt_predictives_bregman_fbar"] = gt_predictives_bregman_fbar
    estimates["gt_predictives_bregman_fbar"] = gt_predictives_bregman_fbar

    gt_total_predictives_bregman_fbar = torch.empty(num_samples, device=storage_device)
    targets["gt_total_predictives_bregman_fbar"] = gt_total_predictives_bregman_fbar
    estimates["gt_total_predictives_bregman_fbar"] = gt_total_predictives_bregman_fbar

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

    log_fbars = torch.empty(num_samples, model.num_classes, device=storage_device)
    log_probs["log_fbars"] = log_fbars

    log_bmas = torch.empty(num_samples, model.num_classes, device=storage_device)
    log_probs["log_bmas"] = log_bmas

    # AU
    expected_entropies = torch.empty(num_samples, device=storage_device)
    estimates["expected_entropies"] = expected_entropies
    one_minus_expected_max_probs = torch.empty(num_samples, device=storage_device)
    estimates["one_minus_expected_max_probs"] = one_minus_expected_max_probs

    # PU
    entropies_of_bma = torch.empty(num_samples, device=storage_device)
    estimates["entropies_of_bma"] = entropies_of_bma
    entropies_of_fbar = torch.empty(num_samples, device=storage_device)
    estimates["entropies_of_fbar"] = entropies_of_fbar
    one_minus_max_probs_of_bma = torch.empty(num_samples, device=storage_device)
    estimates["one_minus_max_probs_of_bma"] = one_minus_max_probs_of_bma
    one_minus_max_probs_of_fbar = torch.empty(
        num_samples, device=storage_device
    )  # Just an extra thing to try out
    estimates["one_minus_max_probs_of_fbar"] = one_minus_max_probs_of_fbar
    expected_entropies_plus_expected_divergences = torch.empty(
        num_samples, device=storage_device
    )
    estimates["expected_entropies_plus_expected_divergences"] = (
        expected_entropies_plus_expected_divergences
    )

    # EU
    dempster_shafer_values = torch.empty(num_samples, device=storage_device)
    estimates["dempster_shafer_values"] = dempster_shafer_values
    # Just a duplicate
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

    # This class modifies the model when it's not frozen, leading to different
    # logits.
    if isinstance(model, BaseLossPredictionWrapper):
        # PU
        loss_values = torch.empty(num_samples, device=storage_device)
        estimates["loss_values"] = loss_values
    elif isinstance(model, DDUWrapper):
        gmm_neg_log_densities = torch.empty(num_samples, device=storage_device)
        estimates["gmm_neg_log_densities"] = gmm_neg_log_densities
    # This class also modifies the model when it's not frozen.
    elif isinstance(model, BaseCorrectnessPredictionWrapper):
        # PU
        error_probabilities = torch.empty(num_samples, device=storage_device)
        estimates["error_probabilities"] = error_probabilities
    # This class gives "logits" that are different from the baseline model.
    elif isinstance(model, DUQWrapper):
        # EU
        duq_values = torch.empty(num_samples, device=storage_device)
        estimates["duq_values"] = duq_values
    # While this class returns logits, it's post-hoc. As such, the logits are not
    # changed compared to the baseline model, so we'd get the same results.
    elif isinstance(model, MahalanobisWrapper):
        # EU
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
            log_fbars=log_fbars,
            log_bmas=log_bmas,
            gt_epistemics_bregman=gt_epistemics_bregman,
            time_forward_m=time_forward_m,
            expected_entropies=expected_entropies,
            expected_entropies_plus_expected_divergences=expected_entropies_plus_expected_divergences,
            one_minus_expected_max_probs=one_minus_expected_max_probs,
            entropies_of_bma=entropies_of_bma,
            entropies_of_fbar=entropies_of_fbar,
            one_minus_max_probs_of_bma=one_minus_max_probs_of_bma,
            one_minus_max_probs_of_fbar=one_minus_max_probs_of_fbar,
            jensen_shannon_divergences=jensen_shannon_divergences,
            dempster_shafer_values=dempster_shafer_values,
            expected_variances_of_probs=expected_variances_of_probs,
            expected_variances_of_logits=expected_variances_of_logits,
            gt_aleatorics_bregman=gt_aleatorics_bregman,
            gt_biases_bregman_fbar=gt_biases_bregman_fbar,
            gt_biases_bregman_bma=gt_biases_bregman_bma,
            gt_predictives_bregman_fbar=gt_predictives_bregman_fbar,
            gt_predictives_bregman_bma=gt_predictives_bregman_bma,
            gt_total_predictives_bregman_fbar=gt_total_predictives_bregman_fbar,
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
            log_fbars=log_fbars,
            log_bmas=log_bmas,
            gt_epistemics_bregman=gt_epistemics_bregman,
            time_forward_m=time_forward_m,
            expected_entropies=expected_entropies,
            expected_entropies_plus_expected_divergences=expected_entropies_plus_expected_divergences,
            one_minus_expected_max_probs=one_minus_expected_max_probs,
            entropies_of_bma=entropies_of_bma,
            entropies_of_fbar=entropies_of_fbar,
            one_minus_max_probs_of_bma=one_minus_max_probs_of_bma,
            one_minus_max_probs_of_fbar=one_minus_max_probs_of_fbar,
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
            gt_biases_bregman_fbar=gt_biases_bregman_fbar,
            gt_biases_bregman_bma=gt_biases_bregman_bma,
            gt_predictives_bregman_fbar=gt_predictives_bregman_fbar,
            gt_predictives_bregman_bma=gt_predictives_bregman_bma,
            gt_total_predictives_bregman_fbar=gt_total_predictives_bregman_fbar,
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


def convert_inference_dict(model, inference_dict, time_forward, args):
    converted_inference_dict = {}

    converted_inference_dict["time_forward"] = time_forward

    if isinstance(model, DirichletWrapper):
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

        log_fbar = F.log_softmax(log_probs.mean(dim=1), dim=-1)  # [B, C]
        converted_inference_dict["log_fbar"] = log_fbar

        digamma_term = torch.digamma(alphas + 1) - torch.digamma(
            sum_alphas + 1
        ).unsqueeze(1)  # [B, C]
        expected_entropy = -mean_alphas.mul(digamma_term).sum(dim=1)  # [B]
        converted_inference_dict["expected_entropy"] = expected_entropy

        expected_divergence = kl_divergence(log_fbar, log_probs.permute(1, 0, 2)).mean(
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

        fbar = log_fbar.exp()

        entropy_of_fbar = entropy(fbar)
        converted_inference_dict["entropy_of_fbar"] = entropy_of_fbar

        max_prob_of_bma = mean_alphas.max(dim=-1)[0]
        converted_inference_dict["max_prob_of_bma"] = max_prob_of_bma

        max_prob_of_fbar = fbar.max(dim=-1)[0]
        converted_inference_dict["max_prob_of_fbar"] = max_prob_of_fbar

        jensen_shannon_divergence = entropy_of_bma - expected_entropy
        converted_inference_dict["jensen_shannon_divergence"] = (
            jensen_shannon_divergence
        )

        num_classes = alphas.shape[1]
        dempster_shafer_value = num_classes / sum_alphas  # [B]
        converted_inference_dict["dempster_shafer_value"] = dempster_shafer_value
    else:
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

        log_fbar = F.log_softmax(log_probs.mean(dim=1), dim=-1)  # [B, C]

        fbar = log_fbar.exp()
        converted_inference_dict["log_fbar"] = log_fbar

        bma = probs.mean(dim=1)  # [B, C]

        log_bma = bma.log()  # [B, C]
        log_bma = torch.clamp(log_bma, min=min_real)
        converted_inference_dict["log_bma"] = log_bma

        expected_entropy = entropy(probs).mean(dim=-1)
        converted_inference_dict["expected_entropy"] = expected_entropy

        expected_divergence = kl_divergence(log_fbar, log_probs.permute(1, 0, 2)).mean(
            dim=0
        )
        converted_inference_dict["expected_divergence"] = expected_divergence

        expected_max_prob = probs.max(dim=-1)[0].mean(dim=1)
        converted_inference_dict["expected_max_prob"] = expected_max_prob

        entropy_of_bma = entropy(bma)

        converted_inference_dict["entropy_of_bma"] = entropy_of_bma

        entropy_of_fbar = entropy(fbar)
        converted_inference_dict["entropy_of_fbar"] = entropy_of_fbar

        max_prob_of_bma = bma.max(dim=-1)[0]
        converted_inference_dict["max_prob_of_bma"] = max_prob_of_bma

        max_prob_of_fbar = fbar.max(dim=-1)[0]
        converted_inference_dict["max_prob_of_fbar"] = max_prob_of_fbar

        jensen_shannon_divergence = entropy_of_bma - expected_entropy
        converted_inference_dict["jensen_shannon_divergence"] = (
            jensen_shannon_divergence
        )

        dempster_shafer_value = dempster_shafer_metric(logits.mean(dim=1))
        converted_inference_dict["dempster_shafer_value"] = dempster_shafer_value

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


def update_logit_based(
    inference_dict,
    indices,
    batch_size,
    log_fbars,
    log_bmas,
    gt_epistemics_bregman,
    time_forward_m,
    expected_entropies,
    expected_entropies_plus_expected_divergences,
    one_minus_expected_max_probs,
    entropies_of_bma,
    entropies_of_fbar,
    one_minus_max_probs_of_bma,
    one_minus_max_probs_of_fbar,
    jensen_shannon_divergences,
    dempster_shafer_values,
    expected_variances_of_probs,
    expected_variances_of_logits,
):
    log_fbars[indices] = inference_dict["log_fbar"]
    log_bmas[indices] = inference_dict["log_bma"]
    gt_epistemics_bregman[indices] = inference_dict["expected_divergence"]

    time_forward_m.update(inference_dict["time_forward"], batch_size)

    expected_entropies[indices] = inference_dict["expected_entropy"]
    expected_entropies_plus_expected_divergences[indices] = (
        expected_entropies[indices] + gt_epistemics_bregman[indices]
    )
    one_minus_expected_max_probs[indices] = 1 - inference_dict["expected_max_prob"]
    entropies_of_bma[indices] = inference_dict["entropy_of_bma"]
    entropies_of_fbar[indices] = inference_dict["entropy_of_fbar"]
    one_minus_max_probs_of_bma[indices] = 1 - inference_dict["max_prob_of_bma"]
    one_minus_max_probs_of_fbar[indices] = 1 - inference_dict["max_prob_of_fbar"]
    jensen_shannon_divergences[indices] = inference_dict["jensen_shannon_divergence"]
    dempster_shafer_values[indices] = inference_dict["dempster_shafer_value"]
    expected_variances_of_probs[indices] = inference_dict["expected_variance_of_probs"]
    expected_variances_of_logits[indices] = inference_dict[
        "expected_variance_of_logits"
    ]


def update_losspred(inference_dict, indices, loss_values):
    loss_values[indices] = inference_dict["loss_value"]


def update_ddu(
    inference_dict,
    indices,
    gmm_neg_log_densities,
):
    gmm_neg_log_densities[indices] = inference_dict["gmm_neg_log_density"]


def update_corrpred(inference_dict, indices, error_probabilities):
    error_probabilities[indices] = inference_dict["error_probability"]


def update_duq(inference_dict, indices, duq_values):
    duq_values[indices] = inference_dict["duq_value"]


def update_mahalanobis(
    inference_dict,
    indices,
    mahalanobis_values,
):
    mahalanobis_values[indices] = inference_dict["mahalanobis_value"]


def update_hetclassnn(
    inference_dict,
    indices,
    expected_variances_of_internal_probs,
    expected_variances_of_internal_logits,
):
    expected_variances_of_internal_probs[indices] = inference_dict[
        "expected_variance_of_internal_probs"
    ]
    expected_variances_of_internal_logits[indices] = inference_dict[
        "expected_variance_of_internal_logits"
    ]
