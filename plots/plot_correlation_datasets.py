"""Calculates the correlation of method rankings across ImageNet and CIFAR-10."""

import operator
from typing import Any

import numpy as np
import pandas as pd
import wandb
from scipy.stats import spearmanr
from tqdm import tqdm
from utils import CONSTRAINED_METRICS, ESTIMATOR_CONVERSION_DICT, ESTIMATORLESS_METRICS
from wandb.wandb_run import Run

METHOD_TO_IDS = {
    "GP": ("4nr8lsd1", "3h3gyzxj"),
    "HET-XL": ("t1myokqo", "xaz96x6d"),
    "CE Baseline": ("znhyrrk6", "uo3gu133"),
    "MC Dropout": ("1pqijue2", "u1ozluxv"),
    "SNGP": ("74rysdqf", "mhu72izt"),
    "Shallow Ens.": ("pipwlaae", "lcvixgvo"),
    "Loss Pred.": ("7flvihja", "y5mljm78"),
    "Corr. Pred.": ("11ueh7cq", "sgvtuzo5"),
    "Deep Ens.": ("54kpysjy", "bn1hsbqz"),
    "Laplace": ("42thx27s", "nnle8epz"),
    "Mahalanobis": ("8a3palks", "h0m0bybl"),
    "Temperature": ("jfnn98e3", "lh04ospw"),
    "DDU": ("k9myyurz", "8apeaj9a"),
    "HET": ("t0uem6ob", "k5elnty1"),
    "EDL": ("gl6qgpv6", "vuel80q8"),
    "PostNet": ("zm0o0mo9", "8bqhu92u"),
    "SWAG": ("o04c996o", "zsiqsl6u"),
    "HetClassNN": ("bryrtulr", "wj1sesqf"),
}

METRIC_DICT = {
    "auroc_hard_bma_correctness_original": "Correctness AUROC",
    "ece_hard_bma_correctness_original": "ECE",
    "brier_score_hard_bma_correctness_original": "Correctness Brier",
    "log_prob_score_hard_bma_correctness_original": "Correctness Log Prob.",
    "hard_bma_raulc_original": "rAULC",
    "hard_bma_eaurc_original": "E-AURC",
    "cumulative_hard_bma_abstinence_auc_original": "AUAC",
    "hard_bma_accuracy_original": "Accuracy",
    "log_prob_score_hard_bma_aleatoric_original": "Aleatoric Log Prob.",
    "brier_score_hard_bma_aleatoric_original": "Aleatoric Brier",
    "rank_correlation_bregman_au": "Aleatoric Rank Corr.",
    "auroc_multiple_labels": "Aleatoric AUROC",
    "auroc_oodness": "OOD AUROC",
}


def get_best_estimator(
    method_name: str, metric_id: str, metric: dict[str, float]
) -> str:
    """Determines the best estimator for the method.

    Args:
        method_name: Name of the method being analyzed.
        metric_id: The metric identifier.
        metric: Dictionary of metric values.

    Returns:
        The best estimator key for this method and metric.
    """
    method_metric_to_estimator = {
        ("Corr. Pred.", None): "error_probabilities",
        ("Loss Pred.", None): "risk_values",
        ("Mahalanobis", None): "mahalanobis_values",
        ("DDU", "auroc_oodness"): "gmm_neg_log_densities",
    }

    if metric_id not in ESTIMATORLESS_METRICS and metric_id not in CONSTRAINED_METRICS:
        # Try with specific metric first
        if (method_name, metric_id) in method_metric_to_estimator:
            return method_metric_to_estimator[method_name, metric_id]
        # Try with None (metric-independent) case
        if (method_name, None) in method_metric_to_estimator:
            return method_metric_to_estimator[method_name, None]

    return max(metric.items(), key=operator.itemgetter(1))[0]


def process_run_data(
    run: Run,
    prefix: str,
    metric_id: str,
) -> dict[str, list[float]]:
    """Processes metric data from a single run.

    Args:
        run: The wandb run object.
        prefix: The metric prefix to match.
        metric_id: The metric identifier.

    Returns:
        Dictionary mapping estimator names to metric values.
    """
    metric = {}

    for key in sorted(run.summary.keys()):
        if key.startswith(prefix) and key.endswith(metric_id):
            stripped_key = key.replace(f"{prefix}_", "").replace(f"_{metric_id}", "")

            if (
                "mixed" in stripped_key
                or "gt" in stripped_key
                or run.summary[key] == "NaN"
                or not (
                    stripped_key in ESTIMATOR_CONVERSION_DICT
                    or stripped_key in ESTIMATORLESS_METRICS
                )
            ):
                continue

            if stripped_key not in metric:
                metric[stripped_key] = [run.summary[key]]
            else:
                metric[stripped_key].append(run.summary[key])

    return metric


def aggregate_metrics(
    metric: dict[str, list[float]],
    metric_name: str,
) -> dict[str, float]:
    """Aggregates metrics across runs by taking the mean and applying transformations.

    Args:
        metric: Dictionary of metric values.
        metric_name: Name of the metric being processed.

    Returns:
        Dictionary of aggregated metric values.
    """
    aggregated = {}
    for key, values in metric.items():
        mean_value = np.mean(values)
        if metric_name in {"ECE", "E-AURC"}:
            mean_value *= -1
        aggregated[key] = mean_value

    if "brier_score_hard_bma_aleatoric_original" in aggregated:
        return {
            "brier_score_hard_bma_aleatoric_original": aggregated[
                "brier_score_hard_bma_aleatoric_original"
            ]
        }

    return aggregated


def process_dataset_metrics(
    sweep: Any,
    prefix: str,
    metric_id: str,
    metric_name: str,
) -> dict[str, float]:
    """Processes metrics for a single dataset.

    Args:
        sweep: The wandb sweep object.
        prefix: The metric prefix.
        metric_id: The metric identifier.
        metric_name: Name of the metric.
        method_name: Name of the method.

    Returns:
        Dictionary of processed metrics.
    """
    metric = {}

    for run in sweep.runs:
        if run.state != "finished":
            continue

        run_metrics = process_run_data(run, prefix, metric_id)
        for key, values in run_metrics.items():
            if key not in metric:
                metric[key] = values
            else:
                metric[key].extend(values)

    return aggregate_metrics(metric, metric_name)


def main() -> None:
    """Main function to analyze metrics across methods and datasets."""
    api = wandb.Api()

    performance_matrix_imagenet = np.zeros((len(METRIC_DICT), len(METHOD_TO_IDS)))
    performance_matrix_cifar = np.zeros((len(METRIC_DICT), len(METHOD_TO_IDS)))
    correlation_vector = np.zeros(len(METRIC_DICT))

    id_prefix = "best_id_test"
    mixture_prefix_cifar = "best_ood_test_varied_soft_cifar10_s2_mixed_soft_cifar10"
    mixture_prefix_imagenet = (
        "best_ood_test_varied_soft_imagenet_s2_mixed_soft_imagenet"
    )

    metric_names = []

    for j, (method_name, (imagenet_id, cifar_id)) in enumerate(
        tqdm(METHOD_TO_IDS.items())
    ):
        sweep_imagenet = api.sweep(f"bmucsanyi/untangle/{imagenet_id}")
        sweep_cifar = api.sweep(f"bmucsanyi/untangle/{cifar_id}")

        for i, (metric_id, metric_name) in enumerate(METRIC_DICT.items()):
            if j == 0:
                metric_names.append(metric_name)

            for sweep, performance_matrix, prefix in [
                (sweep_imagenet, performance_matrix_imagenet, mixture_prefix_imagenet),
                (sweep_cifar, performance_matrix_cifar, mixture_prefix_cifar),
            ]:
                prefix = id_prefix if metric_name != "OOD AUROC" else prefix

                metric = process_dataset_metrics(sweep, prefix, metric_id, metric_name)

                if metric:
                    best_key = get_best_estimator(method_name, metric_id, metric)
                    try:
                        performance_matrix[i, j] = metric[best_key]
                    except KeyError:
                        print(best_key, metric_id)

    print("Perf matrix ImageNet:", performance_matrix_imagenet)
    print("Perf matrix CIFAR-10:", performance_matrix_cifar)

    best_methods_imagenet = np.argmax(performance_matrix_imagenet, axis=-1)
    best_methods_cifar = np.argmax(performance_matrix_cifar, axis=-1)

    print("Best methods on ImageNet:", best_methods_imagenet)
    print("Best methods on CIFAR-10:", best_methods_cifar)

    for i in range(len(METRIC_DICT)):
        perf_imagenet = performance_matrix_imagenet[i, :]
        perf_cifar = performance_matrix_cifar[i, :]
        correlation_vector[i] = spearmanr(perf_imagenet, perf_cifar)[0]

    df = pd.DataFrame({"Metric": metric_names, "Rank Corr.": correlation_vector})
    print(df)


if __name__ == "__main__":
    main()
