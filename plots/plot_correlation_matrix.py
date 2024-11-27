"""Generates correlation matrices for various metrics across different methods."""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from tueplots import bundles
from utils import (
    CORRELATION_MATRIX_ESTIMATORS,
    DISTRIBUTIONAL_METHODS,
    ESTIMATOR_CONVERSION_DICT,
    ESTIMATORLESS_METRICS,
    EVIDENTIAL_METHODS,
    ID_TO_METHOD,
    ONLY_DISTRIBUTIONAL_ESTIMATORS,
    ONLY_NON_EVIDENTIAL_ESTIMATORS,
    setup_logging,
)
from wandb.apis.public.sweeps import Sweep

setup_logging()
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Process run results for plotting")
parser.add_argument("dataset", help="Dataset to use")

METRIC_DICT = {
    "auroc_hard_bma_correctness_original": "Correctness AUROC",
    "ece_hard_bma_correctness_original": "-ECE",
    "brier_score_hard_bma_correctness_original": "Correctness Brier",
    "log_prob_score_hard_bma_correctness_original": "Correctness Log Prob.",
    "rank_correlation_bregman_au": "Aleatoric Rank Corr.",
    "cumulative_hard_bma_abstinence_auc_original": "AUAC",
    "hard_bma_accuracy_original": "Accuracy",
    "auroc_oodness": "OOD AUROC",
}

ID_TO_METHOD_IMAGENET = {
    "znhyrrk6": "CE Baseline",
    "11ueh7cq": "Corr. Pred.",
    "k9myyurz": "DDU",
    "54kpysjy": "Deep Ens.",
    "4nr8lsd1": "GP",
    "t0uem6ob": "HET",
    "bryrtulr": "HetClassNN",
    "t1myokqo": "HET-XL",
    "42thx27s": "Laplace",
    "7flvihja": "Loss Pred.",
    "8a3palks": "Mahalanobis",
    "1pqijue2": "MC Dropout",
    "pipwlaae": "Shallow Ens.",
    "74rysdqf": "SNGP",
    "o04c996o": "SWAG",
    "jfnn98e3": "Temperature",
}

ID_PREFIX = "best_id_test"
MIXTURE_PREFIX_IMAGENET = "best_ood_test_varied_soft_imagenet_s2_mixed_soft_imagenet"
MIXTURE_PREFIX_CIFAR10 = "best_ood_test_varied_soft_cifar10_s2_mixed_soft_cifar10"
MIXTURE_PREFIX = {
    "imagenet": MIXTURE_PREFIX_IMAGENET,
    "cifar10": MIXTURE_PREFIX_CIFAR10,
}


def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    plt.rcParams.update(bundles.icml2024(family="serif", column="half", usetex=True))
    plt.rcParams["text.latex.preamble"] += r"\usepackage{amsmath} \usepackage{amsfonts}"


def process_run_data(
    run: wandb.apis.public.Run, prefix: str, suffix: str
) -> dict[str, list[float]]:
    """Processes data from a single Wandb run.

    Args:
        run: A Weights & Biases run object.
        prefix: The prefix for the metric keys.
        suffix: The ID of the metric being processed.

    Returns:
        A dictionary mapping estimator names to lists of their metrics.
    """

    def is_valid_key(stripped_key: str) -> bool:
        return (
            "mixed" not in stripped_key
            and "gt" not in stripped_key
            and (
                stripped_key in ESTIMATOR_CONVERSION_DICT
                or stripped_key in ESTIMATORLESS_METRICS
            )
        )

    metric_dict = defaultdict(list)

    for key in sorted(run.summary.keys()):
        if key.startswith(prefix) and key.endswith(suffix):
            stripped_key = key.removeprefix(f"{prefix}_").removesuffix(f"_{suffix}")
            if is_valid_key(stripped_key):
                metric_dict[stripped_key].append(run.summary[key])

    return metric_dict


def get_sweep_data(
    sweep: Sweep,
    method_name: str,
    metric_id: str,
    metric_name: str,
    mixture_prefix: str,
) -> dict[str, list[float]]:
    """Retrieves and processes data for a specific sweep.

    Args:
        sweep: The Weights & Biases Sweep object.
        method_name: The name of the method being processed.
        metric_id: The ID of the metric being processed.
        metric_name: The name of the metric being processed.
        mixture_prefix: Prefix for the mixture dataset.

    Returns:
        A dictionary mapping estimator names to lists of their values across all runs.
    """
    prefix = ID_PREFIX if metric_name != "OOD AUROC" else mixture_prefix
    metric_dict = defaultdict(list)

    for run in sweep.runs:
        if run.state != "finished":
            logger.info(f"Run {run.id} has not finished yet.")
            continue

        run_metric_dict = process_run_data(run, prefix, metric_id)

        for key, values in run_metric_dict.items():
            non_distributional_check = (
                method_name not in DISTRIBUTIONAL_METHODS
                and key in ONLY_DISTRIBUTIONAL_ESTIMATORS
            )
            evidential_check = (
                method_name in EVIDENTIAL_METHODS
                and key in ONLY_NON_EVIDENTIAL_ESTIMATORS
            )
            duq_check = method_name == "DUQ" and key == "dempster_shafer_values"
            if non_distributional_check or evidential_check or duq_check:
                continue
            metric_dict[key].extend(values)

    return metric_dict


def process_estimator_dict(
    estimator_dict: dict[str, list[float]], metric_name: str
) -> dict[str, float]:
    """Processes the estimator dictionary.

    Removes NaNs and applies necessary transformations.

    Args:
        estimator_dict: A dictionary mapping estimator names to lists of their values.
        metric_name: The name of the metric being processed.

    Returns:
        A dictionary mapping estimator names to their processed (mean) values.
    """
    for key in list(estimator_dict.keys()):
        if "NaN" in estimator_dict[key]:
            del estimator_dict[key]

            logging.info(f"NaN detected in {key}")
            continue

        estimator_dict[key] = np.mean(estimator_dict[key])

        if metric_name in {"-ECE", "-E-AURC"}:
            estimator_dict[key] *= -1

    return estimator_dict


def get_estimator_value(
    estimator_dict: dict[str, float], distributional_estimator: str
) -> float:
    """Determines the appropriate estimator and returns its value.

    Args:
        estimator_dict: A dictionary mapping estimator names to their values.
        distributional_estimator: The name of the distributional estimator being used.

    Returns:
        The value of the appropriate estimator.
    """
    if len(estimator_dict) > 1:
        estimator = distributional_estimator
    else:
        estimator = next(iter(estimator_dict.keys()))

    return estimator_dict[estimator]


def calculate_performance_matrix(
    api: wandb.Api, id_to_method: dict[str, str], mixture_prefix: str
) -> np.ndarray:
    """Calculates the performance matrix for all metrics and methods.

    Args:
        api: The Weights & Biases API object.
        id_to_method: Mapping from sweep IDs to method names.
        mixture_prefix: Prefix for the mixture dataset.

    Returns:
        A 2D numpy array representing the performance matrix.
    """
    performance_matrix = [[] for _ in METRIC_DICT]

    for method_id, method_name in tqdm(id_to_method.items()):
        estimators = (
            ["error_probabilities"]
            if method_name == "Corr. Pred."
            else ["one_minus_max_probs_of_bma"]
            if method_name not in DISTRIBUTIONAL_METHODS
            else CORRELATION_MATRIX_ESTIMATORS
        )
        for distributional_estimator in estimators:
            sweep = api.sweep(f"bmucsanyi/untangle/{method_id}")
            for i, (metric_id, metric_name) in enumerate(METRIC_DICT.items()):
                estimator_dict = get_sweep_data(
                    sweep=sweep,
                    method_name=method_name,
                    metric_id=metric_id,
                    metric_name=metric_name,
                    mixture_prefix=mixture_prefix,
                )
                processed_dict = process_estimator_dict(
                    estimator_dict=estimator_dict, metric_name=metric_name
                )
                performance_matrix[i].append(
                    get_estimator_value(
                        estimator_dict=processed_dict,
                        distributional_estimator=distributional_estimator,
                    )
                )

    return np.array(performance_matrix)


def calculate_correlation_matrices(
    performance_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates Spearman and Pearson correlation matrices.

    Args:
        performance_matrix: A 2D numpy array representing the performance matrix.

    Returns:
        A tuple containing the Spearman and Pearson correlation matrices.
    """
    num_metrics = performance_matrix.shape[0]
    correlation_matrix_spearman = np.zeros((num_metrics, num_metrics))
    correlation_matrix_pearson = np.zeros((num_metrics, num_metrics))

    for i in range(num_metrics):
        for j in range(num_metrics):
            perf_i, perf_j = performance_matrix[i, :], performance_matrix[j, :]
            correlation_matrix_spearman[i, j] = spearmanr(perf_i, perf_j)[0]
            correlation_matrix_pearson[i, j] = pearsonr(perf_i, perf_j)[0]

    return correlation_matrix_spearman, correlation_matrix_pearson


def plot_correlation_matrix(
    correlation_matrix: np.ndarray, name: str, save_path: Path
) -> None:
    """Plots and saves a correlation matrix.

    Args:
        correlation_matrix: A 2D numpy array representing the correlation matrix.
        name: The name of the correlation type (e.g., "spearman" or "pearson").
        save_path: The path where the plot should be saved.
    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("coolwarm")
    cax = ax.imshow(
        correlation_matrix, interpolation="nearest", cmap=cmap, vmin=-1, vmax=1
    )

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(width=0.1)
    cbar.set_ticks([-0.983, 0, 1.01])
    cbar.set_ticklabels(["-1", "0", "1"])

    ax.set_xticks(np.arange(len(METRIC_DICT)))
    ax.set_yticks(np.arange(len(METRIC_DICT)))
    ax.set_xticklabels(METRIC_DICT.values())
    ax.set_yticklabels(METRIC_DICT.values())

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(METRIC_DICT)):
        for j in range(len(METRIC_DICT)):
            ax.text(
                j,
                i,
                round(correlation_matrix[i, j], 2),
                ha="center",
                va="center",
                color="black",
                fontsize=5,
            )

    ax.spines[["right", "top"]].set_visible(False)
    plt.savefig(save_path / f"correlation_matrix_{name}.pdf")
    plt.close()


def main() -> None:
    """Main function to process metrics and generate correlation matrices."""
    args = parser.parse_args()
    setup_plot_style()
    api = wandb.Api()

    id_to_method = ID_TO_METHOD[args.dataset]
    mixture_prefix = MIXTURE_PREFIX[args.dataset]

    save_path = Path(f"results/{args.dataset}/correlation_matrix")
    save_path.mkdir(parents=True, exist_ok=True)

    performance_matrix = calculate_performance_matrix(
        api=api, id_to_method=id_to_method, mixture_prefix=mixture_prefix
    )
    correlation_matrix_spearman, correlation_matrix_pearson = (
        calculate_correlation_matrices(performance_matrix=performance_matrix)
    )

    plot_correlation_matrix(
        correlation_matrix=correlation_matrix_spearman,
        name="spearman",
        save_path=save_path,
    )
    plot_correlation_matrix(
        correlation_matrix=correlation_matrix_pearson,
        name="pearson",
        save_path=save_path,
    )


if __name__ == "__main__":
    main()