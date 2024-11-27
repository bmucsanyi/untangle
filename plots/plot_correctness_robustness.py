"""Generates plots to show the robustness of methods on correctness prediction."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm
from tueplots import bundles
from utils import ID_TO_METHOD

parser = argparse.ArgumentParser(description="Process run results for plotting")
parser.add_argument("dataset", help="Dataset to use")

DATASET_PAIR_LIST_IMAGENET = [
    ("best_id_test", None),
    (
        "best_ood_test_varied_soft_imagenet_s1",
        "best_ood_test_varied_soft_imagenet_s1_mixed_soft_imagenet",
    ),
    (
        "best_ood_test_varied_soft_imagenet_s2",
        "best_ood_test_varied_soft_imagenet_s2_mixed_soft_imagenet",
    ),
    (
        "best_ood_test_varied_soft_imagenet_s3",
        "best_ood_test_varied_soft_imagenet_s3_mixed_soft_imagenet",
    ),
    (
        "best_ood_test_varied_soft_imagenet_s4",
        "best_ood_test_varied_soft_imagenet_s4_mixed_soft_imagenet",
    ),
    (
        "best_ood_test_varied_soft_imagenet_s5",
        "best_ood_test_varied_soft_imagenet_s5_mixed_soft_imagenet",
    ),
]

DATASET_PAIR_LIST_CIFAR10 = [
    ("best_id_test", None),
    (
        "best_ood_test_varied_soft_cifar10_s1",
        "best_ood_test_varied_soft_cifar10_s1_mixed_soft_cifar10",
    ),
    (
        "best_ood_test_varied_soft_cifar10_s2",
        "best_ood_test_varied_soft_cifar10_s2_mixed_soft_cifar10",
    ),
    (
        "best_ood_test_varied_soft_cifar10_s3",
        "best_ood_test_varied_soft_cifar10_s3_mixed_soft_cifar10",
    ),
    (
        "best_ood_test_varied_soft_cifar10_s4",
        "best_ood_test_varied_soft_cifar10_s4_mixed_soft_cifar10",
    ),
    (
        "best_ood_test_varied_soft_cifar10_s5",
        "best_ood_test_varied_soft_cifar10_s5_mixed_soft_cifar10",
    ),
]

DATASET_PAIR_LIST = {
    "imagenet": DATASET_PAIR_LIST_IMAGENET,
    "cifar10": DATASET_PAIR_LIST_CIFAR10,
}


def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    config = bundles.neurips2024()
    config["figure.figsize"] = (2.64, 1.3)  # (2.64, 2.64 / 1.618)
    plt.rcParams.update(config)
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )


def get_key_auroc(method_name: str) -> str:
    """Determines the appropriate key for AUROC based on the method name.

    Args:
        method_name: Name of the method.

    Returns:
        Key for AUROC.
    """
    if method_name == "Corr. Pred.":
        return "_error_probabilities_"
    if method_name == "Loss Pred.":
        return "_loss_values_"
    if method_name == "Mahalanobis":
        return "_mahalanobis_values_"
    return "_one_minus_max_probs_of_bma_"


def process_run_data(
    run: wandb.apis.public.Run, prefix: str, key_auroc: str, key_accuracy: str
) -> tuple[float, float, float]:
    """Processes data from a single W&B run.

    Args:
        run: W&B run object.
        prefix: Prefix for the metric keys.
        key_auroc: Key for AUROC.
        key_accuracy: Key for accuracy.

    Returns:
        AUROC correctness, AUC abstinence, and accuracy.
    """
    suffix_correctness = "auroc_hard_bma_correctness_original"
    suffix_abstinence = "cumulative_hard_bma_abstinence_auc_original"

    auroc_correctness = run.summary[prefix + key_auroc + suffix_correctness]
    auc_abstinence = run.summary[prefix + key_auroc + suffix_abstinence]
    accuracy = run.summary[prefix + key_accuracy]

    return auroc_correctness, auc_abstinence, accuracy


def collect_method_data(
    api: wandb.Api,
    method_id: str,
    method_name: str,
    dataset_pair_list: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collects data for a specific method across all severities and runs.

    Args:
        api: W&B API object.
        method_id: ID of the method.
        method_name: Name of the method.
        dataset_pair_list: List of pairs of corresponding datasets.

    Returns:
        Arrays for AUROC correctness, AUC abstinence, and accuracy.
    """
    sweep = api.sweep(f"bmucsanyi/untangle/{method_id}")
    num_successful_runs = sum(1 for run in sweep.runs if run.state == "finished")

    auroc_correctness_matrix = np.zeros((num_successful_runs, 6))
    auc_abstinence_matrix = np.zeros((num_successful_runs, 6))
    accuracy_matrix = np.zeros((num_successful_runs, 6))

    key_auroc = get_key_auroc(method_name)
    key_accuracy = "_hard_bma_accuracy_original"

    for j, (prefix_normal, _) in enumerate(dataset_pair_list):
        i = 0
        for run in sweep.runs:
            if run.state != "finished":
                continue
            auroc_correctness, auc_abstinence, accuracy = process_run_data(
                run, prefix_normal, key_auroc, key_accuracy
            )
            auroc_correctness_matrix[i, j] = auroc_correctness
            auc_abstinence_matrix[i, j] = auc_abstinence
            accuracy_matrix[i, j] = accuracy
            i += 1

    return auroc_correctness_matrix, auc_abstinence_matrix, accuracy_matrix


def calculate_metrics(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates mean, min, and max values for a given metric matrix.

    Args:
        matrix: Matrix containing metric values.

    Returns:
        Mean, min, and max values.
    """
    means = np.mean(matrix, axis=0)
    min_values = np.min(matrix, axis=0)
    max_values = np.max(matrix, axis=0)
    return means, min_values, max_values


def calculate_error_bars(
    means: np.ndarray, min_values: np.ndarray, max_values: np.ndarray
) -> list[np.ndarray]:
    """Calculates error bar values for plotting.

    Args:
        means: Mean values.
        min_values: Minimum values.
        max_values: Maximum values.

    Returns:
        Lower and upper error bar values.
    """
    lower_errors = np.maximum(0, means - min_values)
    upper_errors = np.maximum(0, max_values - means)
    return [lower_errors, upper_errors]


def plot_metric(
    severities: np.ndarray,
    means: np.ndarray,
    error_bars: list[np.ndarray],
    label: str,
    color: str,
    alpha: float = 1.0,
) -> None:
    """Plots a single metric with error bars.

    Args:
        severities: X-axis values (severity levels).
        means: Y-axis values (metric means).
        error_bars: Error bar values.
        label: Label for the plot legend.
        color: Color for the plot line and markers.
        alpha: Alpha value for transparency. Defaults to 1.0.
    """
    plt.errorbar(
        severities,
        means,
        yerr=error_bars,
        fmt="-o" if alpha == 1.0 else "--o",
        label=label if alpha == 1.0 else None,
        color=color,
        markersize=3,
        ecolor=np.array([105.0, 109.0, 113.0]) / 255.0,
        elinewidth=1,
        capsize=5,
        alpha=alpha,
    )


def plot_method_results(
    auroc_correctness_matrix: np.ndarray,
    auc_abstinence_matrix: np.ndarray,
    accuracy_matrix: np.ndarray,
    save_path: str,
) -> None:
    """Plots the results for a single method and saves the figure.

    Args:
        auroc_correctness_matrix: Matrix of AUROC correctness values.
        auc_abstinence_matrix: Matrix of AUC abstinence values.
        accuracy_matrix: Matrix of accuracy values.
        save_path: Path to save the resulting plot.
    """
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    severities = np.arange(6)

    means_auroc, min_auroc, max_auroc = calculate_metrics(auroc_correctness_matrix)
    means_auac, min_auac, max_auac = calculate_metrics(auc_abstinence_matrix)
    means_accuracy, min_accuracy, max_accuracy = calculate_metrics(accuracy_matrix)

    error_bars_auroc = calculate_error_bars(means_auroc, min_auroc, max_auroc)
    error_bars_auac = calculate_error_bars(means_auac, min_auac, max_auac)
    error_bars_accuracy = calculate_error_bars(
        means_accuracy, min_accuracy, max_accuracy
    )

    plt.gca().spines[["right", "top"]].set_visible(False)

    plot_metric(
        severities,
        2 * (means_auroc - 0.5),
        [2 * error for error in error_bars_auroc],
        "AUROC Correctness",
        default_colors[0],
    )
    plot_metric(
        severities, means_auroc, error_bars_auroc, None, default_colors[0], alpha=0.2
    )

    plot_metric(
        severities,
        (1 - means_accuracy) ** (-1) * (means_auac - means_accuracy),
        [(1 - means_accuracy) ** (-1) * error for error in error_bars_auac],
        "AUAC",
        default_colors[1],
    )
    plot_metric(
        severities, means_auac, error_bars_auac, None, default_colors[1], alpha=0.2
    )

    plot_metric(
        severities,
        1000 / 999 * (means_accuracy - 0.001),
        [1000 / 999 * error for error in error_bars_accuracy],
        "Accuracy",
        default_colors[2],
    )
    plot_metric(
        severities,
        means_accuracy,
        error_bars_accuracy,
        None,
        default_colors[2],
        alpha=0.2,
    )

    plt.xlabel("Severity Level")
    plt.ylabel(r"Metric Values $\uparrow$")
    plt.ylim(0, 1)

    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [h[0] for h in handles]

    plt.legend(
        handles,
        labels,
        frameon=False,
        fontsize="x-small",
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
    )
    plt.grid(True, linewidth=0.5)
    plt.savefig(save_path)
    plt.clf()


def main() -> None:
    """Main function to process metrics and generate plots."""
    args = parser.parse_args()
    setup_plot_style()
    api = wandb.Api()

    id_to_method = ID_TO_METHOD[args.dataset]
    dataset_pair_list = DATASET_PAIR_LIST[args.dataset]

    save_path = Path(f"results/{args.dataset}/correctness_robustness")
    save_path.mkdir(exist_ok=True)

    for method_id, method_name in tqdm(id_to_method.items()):
        auroc_correctness_matrix, auc_abstinence_matrix, accuracy_matrix = (
            collect_method_data(api, method_id, method_name, dataset_pair_list)
        )

        method_save_path = (
            save_path / f"{method_name.replace('.', '').replace(' ', '_').lower()}.pdf"
        )
        plot_method_results(
            auroc_correctness_matrix,
            auc_abstinence_matrix,
            accuracy_matrix,
            method_save_path,
        )


if __name__ == "__main__":
    main()
