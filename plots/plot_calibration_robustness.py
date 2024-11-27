"""Generates plots to compare ECE generalization across different methods."""

import argparse
import operator
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm
from tueplots import bundles
from utils import (
    COLOR_ERROR_BAR,
    ESTIMATOR_CONVERSION_DICT,
    ESTIMATORLESS_METRICS,
    lighten_color,
)
from wandb.apis.public.sweeps import Sweep

parser = argparse.ArgumentParser(description="Process run results for plotting")
parser.add_argument("dataset", help="Dataset to use")

ID_TO_METHOD_IMAGENET = {
    "idk8ctw8": "Laplace",
    "pipwlaae": "Shallow Ens.",
    "gl6qgpv6": "EDL",
    "znhyrrk6": "CE Baseline",
}

ID_TO_METHOD_CIFAR10 = {
    "nnle8epz": "Laplace",
    "lcvixgvo": "Shallow Ens.",
    "vuel80q8": "EDL",
    "uo3gu133": "CE Baseline",
}

ID_TO_METHOD = {"imagenet": ID_TO_METHOD_IMAGENET, "cifar10": ID_TO_METHOD_CIFAR10}

DATASET_PREFIX_LIST_IMAGENET = [
    "best_id_test",
    "best_ood_test_varied_soft_imagenet_s1",
    "best_ood_test_varied_soft_imagenet_s2",
    "best_ood_test_varied_soft_imagenet_s3",
    "best_ood_test_varied_soft_imagenet_s4",
    "best_ood_test_varied_soft_imagenet_s5",
]

DATASET_PREFIX_LIST_CIFAR10 = [
    "best_id_test",
    "best_ood_test_varied_soft_cifar10_s1",
    "best_ood_test_varied_soft_cifar10_s2",
    "best_ood_test_varied_soft_cifar10_s3",
    "best_ood_test_varied_soft_cifar10_s4",
    "best_ood_test_varied_soft_cifar10_s5",
]

DATASET_PREFIX_LIST = {
    "imagenet": DATASET_PREFIX_LIST_IMAGENET,
    "cifar10": DATASET_PREFIX_LIST_CIFAR10,
}


def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    config = bundles.neurips2024()
    config["figure.figsize"] = (2.64, 1.176)
    plt.rcParams.update(config)
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )


def process_sweep_data(
    sweep: Sweep, suffix: str, dataset_prefix_list: list[str]
) -> list[dict[str, list[float]]]:
    """Processes data from a W&B sweep for ECE calculations.

    Args:
        sweep: W&B sweep object.
        suffix: Suffix for metric keys.
        dataset_prefix_list: List of dataset prefixes.

    Returns:
        List of dictionaries containing ECE values for each dataset and estimator.
    """

    def is_valid_key(stripped_key: str) -> bool:
        return "mixed" not in stripped_key and (
            stripped_key in ESTIMATOR_CONVERSION_DICT
            or stripped_key in ESTIMATORLESS_METRICS
        )

    ece_list = []
    for prefix in dataset_prefix_list:
        metric_dict = defaultdict(list)
        for run in sweep.runs:
            if run.state != "finished":
                continue
            for key in sorted(run.summary.keys()):
                if key.startswith(prefix) and key.endswith(suffix):
                    processed_key = key.removeprefix(f"{prefix}_").removesuffix(
                        f"_{suffix}"
                    )
                    if is_valid_key(processed_key):
                        metric_dict[processed_key].append(run.summary[key])
        ece_list.append(metric_dict)

    return ece_list


def calculate_ece_statistics(
    ece_list: list[dict[str, list[float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates ECE statistics (mean, min, max) for each dataset.

    Args:
        ece_list: List of dictionaries containing ECE values.

    Returns:
        Tuple of arrays containing means, minimums, and maximums of ECE values.
    """
    means = np.empty((6,))
    mins = np.empty((6,))
    maxs = np.empty((6,))

    for i, ece_dict in enumerate(ece_list):
        means_across_keys = {
            key: np.mean(ece_dict[key]) for key in ece_dict if "gt" not in key
        }
        best_estimator = min(means_across_keys.items(), key=operator.itemgetter(1))[0]

        values = ece_dict[best_estimator]
        means[i] = np.mean(values)
        mins[i] = np.min(values)
        maxs[i] = np.max(values)

    return means, mins, maxs


def plot_ece_bars(
    ax: plt.Axes,
    means: np.ndarray,
    error_bars: list[np.ndarray],
    method_name: str,
    color: tuple,
    bar_positions: np.ndarray,
    bar_width: float,
) -> None:
    """Plots ECE bars for a single method.

    Args:
        ax: Matplotlib axes object.
        means: Array of mean ECE values.
        error_bars: List of arrays containing lower and upper error bar values.
        method_name: Name of the method being plotted.
        color: Color to use for the bars.
        bar_positions: Positions of the bars on the x-axis.
        bar_width: Width of the bars.
    """
    ax.bar(
        bar_positions,
        means[:2],
        bar_width,
        color=color,
        label=method_name,
        zorder=2,
    )

    ax.errorbar(
        bar_positions,
        means[:2],
        yerr=error_bars,
        fmt="none",
        capsize=5,
        ecolor=COLOR_ERROR_BAR,
        elinewidth=1,
        zorder=2,
    )


def setup_plot_styling(ax: plt.Axes) -> None:
    """Sets up the styling for the plot.

    Args:
        ax: Matplotlib axes object.
    """
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel(r"ECE $\downarrow$")
    ax.set_ylim([0, 0.15])
    ax.legend(
        frameon=False,
        ncols=1,
        fontsize="xx-small",
        loc="upper left",
        bbox_to_anchor=(0, 1.1),
    )

    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["ID", "OOD Severity 1"])
    ax.grid(axis="y", zorder=1, linewidth=0.5)


def main() -> None:
    """Main function to process ECE data and generate comparison plot."""
    setup_plot_style()
    args = parser.parse_args()
    api = wandb.Api()
    id_to_method = ID_TO_METHOD[args.dataset]
    dataset_prefix_list = DATASET_PREFIX_LIST[args.dataset]

    _, ax = plt.subplots()

    bar_width = 0.15
    num_methods = len(id_to_method)

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lightened_colors = [lighten_color(color, amount=0.15) for color in default_colors]

    for j, (method_id, method_name) in enumerate(tqdm(id_to_method.items())):
        sweep = api.sweep(f"bmucsanyi/untangle/{method_id}")
        suffix_ece = "ece_hard_bma_correctness_original"

        ece_list = process_sweep_data(sweep, suffix_ece, dataset_prefix_list)
        means, mins, maxs = calculate_ece_statistics(ece_list)

        lower_errors = np.maximum(0, means - mins)
        upper_errors = np.maximum(0, maxs - means)
        error_bars = [lower_errors[:2], upper_errors[:2]]

        bar_positions = (
            np.arange(2) - (num_methods * bar_width / 2) + j * bar_width + bar_width / 2
        )

        plot_ece_bars(
            ax,
            means,
            error_bars,
            method_name,
            lightened_colors[j],
            bar_positions,
            bar_width,
        )

    save_path = Path(f"results/{args.dataset}/calibration_robustness")
    save_path.mkdir(exist_ok=True)
    setup_plot_styling(ax)
    plt.savefig(save_path / "calibration_robustness.pdf")
    plt.clf()


if __name__ == "__main__":
    main()
