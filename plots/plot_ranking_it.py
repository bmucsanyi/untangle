"""Plots rankings of methods as bar plots."""

import argparse
import operator
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
from tueplots import bundles
from utils import (
    COLOR_BASELINE,
    COLOR_DETERMINISTIC,
    COLOR_DISTRIBUTIONAL,
    COLOR_ERROR_BAR,
    COLOR_ESTIMATE,
    COLOR_GT,
    DATASET_PREFIX_LIST,
    DISTRIBUTIONAL_METHODS,
    ESTIMATOR_CONVERSION_DICT,
    ESTIMATORLESS_METRICS,
    EVIDENTIAL_METHODS,
    ID_TO_METHOD,
    ONLY_DISTRIBUTIONAL_ESTIMATORS,
    ONLY_NON_EVIDENTIAL_ESTIMATORS,
)
from wandb.wandb_run import Run

parser = argparse.ArgumentParser(description="Process run results for plotting")
parser.add_argument("dataset", help="Dataset to use")
parser.add_argument("ylabel", help="Label of the y axis")
parser.add_argument("metric", help="Metric to be used in the analysis")
parser.add_argument("--y-min", type=float, default=None, help="Minimum y value")
parser.add_argument("--y-max", type=float, default=None, help="Maximum y value")
parser.add_argument(
    "--labels-to-offset",
    nargs="*",
    default=[],
    help="List of labels that require y-offset adjustments",
)
parser.add_argument(
    "--offset-values",
    nargs="*",
    type=float,
    default=[],
    help="List of y-offset values corresponding to the labels",
)
parser.add_argument(
    "--cross", action="store_true", help="Whether to use cross-evaluation"
)
parser.add_argument(
    "--decreasing", action="store_true", help="Whether the metric is decreasing"
)


def setup_plot_style() -> None:
    """Sets up the plot style using tueplots and custom configurations."""
    config = bundles.neurips2024()
    config["figure.figsize"] = (2.64, 1.08)  # (2.64, 0.9)
    plt.rcParams.update(config)
    plt.rcParams["text.latex.preamble"] += (
        r"\usepackage{amsmath} \usepackage{amsfonts} \usepackage{bm}"
    )


def plot_single_method(
    ylabel: str,
    metric_dict: dict[str, list[float]],
    save_path: Path,
    y_min: float,
    y_max: float,
    decreasing: bool,
) -> None:
    """Plots the bar chart of a single method with all estimators.

    Args:
        ylabel: The y-axis label.
        metric_dict: The method's metrics to plot w.r.t different estimators.
        save_path: The path to save the plot.
        y_min: The minimum y-axis value.
        y_max: The maximum y-axis value.
        decreasing: Whether the metric is decreasing.
    """
    means = [np.mean(values) for values in metric_dict.values()]
    mins = [np.min(values) for values in metric_dict.values()]
    maxs = [np.max(values) for values in metric_dict.values()]
    error_bars = [
        (mean - min_val, max_val - mean)
        for min_val, mean, max_val in zip(mins, means, maxs, strict=False)
    ]
    labels = list(metric_dict.keys())

    # Combine lists and sort by `means` in decreasing order
    combined = sorted(
        zip(labels, means, error_bars, strict=False),
        key=operator.itemgetter(1),
        reverse=True,
    )
    labels, means, error_bars = zip(*combined, strict=False)

    _, ax = plt.subplots()

    # Set styling
    ax.grid(axis="y", which="both", zorder=1, linewidth=0.5)
    multiplier = 2 if "Corr" in ylabel or "rAULC" in ylabel else 1
    ax.yaxis.set_major_locator(MultipleLocator(0.1 * multiplier))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05 * multiplier))
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel(
        ylabel=ylabel + (r" $\uparrow$" if not decreasing else r" $\downarrow$")
    )
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)
    ax.set_ylim(bottom=y_min, top=y_max)

    # Plot bars with error bars
    bars = ax.bar(labels, means, zorder=2)
    ax.errorbar(
        labels,
        means,
        yerr=np.array(error_bars).T,
        fmt="none",
        ecolor=COLOR_ERROR_BAR,
        elinewidth=1,
        capsize=4,
        markeredgewidth=0.5,
        zorder=3,
    )

    # Add labels to bars and set bar colors
    for bar, label in zip(bars, labels, strict=False):
        y_offset = 0.005 * 2 * (y_max - y_min)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_min + y_offset,
            ESTIMATOR_CONVERSION_DICT.get(label, "No estimator"),
            ha="center",
            va="bottom",
            rotation="vertical",
            zorder=4,
            fontsize=6,
        )

        bar.set_color(COLOR_GT if "gt" in label else COLOR_ESTIMATE)

    # Add legend for bar colors
    legend_handles = [
        Patch(facecolor=COLOR_ESTIMATE, label="Estimate"),
        Patch(facecolor=COLOR_GT, label="Ground Truth"),
    ]
    ax.legend(
        frameon=False,
        handles=legend_handles,
        loc="upper right",
        fontsize="small",
        handlelength=1,
        ncol=2,
        bbox_to_anchor=(1, 1.13),
    )

    # Save and close plot
    plt.savefig(save_path)
    plt.close()


def plot_all_methods(
    ylabel: str,
    best_metric_dict_means: dict[str, float],
    best_metric_dict_mins_maxs: dict[str, tuple[float, float]],
    save_path: Path,
    y_min: float,
    y_max: float,
    decreasing: bool,
    labels_to_offset: list[str],
    offset_values: list[float],
) -> None:
    """Plots the bar chart of all methods with their best estimator.

    Args:
        ylabel: The y-axis label.
        best_metric_dict_means: The best metric values for each method.
        best_metric_dict_mins_maxs: The min and max values for each method.
        save_path: The path to save the plot.
        y_min: The minimum y-axis value.
        y_max: The maximum y-axis value.
        decreasing: Whether the metric is decreasing.
        labels_to_offset: Labels that require y-offset adjustments.
        offset_values: Y-offset values corresponding to the labels.
        only_distributional: Whether to only plot distributional methods.
    """
    labels, best_values = zip(*best_metric_dict_means.items(), strict=False)
    error_bars = [
        (best - min_val, max_val - best)
        for (min_val, max_val), best in zip(
            best_metric_dict_mins_maxs.values(), best_values, strict=False
        )
    ]

    # Combine lists and sort by `means` in decreasing order
    combined = sorted(
        zip(labels, best_values, error_bars, strict=False),
        key=operator.itemgetter(1),
        reverse=not decreasing,
    )
    labels, best_values, error_bars = zip(*combined, strict=False)

    _, ax = plt.subplots()

    # Set styling
    ax.grid(axis="y", which="both", zorder=1, linewidth=0.5)
    multiplier = 2 if "Corr" in ylabel or "rAULC" in ylabel else 1
    ax.yaxis.set_major_locator(MultipleLocator(0.1 * multiplier))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05 * multiplier))
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylabel(
        ylabel=ylabel + (r" $\uparrow$" if not decreasing else r" $\downarrow$")
    )
    ax.set(xticklabels=[])
    ax.tick_params(bottom=False)
    ax.set_ylim(bottom=y_min, top=y_max)

    # Plot bars with error bars
    bars = ax.bar(labels, best_values, zorder=2)
    ax.errorbar(
        labels,
        best_values,
        yerr=np.array(error_bars).T,
        fmt="none",
        ecolor=COLOR_ERROR_BAR,
        elinewidth=1,
        capsize=4,
        markeredgewidth=0.5,
        zorder=3,
    )

    label_offset_dict = dict(zip(labels_to_offset, offset_values, strict=False))

    # Add labels to bars and set bar colors
    for bar, label in zip(bars, labels, strict=False):
        y_offset = label_offset_dict.get(label, 0.005 * 2 * (y_max - y_min))
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_min + y_offset,
            label,
            ha="center",
            va="bottom",
            rotation="vertical",
            zorder=4,
            fontsize=6,
        )

        if label in DISTRIBUTIONAL_METHODS:
            bar.set_color(COLOR_DISTRIBUTIONAL)
        elif label == "CE Baseline":
            bar.set_color(COLOR_BASELINE)
        else:
            bar.set_color(COLOR_DETERMINISTIC)

    # Save and close plot
    plt.savefig(save_path)
    plt.close()


def process_run_data(run: Run, prefix: str, suffix: str) -> dict[str, float]:
    """Processes data from a single run.

    Args:
        run: The wandb run object.
        prefix: The metric prefix.
        suffix: The metric suffix.

    Returns:
        A dictionary mapping estimator names to lists of their metrics.
    """

    def is_valid_key(stripped_key: str) -> bool:
        return "mixed" not in stripped_key and (
            stripped_key in ESTIMATOR_CONVERSION_DICT
            or stripped_key in ESTIMATORLESS_METRICS
        )

    metric_dict = defaultdict(list)

    for key in sorted(run.summary.keys()):
        if key.startswith(prefix) and key.endswith(suffix):
            processed_key = process_key(key, prefix, suffix)

            if is_valid_key(processed_key):
                metric_dict[processed_key].append(run.summary[key])

    return metric_dict


def process_key(key: str, prefix: str, suffix: str) -> str:
    """Processes the key according to the prefix and suffix.

    The function removes the prefix and suffix from the key when the key does not
    correspond to an estimatorless metric. Otherwise, it only removes the prefix.

    Args:
        key: The key to process.
        prefix: The prefix of the key to remove.
        suffix: The suffix of the key to remove.

    Returns:
        The processed key.
    """
    removeprefixed_key = key.removeprefix(f"{prefix}_")
    is_estimatorless_metric = removeprefixed_key == suffix

    if is_estimatorless_metric:
        return removeprefixed_key

    stripped_key = removeprefixed_key.removesuffix(f"_{suffix}")

    return stripped_key


def get_best_estimator(metric: str, cross: bool) -> str:
    """Determines the best estimator for the method.

    Args:
        metric: The metric identifier.
        cross: Whether we have cross-evaluation.

    Returns:
        The estimator for the method.
    """
    if not cross:
        if metric == "auroc_oodness":
            return "jensen_shannon_divergences"
        if metric == "rank_correlation_bregman_au":
            return "expected_entropies"

        msg = "Invalid metric"
        raise ValueError(msg)
    if metric == "auroc_oodness":
        return "expected_entropies"
    if metric == "rank_correlation_bregman_au":
        return "jensen_shannon_divergences"

    msg = "Invalid metric"
    raise ValueError(msg)


def plot_figures_for_dataset(
    prefix: str, api: wandb.Api, id_to_method: dict[str, str], args: argparse.Namespace
) -> None:
    """Plots the figures for the dataset corresponding to `prefix`.

    Args:
        prefix: The dataset prefix to match keys to.
        api: The wandb API.
        id_to_method: Mapping from sweep IDs to method names.
        args: Keyword arguments.
    """
    if args.metric == "auroc_oodness" and "mixed" not in prefix:
        return

    save_path = Path(f"results/{args.dataset}") / args.metric / prefix.replace("/", "-")
    save_path.mkdir(parents=True, exist_ok=True)

    best_metric_dict_means = {}
    best_metric_dict_mins_maxs = {}

    for sweep_id, method in tqdm(id_to_method.items()):
        if method not in DISTRIBUTIONAL_METHODS:
            continue

        sweep = api.sweep(f"bmucsanyi/untangle/{sweep_id}")
        metric_dict = defaultdict(list)

        for run in sweep.runs:
            if run.state != "finished":
                continue

            run_metric = process_run_data(run, prefix, args.metric)

            for key, value in run_metric.items():
                non_distributional_check = (
                    method not in DISTRIBUTIONAL_METHODS
                    and key in ONLY_DISTRIBUTIONAL_ESTIMATORS
                )
                evidential_check = (
                    method in EVIDENTIAL_METHODS
                    and key in ONLY_NON_EVIDENTIAL_ESTIMATORS
                )
                duq_check = method == "DUQ" and key == "dempster_shafer_values"
                if non_distributional_check or evidential_check or duq_check:
                    continue

                metric_dict[key].append(value[0])

        if not metric_dict:
            continue

        if len(metric_dict) > 1:
            single_save_path = (
                save_path / f"{method.replace('.', '').replace(' ', '_').lower()}.pdf"
            )
            plot_single_method(
                ylabel=args.ylabel,
                metric_dict=metric_dict,
                save_path=single_save_path,
                y_min=args.y_min,
                y_max=args.y_max,
                decreasing=args.decreasing,
            )

        best_estimator = get_best_estimator(metric=args.metric, cross=args.cross)
        best_metric_dict_means[method] = np.mean(metric_dict[best_estimator])
        best_metric_dict_mins_maxs[method] = (
            np.min(metric_dict[best_estimator]),
            np.max(metric_dict[best_estimator]),
        )

    all_save_path = save_path / f"{args.metric}.pdf"

    plot_all_methods(
        ylabel=args.ylabel,
        best_metric_dict_means=best_metric_dict_means,
        best_metric_dict_mins_maxs=best_metric_dict_mins_maxs,
        save_path=all_save_path,
        y_min=args.y_min,
        y_max=args.y_max,
        decreasing=args.decreasing,
        labels_to_offset=args.labels_to_offset,
        offset_values=args.offset_values,
    )


def main() -> None:
    """Main function to process metrics and generate plots."""
    setup_plot_style()
    args = parser.parse_args()
    api = wandb.Api()

    id_to_method = ID_TO_METHOD[args.dataset]
    dataset_prefix_list = DATASET_PREFIX_LIST[args.dataset]

    for prefix in dataset_prefix_list:
        plot_figures_for_dataset(
            prefix=prefix, api=api, id_to_method=id_to_method, args=args
        )


if __name__ == "__main__":
    main()
