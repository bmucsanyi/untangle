"""Plotting utilities."""

import logging

import matplotlib.colors as mcolors
import numpy as np


def setup_logging() -> None:
    """Sets up logging config."""
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        datefmt="%Y-%m-%d %H:%M",
        style="{",
        level=logging.INFO,
        force=True,
    )


def lighten_color(color: str | tuple, amount: float = 0.5) -> tuple:
    """Lightens the given color by mixing it with white.

    Args:
        color: Original color (hex or RGB).
        amount: Amount of white to mix in. 0 is the original color, 1 is white.

    Returns:
        Lightened color in RGB format.
    """
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        c = color
    c_white = np.array([1.0, 1.0, 1.0])
    new_color = c + (c_white - c) * amount
    return tuple(new_color)


COLOR_GT = np.array([234.0, 67.0, 53.0]) / 255.0
COLOR_ESTIMATE = np.array([66.0, 103.0, 210.0]) / 255.0
COLOR_ERROR_BAR = np.array([105.0, 109.0, 113.0]) / 255.0
COLOR_DISTRIBUTIONAL = np.array([52.0, 168.0, 83.0]) / 255.0
COLOR_BASELINE = np.array([154.0, 160.0, 166.0]) / 255.0
COLOR_DETERMINISTIC = np.array([251.0, 188.0, 4.0]) / 255.0

ID_TO_METHOD_CIFAR10 = {
    "uo3gu133": "CE Baseline",
    "sgvtuzo5": "Corr. Pred.",
    "8apeaj9a": "DDU",
    "bn1hsbqz": "Deep Ens.",
    "ii1o54ln": "DUQ",
    "vuel80q8": "EDL",
    "3h3gyzxj": "GP",
    "k5elnty1": "HET",
    "wj1sesqf": "HetClassNN",
    "xaz96x6d": "HET-XL",
    "nnle8epz": "Laplace",
    "y5mljm78": "Loss Pred.",
    "h0m0bybl": "Mahalanobis",
    "u1ozluxv": "MC Dropout",
    "8bqhu92u": "PostNet",
    "lcvixgvo": "Shallow Ens.",
    "mhu72izt": "SNGP",
    "zsiqsl6u": "SWAG",
    "lh04ospw": "Temperature",
}

ID_TO_METHOD_IMAGENET = {
    "znhyrrk6": "CE Baseline",
    "11ueh7cq": "Corr. Pred.",
    "k9myyurz": "DDU",
    "54kpysjy": "Deep Ens.",
    "gl6qgpv6": "EDL",
    "4nr8lsd1": "GP",
    "t0uem6ob": "HET",
    "bryrtulr": "HetClassNN",
    "t1myokqo": "HET-XL",
    "42thx27s": "Laplace",
    "7flvihja": "Loss Pred.",
    "8a3palks": "Mahalanobis",
    "1pqijue2": "MC Dropout",
    "zm0o0mo9": "PostNet",
    "pipwlaae": "Shallow Ens.",
    "74rysdqf": "SNGP",
    "o04c996o": "SWAG",
    "jfnn98e3": "Temperature",
}

ID_TO_METHOD = {
    "imagenet": ID_TO_METHOD_IMAGENET,
    "cifar10": ID_TO_METHOD_CIFAR10,
}

DATASET_PREFIX_LIST_IMAGENET = [
    "best_id_test",
    "best_ood_test_varied_soft_imagenet_s1",
    "best_ood_test_varied_soft_imagenet_s2",
    "best_ood_test_varied_soft_imagenet_s3",
    "best_ood_test_varied_soft_imagenet_s4",
    "best_ood_test_varied_soft_imagenet_s5",
    # "best_ood_test_avg_soft_imagenet_s1",
    # "best_ood_test_avg_soft_imagenet_s2",
    # "best_ood_test_avg_soft_imagenet_s3",
    # "best_ood_test_avg_soft_imagenet_s4",
    # "best_ood_test_avg_soft_imagenet_s5",
    "best_ood_test_varied_soft_imagenet_s1_mixed_soft_imagenet",
    "best_ood_test_varied_soft_imagenet_s2_mixed_soft_imagenet",
    "best_ood_test_varied_soft_imagenet_s3_mixed_soft_imagenet",
    "best_ood_test_varied_soft_imagenet_s4_mixed_soft_imagenet",
    "best_ood_test_varied_soft_imagenet_s5_mixed_soft_imagenet",
    # "best_ood_test_avg_soft_imagenet_s1_mixed_soft_imagenet",
    # "best_ood_test_avg_soft_imagenet_s2_mixed_soft_imagenet",
    # "best_ood_test_avg_soft_imagenet_s3_mixed_soft_imagenet",
    # "best_ood_test_avg_soft_imagenet_s4_mixed_soft_imagenet",
    # "best_ood_test_avg_soft_imagenet_s5_mixed_soft_imagenet",
]

DATASET_PREFIX_LIST_CIFAR10 = [
    "best_id_test",
    "best_ood_test_varied_soft_cifar10_s1",
    "best_ood_test_varied_soft_cifar10_s2",
    "best_ood_test_varied_soft_cifar10_s3",
    "best_ood_test_varied_soft_cifar10_s4",
    "best_ood_test_varied_soft_cifar10_s5",
    # "best_ood_test_avg_soft_cifar10_s1",
    # "best_ood_test_avg_soft_cifar10_s2",
    # "best_ood_test_avg_soft_cifar10_s3",
    # "best_ood_test_avg_soft_cifar10_s4",
    # "best_ood_test_avg_soft_cifar10_s5",
    "best_ood_test_varied_soft_cifar10_s1_mixed_soft_cifar10",
    "best_ood_test_varied_soft_cifar10_s2_mixed_soft_cifar10",
    "best_ood_test_varied_soft_cifar10_s3_mixed_soft_cifar10",
    "best_ood_test_varied_soft_cifar10_s4_mixed_soft_cifar10",
    "best_ood_test_varied_soft_cifar10_s5_mixed_soft_cifar10",
    # "best_ood_test_avg_soft_cifar10_s1_mixed_soft_cifar10",
    # "best_ood_test_avg_soft_cifar10_s2_mixed_soft_cifar10",
    # "best_ood_test_avg_soft_cifar10_s3_mixed_soft_cifar10",
    # "best_ood_test_avg_soft_cifar10_s4_mixed_soft_cifar10",
    # "best_ood_test_avg_soft_cifar10_s5_mixed_soft_cifar10",
]

DATASET_PREFIX_LIST = {
    "imagenet": DATASET_PREFIX_LIST_IMAGENET,
    "cifar10": DATASET_PREFIX_LIST_CIFAR10,
}

DISTRIBUTIONAL_METHODS = [
    "Deep Ens.",
    "EDL",
    "GP",
    "HET",
    "HetClassNN",
    "HET-XL",
    "Laplace",
    "MC Dropout",
    "PostNet",
    "Shallow Ens.",
    "SNGP",
    "SWAG",
]

EVIDENTIAL_METHODS = ["EDL", "PostNet"]

ESTIMATOR_CONVERSION_DICT = {
    "entropies_of_bma": r"$\text{PU}^\text{it}$",
    "expected_entropies": r"$\text{AU}^\text{it}$",
    "jensen_shannon_divergences": r"$\text{EU}^\text{it}$",
    "gt_total_predictives_bregman_dual_bma": r"$\text{PU}^\text{b}$",
    "gt_aleatorics_bregman": r"$\text{AU}^\text{b}$",
    "expected_divergences": r"$\text{EU}^\text{b}$",
    "gt_predictives_bregman_dual_bma": r"$\text{AU}^\text{b} + \text{B}^\text{b}$",
    "gt_biases_bregman_dual_bma": r"$\text{B}^\text{b}$",
    "expected_entropies_plus_expected_divergences": (
        r"$\text{AU}^\text{it} + \text{EU}^\text{b}$"
    ),
    "entropies_of_dual_bma": r"$\mathbb{H}\left(\tilde{\bm{\pi}}\right)$",
    "one_minus_expected_max_probs": r"$1 - \mathbb{E}\left[\max \bm{\pi}\right]$",
    "one_minus_max_probs_of_bma": r"$1 - \max \bar{\bm{\pi}}$",
    "one_minus_max_probs_of_dual_bma": r"$1 - \max \tilde{\bm{\pi}}$",
    "dempster_shafer_values": r"$\text{D-S}$",
    "error_probabilities": r"$u^\text{cp}$",
    "duq_values": r"$u^\text{duq}$",
    "mahalanobis_values": r"$u^\text{mah}$",
    "loss_values": r"$u^\text{rp}$",
    "gmm_neg_log_densities": r"$u^\text{ddu}$",
    "expected_variances_of_logits": r"$\mathbb{E}\left[\text{var }\bm{f}\right]$",
    "expected_variances_of_internal_logits": (
        r"$\mathbb{E}\left[\text{var }\bm{f}^\text{int}\right]$"
    ),
    "expected_variances_of_probs": r"$\mathbb{E}\left[\text{var }\bm{\pi}\right]$",
    "expected_variances_of_internal_probs": (
        r"$\mathbb{E}\left[\text{var }\bm{\pi}^\text{int}\right]$"
    ),
}

ONLY_DISTRIBUTIONAL_ESTIMATORS = [
    "expected_entropies",
    "jensen_shannon_divergences",
    "gt_total_predictives_bregman_dual_bma",
    "expected_divergences",
    "expected_entropies_plus_expected_divergences",
    "entropies_of_dual_bma",
    "one_minus_expected_max_probs",
    "one_minus_max_probs_of_dual_bma",
    "expected_variances_of_logits",
    "expected_variances_of_internal_logits",
    "expected_variances_of_probs",
    "expected_variances_of_internal_probs",
]

ONLY_NON_EVIDENTIAL_ESTIMATORS = [
    "expected_variances_of_logits",
    "expected_variances_of_internal_logits",
    "expected_variances_of_probs",
    "expected_variances_of_internal_probs",
]

ESTIMATORLESS_METRICS = [
    "hard_bma_accuracy_original",
    "correlation_it_au_eu",
    "correlation_it_eu_pu",
    "correlation_it_au_pu",
    "correlation_bregman_au_b_dual_bma",
    "correlation_bregman_eu_au_hat",
    "correlation_bregman_au_eu",
    "correlation_kendall_gal_au_eu_prob",
    "correlation_kendall_gal_au_eu_logit",
    "correlation_kendall_gal_au_eu_internal_prob",
    "correlation_kendall_gal_au_eu_internal_logit",
    "rank_correlation_it_au_eu",
    "rank_correlation_it_eu_pu",
    "rank_correlation_it_au_pu",
    "rank_correlation_bregman_au_b_dual_bma",
    "rank_correlation_bregman_eu_au_hat",
    "rank_correlation_bregman_au_eu",
    "rank_correlation_kendall_gal_au_eu_prob",
    "rank_correlation_kendall_gal_au_eu_logit",
    "rank_correlation_kendall_gal_au_eu_internal_prob",
    "rank_correlation_kendall_gal_au_eu_internal_logit",
    "log_prob_score_hard_bma_aleatoric_original",
    "brier_score_hard_bma_aleatoric_original",
    "time_forward_m",
]

CONSTRAINED_METRICS = [
    "ece_hard_bma_correctness_original",
    "ece_soft_bma_correctness_original",
    "brier_score_hard_bma_correctness_original",
    "brier_score_soft_bma_correctness_original",
    "log_prob_score_hard_bma_correctness_original",
    "log_prob_score_soft_bma_correctness_original",
]

CORRELATION_MATRIX_ESTIMATORS = [
    "one_minus_max_probs_of_dual_bma",
    "one_minus_max_probs_of_bma",
    "one_minus_expected_max_probs",
]
