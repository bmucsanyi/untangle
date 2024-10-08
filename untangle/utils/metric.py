"""Collection of eval metrics."""

import torch
import torch.nn.functional as F
from torch import Tensor


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Resets all statistics of the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        """Updates the meter with a new value.

        Args:
            val: The new value to be added.
            n: The number of instances this value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_correct_pred(output: Tensor, target: Tensor) -> Tensor:
    """Computes whether each target label is the top-1 prediction of the output.

    Args:
        output: The model output tensor.
        target: The ground truth label tensor.

    Returns:
        A tensor of floats indicating correctness of predictions.
    """
    _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
    pred = pred.flatten()
    target = target.flatten()
    correct = pred == target
    return correct.float()


def accuracy(
    output: Tensor, target: Tensor, topk: tuple[int, ...] = (1,)
) -> list[Tensor]:
    """Computes the accuracy for the specified k top predictions.

    Args:
        output: The model output tensor.
        target: The ground truth label tensor.
        topk: A tuple of k-values for which to compute the accuracy.

    Returns:
        A list of accuracies for each k.
    """
    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100 / batch_size
        for k in topk
    ]


def entropy(probs: Tensor, dim: int = -1) -> Tensor:
    """Computes the entropy of a probability distribution.

    Args:
        probs: The probability distribution tensor.
        dim: The dimension along which to compute entropy.

    Returns:
        The entropy tensor.
    """
    log_probs = probs.log()
    min_real = torch.finfo(log_probs.dtype).min
    log_probs = torch.clamp(log_probs, min=min_real)
    p_log_p = log_probs * probs

    return -p_log_p.sum(dim=dim)


def cross_entropy(probs_p: Tensor, log_probs_q: Tensor, dim: int = -1) -> Tensor:
    """Computes the cross-entropy between two probability distributions.

    Args:
        probs_p: The first probability distribution tensor.
        log_probs_q: The log of the second probability distribution tensor.
        dim: The dimension along which to compute cross-entropy.

    Returns:
        The cross-entropy tensor.
    """
    p_log_q = probs_p * log_probs_q

    return -p_log_q.sum(dim=dim)


def kl_divergence(log_probs_p: Tensor, log_probs_q: Tensor, dim: int = -1) -> Tensor:
    """Computes the KL divergence between two probability distributions.

    Args:
        log_probs_p: The log of the first probability distribution tensor.
        log_probs_q: The log of the second probability distribution tensor.
        dim: The dimension along which to compute KL divergence.

    Returns:
        The KL divergence tensor.
    """
    return (log_probs_p.exp() * (log_probs_p - log_probs_q)).sum(dim=dim)


def binary_log_probability(confidences: Tensor, targets: Tensor) -> Tensor:
    """Computes the binary log probability.

    Args:
        confidences: The predicted confidence scores.
        targets: The binary target labels.

    Returns:
        The binary log probability tensor.
    """
    confidences = confidences.clamp(min=1e-7, max=1 - 1e-7)
    return (
        targets * confidences.log() + (1 - targets) * (1 - confidences).log()
    ).mean()


def binary_brier(confidences: Tensor, targets: Tensor) -> Tensor:
    """Computes the binary Brier score.

    Args:
        confidences: The predicted confidence scores.
        targets: The binary target labels.

    Returns:
        The binary Brier score tensor.
    """
    return (-confidences.square() - targets + 2 * confidences * targets).mean()


def multiclass_log_probability(log_preds: Tensor, targets: Tensor) -> Tensor:
    """Computes the multiclass log probability.

    Args:
        log_preds: The log of predicted probabilities.
        targets: The target labels.

    Returns:
        The multiclass log probability tensor.
    """
    return -F.cross_entropy(log_preds, targets)


def multiclass_brier(
    log_preds: Tensor, targets: Tensor, is_soft_targets: bool
) -> Tensor:
    """Computes the multiclass Brier score.

    Args:
        log_preds: The log of predicted probabilities.
        targets: The target labels.
        is_soft_targets: Whether the targets are soft (probabilistic) or hard.

    Returns:
        The multiclass Brier score tensor.
    """
    preds = log_preds.exp()

    if not is_soft_targets:
        targets = F.one_hot(targets, num_classes=preds.shape[-1])

    return -(
        targets * (1 - 2 * preds + preds.square().sum(dim=-1, keepdim=True))
    ).mean()


def calculate_bin_metrics(
    confidences: Tensor, correctnesses: Tensor, num_bins: int = 10
) -> tuple[Tensor, Tensor, Tensor]:
    """Calculates the binwise accuracies, confidences and proportions of samples.

    Args:
        confidences: Tensor of shape (n,) containing predicted confidences.
        correctnesses: Tensor of shape (n,) containing the true correctness labels.
        num_bins: Number of equally sized bins.

    Returns:
        bin_proportions: Float tensor of shape (num_bins,) containing proportion
            of samples in each bin. Sums up to 1.
        bin_confidences: Float tensor of shape (num_bins,) containing the average
            confidence for each bin.
        bin_accuracies: Float tensor of shape (num_bins,) containing the average
            accuracy for each bin.
    """
    correctnesses = correctnesses.float()

    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=confidences.device)
    indices = torch.bucketize(confidences.contiguous(), bin_boundaries) - 1
    indices = torch.clamp(indices, min=0, max=num_bins - 1)

    bin_counts = torch.zeros(
        num_bins, dtype=confidences.dtype, device=confidences.device
    )
    bin_counts.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))
    bin_proportions = bin_counts / bin_counts.sum()
    pos_counts = bin_counts > 0

    bin_confidences = torch.zeros(
        num_bins, dtype=confidences.dtype, device=confidences.device
    )
    bin_confidences.scatter_add_(dim=0, index=indices, src=confidences)
    bin_confidences[pos_counts] /= bin_counts[pos_counts]

    bin_accuracies = torch.zeros(
        num_bins, dtype=correctnesses.dtype, device=confidences.device
    )
    bin_accuracies.scatter_add_(dim=0, index=indices, src=correctnesses)
    bin_accuracies[pos_counts] /= bin_counts[pos_counts]

    return bin_proportions, bin_confidences, bin_accuracies


def calibration_error(
    confidences: Tensor, correctnesses: Tensor, num_bins: int, norm: str
) -> Tensor:
    """Computes the expected/maximum calibration error.

    Args:
        confidences: Tensor of shape (n,) containing predicted confidences.
        correctnesses: Tensor of shape (n,) containing the true correctness labels.
        num_bins: Number of equally sized bins.
        norm: Whether to return ECE (L1 norm) or MCE (inf norm)

    Returns:
        The ECE/MCE.

    Raises:
        ValueError: If the provided norm is neither 'l1' nor 'inf'.
    """
    bin_proportions, bin_confidences, bin_accuracies = calculate_bin_metrics(
        confidences, correctnesses, num_bins
    )

    abs_diffs = (bin_accuracies - bin_confidences).abs()

    if norm == "l1":
        score = (bin_proportions * abs_diffs).sum()
    elif norm == "inf":
        score = abs_diffs.max()
    else:
        msg = f"Provided norm {norm} not l1 nor inf"
        raise ValueError(msg)

    return score


def area_under_lift_curve(
    uncertainties: Tensor,
    correctnesses: Tensor,
    *,
    reverse_sort: bool = False,
) -> Tensor:
    """Computes the area under the lift curve.

    Args:
        uncertainties: Tensor of uncertainties.
        correctnesses: Tensor of correctness labels.
        reverse_sort: Whether to sort uncertainties in reverse order.

    Returns:
        The area under the lift curve.
    """
    uncertainties = uncertainties.double()
    correctnesses = correctnesses.double()
    batch_size = correctnesses.shape[0]

    sorted_idx = torch.argsort(uncertainties, descending=reverse_sort)
    sorted_correctnesses = correctnesses[sorted_idx]

    accuracy = correctnesses.mean()
    cumulative_correctness = torch.cumsum(sorted_correctnesses, dim=0)
    indices = torch.arange(
        1, batch_size + 1, device=uncertainties.device, dtype=torch.double
    )

    lift = (cumulative_correctness / indices) / accuracy

    step = 1 / batch_size
    result = lift.sum() * step - 1

    return result.float()


def relative_area_under_lift_curve(
    uncertainties: Tensor, correctnesses: Tensor
) -> Tensor:
    """Computes the relative area under the lift curve.

    Args:
        uncertainties: Tensor of uncertainties.
        correctnesses: Tensor of correctness labels.

    Returns:
        The relative area under the lift curve.
    """
    area = area_under_lift_curve(uncertainties, correctnesses)
    area_opt = area_under_lift_curve(correctnesses, correctnesses, reverse_sort=True)

    return area / area_opt


def dempster_shafer_metric(logits: Tensor) -> Tensor:
    """Computes the Dempster-Shafer metric.

    Args:
        logits: Tensor of logits.

    Returns:
        The Dempster-Shafer metric.
    """
    num_classes = logits.shape[-1]
    belief_mass = logits.exp().sum(dim=-1)  # [B]
    dempster_shafer_value = num_classes / (belief_mass + num_classes)

    return dempster_shafer_value


def centered_cov(x: Tensor) -> Tensor:
    """Computes the centered covariance matrix.

    Args:
        x: Input tensor.

    Returns:
        The centered covariance matrix.
    """
    n = x.shape[0]

    return 1 / (n - 1) * x.T @ x


# https://github.com/IdoGalil/benchmarking-uncertainty-estimation-performance/blob/main/utils/uncertainty_metrics.py


def area_under_risk_coverage_curve(
    uncertainties: Tensor, correctnesses: Tensor
) -> Tensor:
    """Computes the area under the risk-coverage curve.

    Args:
        uncertainties: Tensor of uncertainties.
        correctnesses: Tensor of correctness labels.

    Returns:
        The area under the risk-coverage curve.
    """
    uncertainties = uncertainties.double()
    correctnesses = correctnesses.double()

    sorted_indices = torch.argsort(uncertainties)
    correctnesses = correctnesses[sorted_indices]
    total_samples = uncertainties.shape[0]

    cumulative_incorrect = torch.cumsum(1 - correctnesses, dim=0)
    indices = torch.arange(
        1, total_samples + 1, device=uncertainties.device, dtype=torch.double
    )

    aurc = torch.sum(cumulative_incorrect / indices) / total_samples

    return aurc.float()


def excess_area_under_risk_coverage_curve(
    uncertainties: Tensor, correctnesses: Tensor
) -> Tensor:
    """Computes the excess area under the risk-coverage curve.

    Args:
        uncertainties: Tensor of uncertainties.
        correctnesses: Tensor of correctness labels.

    Returns:
        The excess area under the risk-coverage curve.
    """
    aurc = area_under_risk_coverage_curve(uncertainties, correctnesses)

    accuracy = correctnesses.float().mean()
    risk = 1 - accuracy
    # From https://arxiv.org/abs/1805.08206 :
    optimal_aurc = risk + (1 - risk) * torch.log(1 - risk)

    return aurc - optimal_aurc


def coverage_for_accuracy(
    uncertainties: Tensor,
    correctnesses: Tensor,
    accuracy: float = 0.95,
    start_index: int = 200,
) -> Tensor:
    """Computes the coverage for a given accuracy threshold.

    Args:
        uncertainties: Tensor of uncertainties.
        correctnesses: Tensor of correctness labels.
        accuracy: The desired accuracy threshold.
        start_index: The starting index for non-strict measurement.

    Returns:
        The coverage for the given accuracy threshold.
    """
    sorted_indices = torch.argsort(uncertainties)
    correctnesses = correctnesses[sorted_indices]

    cumsum_correctnesses = torch.cumsum(correctnesses, dim=0)
    num_samples = cumsum_correctnesses.shape[0]
    cummean_correctnesses = cumsum_correctnesses / torch.arange(
        1, num_samples + 1, device=uncertainties.device
    )
    coverage_for_accuracy = torch.argmax((cummean_correctnesses < accuracy).float())

    # To ignore statistical noise, start measuring at an index greater than 0
    coverage_for_accuracy_nonstrict = (
        torch.argmax((cummean_correctnesses[start_index:] < accuracy).float())
        + start_index
    )
    if coverage_for_accuracy_nonstrict > start_index:
        # If they were the same, even the first non-noisy measurement didn't satisfy the
        # risk, so its coverage is undue,
        # use the original index. Otherwise, use the non-strict to diffuse noisiness.
        coverage_for_accuracy = coverage_for_accuracy_nonstrict

    coverage_for_accuracy = coverage_for_accuracy.float() / num_samples
    return coverage_for_accuracy


def get_ranks(x: Tensor) -> Tensor:
    """Computes the ranks of elements in a tensor.

    Args:
        x: Input tensor.

    Returns:
        A tensor of ranks.
    """
    return x.argsort().argsort().float()


def spearmanr(x: Tensor, y: Tensor) -> Tensor:
    """Computes the Spearman rank correlation coefficient.

    Args:
        x: First input tensor.
        y: Second input tensor.

    Returns:
        The Spearman rank correlation coefficient.
    """
    if (x == x[0]).all() or (y == y[0]).all():
        return torch.tensor(float("NaN"), device=x.device)

    x_rank = get_ranks(x)
    y_rank = get_ranks(y)

    return torch.corrcoef(torch.stack([x_rank, y_rank]))[0, 1]


def pearsonr(x: Tensor, y: Tensor) -> Tensor:
    """Computes the Pearson correlation coefficient.

    Args:
        x: First input tensor.
        y: Second input tensor.

    Returns:
        The Pearson correlation coefficient.
    """
    return torch.corrcoef(torch.stack([x, y]))[0, 1]


def auroc(y_true: Tensor, y_score: Tensor) -> Tensor:
    """Computes the Area Under the Receiver Operating Characteristic curve (AUROC).

    Args:
        y_true: True binary labels.
        y_score: Target scores.

    Returns:
        The AUROC score.
    """
    # Sort scores and corresponding truth values
    desc_score_indices = torch.argsort(y_score, descending=True)
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Compute the AUC
    distinct_value_indices = torch.where(y_score[1:] - y_score[:-1])[0]
    threshold_idxs = torch.cat([
        distinct_value_indices,
        torch.tensor([y_true.numel() - 1], device=y_score.device),
    ])

    true_positives = torch.cumsum(y_true, dim=0)[threshold_idxs]
    false_positives = 1 + threshold_idxs - true_positives

    true_positives = torch.cat([
        torch.tensor([0], device=true_positives.device),
        true_positives,
    ])
    false_positives = torch.cat([
        torch.tensor([0], device=false_positives.device),
        false_positives,
    ])

    if false_positives[-1] <= 0 or true_positives[-1] <= 0:
        return torch.nan

    false_positive_rate = false_positives / false_positives[-1]
    true_positive_rate = true_positives / true_positives[-1]

    return torch.trapz(true_positive_rate, false_positive_rate)


def diag_hessian_softmax(logit: Tensor) -> Tensor:
    """Computes the diagonal of the Hessian of the softmax function.

    Args:
        logit: Input logits.

    Returns:
        The diagonal of the Hessian of the softmax function.
    """
    prob = logit.softmax(dim=-1)

    return prob * (1 - prob)
