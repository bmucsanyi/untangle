"""Collection of eval metrics."""

import torch
import torch.nn.functional as F
from torch import Tensor


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def is_correct_pred(output, target):
    """Computes whether each target label is the top-1 prediction of the output."""
    _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
    pred = pred.flatten()
    target = target.flatten()
    correct = pred == target
    return correct.float()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy for the specified k top predictions."""
    maxk = min(max(topk), output.shape[1])
    batch_size = target.shape[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100 / batch_size
        for k in topk
    ]


def entropy(probs, dim=-1):
    log_probs = probs.log()
    min_real = torch.finfo(log_probs.dtype).min
    log_probs = torch.clamp(log_probs, min=min_real)
    p_log_p = log_probs * probs

    return -p_log_p.sum(dim=dim)


def cross_entropy(probs_p, log_probs_q, dim=-1):
    p_log_q = probs_p * log_probs_q

    return -p_log_q.sum(dim=dim)


def kl_divergence(log_probs_p, log_probs_q, dim=-1):
    return (log_probs_p.exp() * (log_probs_p - log_probs_q)).sum(dim=dim)


def binary_log_probability(confidences, targets):
    confidences = confidences.clamp(min=1e-7, max=1 - 1e-7)
    return (
        targets * confidences.log() + (1 - targets) * (1 - confidences).log()
    ).mean()


def binary_brier(confidences, targets):
    return (-confidences.square() - targets + 2 * confidences * targets).mean()


def multiclass_log_probability(log_preds, targets):
    return -F.cross_entropy(log_preds, targets)


def multiclass_brier(log_preds, targets, is_soft_targets):
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
    uncertainties: torch.Tensor,
    correctnesses: torch.Tensor,
    *,
    reverse_sort: bool = False,
) -> torch.Tensor:
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
    area = area_under_lift_curve(uncertainties, correctnesses)
    area_opt = area_under_lift_curve(correctnesses, correctnesses, reverse_sort=True)

    return area / area_opt


def dempster_shafer_metric(logits: Tensor) -> Tensor:
    num_classes = logits.shape[-1]
    belief_mass = logits.exp().sum(dim=-1)  # [B]
    dempster_shafer_value = num_classes / (belief_mass + num_classes)

    return dempster_shafer_value


def centered_cov(x):
    n = x.shape[0]

    return 1 / (n - 1) * x.T @ x


# https://github.com/IdoGalil/benchmarking-uncertainty-estimation-performance/blob/main/utils/uncertainty_metrics.py


def area_under_risk_coverage_curve(
    uncertainties: torch.Tensor, correctnesses: torch.Tensor
) -> torch.Tensor:
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


def get_ranks(x: torch.Tensor) -> torch.Tensor:
    return x.argsort().argsort().float()


def spearmanr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if (x == x[0]).all() or (y == y[0]).all():
        return torch.tensor(float("NaN"), device=x.device)

    x_rank = get_ranks(x)
    y_rank = get_ranks(y)

    return torch.corrcoef(torch.stack([x_rank, y_rank]))[0, 1]


def pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.corrcoef(torch.stack([x, y]))[0, 1]


def auroc(y_true, y_score):
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
