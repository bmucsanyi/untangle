"""Logging utilities."""

import argparse
import logging
from collections import OrderedDict

import wandb
from torch.optim import Optimizer


def setup_logging(args: argparse.Namespace, level: int = logging.INFO) -> None:
    """Sets up basic logging configuration and initializes wandb if specified.

    Args:
        args: Arguments containing logging configuration.
        level: Logging level to be set. Defaults to logging.INFO.
    """
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        datefmt="%Y-%m-%d %H:%M",
        style="{",
        level=level,
        force=True,
    )

    if args.log_wandb:
        wandb.init(project="untangle", config=args)


def log_wandb(
    epoch: int | None = None,
    train_metrics: dict[str, float] | None = None,
    eval_metrics: dict[str, float] | None = None,
    best_eval_metrics: dict[str, float] | None = None,
    best_test_metrics: dict[str, float] | None = None,
    optimizer: Optimizer | None = None,
) -> None:
    """Logs various metrics to wandb.

    Args:
        epoch: Current epoch number.
        train_metrics: Dictionary of training metrics.
        eval_metrics: Dictionary of evaluation metrics.
        best_eval_metrics: Dictionary of best evaluation metrics.
        best_test_metrics: Dictionary of best test metrics.
        optimizer: Optimizer object containing learning rate information.
    """
    rowd = OrderedDict()

    if epoch is not None:
        rowd["epoch"] = epoch
    if train_metrics is not None:
        rowd.update(train_metrics)
    if eval_metrics is not None:
        rowd.update(eval_metrics)
    if best_eval_metrics is not None:
        rowd.update([("best_" + k, v) for k, v in best_eval_metrics.items()])
    if best_test_metrics is not None:
        rowd.update([("best_" + k, v) for k, v in best_test_metrics.items()])
    if optimizer is not None:
        lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        lr = sum(lrs) / len(lrs)
        rowd["lr"] = lr

    if rowd:
        wandb.log(rowd)
