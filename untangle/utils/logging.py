"""Logging utilities."""

import logging
from collections import OrderedDict

import wandb


def setup_logging(args, level=logging.INFO):
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        datefmt="%Y-%m-%d %H:%M",
        style="{",
        level=level,
        force=True,
    )

    if args.log_wandb:
        wandb.init(project="probit", config=args)


def log_wandb(
    epoch=None,
    train_metrics=None,
    eval_metrics=None,
    best_eval_metrics=None,
    best_test_metrics=None,
    optimizer=None,
):
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
