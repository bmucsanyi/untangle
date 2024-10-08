"""Loss function utilities."""

import argparse

from torch import nn

from untangle.losses import (
    BMACrossEntropyLoss,
    CorrectnessPredictionLoss,
    DUQLoss,
    EDLLoss,
    FBarCrossEntropyLoss,
    LossPredictionLoss,
    RegularizedUCELoss,
)


def create_loss_fn(args: argparse.Namespace, num_batches: int) -> nn.Module:
    """Creates and returns a loss function based on the provided arguments.

    Args:
        args: Arguments containing loss function configuration.
        num_batches: Number of batches in the dataset.

    Returns:
        A PyTorch loss function (subclass of nn.Module).

    Raises:
        NotImplementedError: If the specified loss function is not implemented.
    """
    # Setup loss function
    if args.loss == "cross-entropy":
        train_loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "bma-cross-entropy":
        train_loss_fn = BMACrossEntropyLoss()
    elif args.loss == "fbar-cross-entropy":
        train_loss_fn = FBarCrossEntropyLoss()
    elif args.loss == "correctness-prediction":
        train_loss_fn = CorrectnessPredictionLoss(
            args.lambda_uncertainty_loss,
            args.detach_uncertainty_target,
            args.use_top5_correctness,
        )
    elif args.loss == "duq":
        train_loss_fn = DUQLoss()
    elif args.loss == "loss-prediction":
        train_loss_fn = LossPredictionLoss(
            args.lambda_uncertainty_loss, args.detach_uncertainty_target
        )
    elif args.loss == "edl":
        train_loss_fn = EDLLoss(
            num_batches=num_batches,
            num_classes=args.num_classes,
            start_epoch=args.edl_start_epoch,
            scaler=args.edl_scaler,
        )
    elif args.loss == "uce":
        train_loss_fn = RegularizedUCELoss(
            regularization_factor=args.uce_regularization_factor
        )
    else:
        msg = f"--loss {args.loss} is not implemented"
        raise NotImplementedError(msg)

    return train_loss_fn
