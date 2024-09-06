"""Loss function utilities."""

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


def create_loss_fn(args, num_batches):
    # Setup loss function
    if args.loss == "cross-entropy":
        train_loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "bma-cross-entropy":
        train_loss_fn = BMACrossEntropyLoss()
    elif args.loss == "fbar-cross-entropy":
        train_loss_fn = FBarCrossEntropyLoss()
    elif args.loss == "correctness-prediction":
        train_loss_fn = CorrectnessPredictionLoss(
            args.lambda_uncertainty_loss, args.is_top5
        )
    elif args.loss == "duq":
        train_loss_fn = DUQLoss()
    elif args.loss == "loss-prediction":
        train_loss_fn = LossPredictionLoss(args.lambda_uncertainty_loss, args.is_detach)
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
