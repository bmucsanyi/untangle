"""Loss function utilities."""

from torch import nn

from untangle.losses import (
    BMACrossEntropyLoss,
    EDLLoss,
    NormCDFNLLLoss,
    RegularizedPredictiveNLLLoss,
    RegularizedUCELoss,
    SigmoidNLLLoss,
    SoftmaxPredictiveNLLLoss,
    UnnormalizedPredictiveNLLLoss,
)


def create_loss_fn(args, num_batches):
    # Setup loss function
    if args.loss == "cross-entropy":
        train_loss_fn = nn.CrossEntropyLoss()
    elif args.loss == "bma-cross-entropy":
        train_loss_fn = BMACrossEntropyLoss(
            predictive=args.predictive,
            use_correction=args.use_correction,
            num_mc_samples=args.num_mc_samples,
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
            regularization_factor=args.regularization_factor
        )
    elif args.loss == "normcdf-nll":
        train_loss_fn = NormCDFNLLLoss()
    elif args.loss == "sigmoid-nll":
        train_loss_fn = SigmoidNLLLoss()
    elif args.loss == "regularized-predictive-nll":
        train_loss_fn = RegularizedPredictiveNLLLoss(
            predictive=args.predictive,
            use_correction=args.use_correction,
            num_mc_samples=args.num_mc_samples,
            regularization_factor=args.regularization_factor,
        )
    elif args.loss == "unnormalized-predictive-nll":
        train_loss_fn = UnnormalizedPredictiveNLLLoss(predictive=args.predictive)
    elif args.loss == "softmax-predictive-nll":
        train_loss_fn = SoftmaxPredictiveNLLLoss(predictive=args.predictive)
    else:
        msg = f"--loss {args.loss} is not implemented"
        raise NotImplementedError(msg)

    return train_loss_fn
