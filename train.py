"""Copyright 2020 Ross Wightman and 2024 Bálint Mucsányi."""

import argparse
import datetime
import logging
import time
from argparse import Namespace
from collections.abc import Callable
from functools import partial
from math import ceil
from pathlib import Path
from typing import Any

import torch
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.scheduler.scheduler import Scheduler
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from untangle.utils import (
    AverageMeter,
    CheckpointSaver,
    DefaultContext,
    NativeScaler,
    create_dataset,
    create_loader,
    create_loss_fn,
    create_model,
    log_wandb,
    optimizer_kwargs,
    parse_args,
    resolve_data_config,
    scheduler_kwargs,
    set_random_seed,
    setup_logging,
    wrap_model,
)
from untangle.utils.loader import PrefetchLoader
from untangle.wrappers import (
    DDUWrapper,
    DUQWrapper,
    LaplaceWrapper,
    MahalanobisWrapper,
    PostNetWrapper,
    SNGPWrapper,
    SWAGWrapper,
    TemperatureWrapper,
)
from validate import (
    evaluate,
    evaluate_on_ood_uniform_test_loaders,
    evaluate_on_ood_varied_test_loaders,
)

logger = logging.getLogger(__name__)


def min_accuracy(args: argparse.Namespace) -> float:
    """Specifies the minimum accuracy the model needs for it to be logged as 'best'.

    Args:
        args: Command-line arguments.

    Returns:
        The minimum accuracy as a float in [0.0, 1.0].
    """
    if args.dataset.endswith("imagenet") and args.train_subset == 1.0:
        return 0.7
    if args.dataset.endswith("cifar10") and args.train_subset == 1.0:
        return 0.9

    return 0.0


def setup_devices(args: argparse.Namespace) -> tuple[torch.device, torch.device]:
    """Sets up the training and storage devices.

    Args:
        args: The command-line arguments.

    Returns:
        A tuple containing the device to use for training and the device
        to use for storing evaluation metrics.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    storage_device = torch.device(args.storage_device)

    logger.info(
        f"Training on device {device}, "
        f"storing eval metrics on device {storage_device}."
    )

    return device, storage_device


def setup_compile(model: nn.Module, args: argparse.Namespace) -> None:
    """Sets up model compilation if enabled.

    Args:
        model: The model to be compiled.
        args: The command-line arguments.

    Raises:
        ValueError: If compilation is enabled for deep ensembles.
    """
    if args.compile:
        if args.method_name == "deep_ensemble":
            msg = "torch.compile is not supported for deep ensembles"
            raise ValueError(msg)
        model.model = torch.compile(model.model, backend=args.compile)


def setup_amp(
    device: torch.device, args: argparse.Namespace
) -> tuple[Callable, NativeScaler | None]:
    """Sets up automatic mixed precision (AMP) for training.

    Args:
        device: The device to use for training.
        args: The command-line arguments.

    Returns:
        A tuple containing the autocast function to use and the loss scaler
        to use, if applicable.

    Raises:
        ValueError: If an invalid amp_dtype is provided or if AMP is not
            supported for the Laplace method.
    """
    # Setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = DefaultContext()  # Do nothing
    loss_scaler = None

    # Resolve AMP arguments based on PyTorch
    if args.amp:
        if args.amp_dtype not in {"float16", "bfloat16"}:
            msg = f"Invalid amp_dtype={args.amp_dtype} provided"
            raise ValueError(msg)

        if args.method_name == "laplace":
            msg = "AMP is not supported for the Laplace method"

        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)

        if device.type == "cuda" and amp_dtype == torch.float16:
            # Loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()

    action = "Training" if args.epochs > 0 else "Testing"

    if isinstance(amp_autocast, DefaultContext):
        logger.info(f"AMP not enabled. {action} in float32.")
    else:
        logger.info(f"Using native Torch AMP. {action} in mixed precision.")

    return amp_autocast, loss_scaler


def setup_learning_rate(args: argparse.Namespace) -> None:
    """Sets up the learning rate based on the batch size and optimizer.

    Args:
        args: The command-line arguments.
    """
    if args.lr is None:
        global_batch_size = args.batch_size * args.accumulation_steps
        batch_ratio = global_batch_size / 256
        optimizer_name = args.opt.lower()
        lr_base_scale = (
            "sqrt" if any(o in optimizer_name for o in ("ada", "lamb")) else "linear"
        )

        if lr_base_scale == "sqrt":
            batch_ratio **= 0.5

        args.lr = args.lr_base * batch_ratio

        logger.info(
            f"Learning rate ({args.lr}) calculated from base learning rate "
            f"({args.lr_base}) and effective global batch size "
            f"({global_batch_size}) with {lr_base_scale} scaling."
        )


def setup_wrapper(
    model: nn.Module,
    train_loader: DataLoader | PrefetchLoader,
) -> None:
    """Sets up the model wrapper with additional calculations if needed.

    Args:
        model: The wrapped model.
        train_loader: The data loader for the training set.
        args: The command-line arguments.
    """
    if isinstance(model, PostNetWrapper):
        model.calculate_sample_counts(train_loader)


def verify_eval_metric(args: argparse.Namespace) -> str:
    """Verifies that the evaluation metric is valid.

    Args:
        args: The command-line arguments.

    Returns:
        The verified evaluation metric.

    Raises:
        ValueError: If an invalid evaluation metric is specified.
    """
    if not (
        args.eval_metric.startswith("id_eval_")
        and args.eval_metric.endswith("_auroc_hard_bma_correctness_original")
    ):
        msg = (
            "Invalid eval metric name specified: must be "
            "'id_eval_<estimator>_auroc_hard_bma_correctness_original'"
        )
        raise ValueError(msg)

    return args.eval_metric


def setup_output_dir(data_config: dict[str, Any], args: argparse.Namespace) -> Path:
    """Sets up the output directory for checkpoints and logs.

    Args:
        data_config: The data configuration.
        args: The command-line arguments.

    Returns:
        The path to the output directory.
    """
    experiment_name = "-".join([
        datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S-%f"),
        args.model_name.replace("/", "_"),
        str(data_config["input_size"][-1]),
    ])
    output_dir = Path("checkpoints") / experiment_name
    output_dir.mkdir(parents=True)

    logger.info(f"Output directory is {output_dir}.")

    return output_dir


def setup_scheduler(
    optimizer: Optimizer,
    train_loader: DataLoader | PrefetchLoader,
    args: argparse.Namespace,
) -> tuple[Scheduler | None, int]:
    """Sets up the learning rate scheduler.

    Args:
        optimizer: The optimizer to use.
        train_loader: The data loader for the training set.
        args: The command-line arguments.

    Returns:
        A tuple containing the learning rate scheduler and the number of
        epochs to train for.
    """
    lr_scheduler = None
    num_epochs = 0

    if args.epochs > 0:
        # Setup learning rate schedule and starting epoch
        updates_per_epoch = (
            len(train_loader) + args.accumulation_steps - 1
        ) // args.accumulation_steps
        lr_scheduler, num_epochs = create_scheduler_v2(
            optimizer,
            **scheduler_kwargs(args=args),
            updates_per_epoch=updates_per_epoch,
        )

        logger.info(f"Scheduled epochs: {num_epochs}.")

        if lr_scheduler is not None:
            logger.info(
                "Learning rate stepped per "
                f'{"epoch" if lr_scheduler.t_in_epochs else "update"}.'
            )
        else:
            logger.info("Using a fixed learning rate.")

    return lr_scheduler, num_epochs


@torch.no_grad()
def initialize_lazy_modules(
    model: nn.Module,
    amp_autocast: Callable,
    data_config: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """Initializes lazy modules in the model.

    Args:
        model: The model to initialize.
        amp_autocast: The autocast function to use.
        data_config: The data configuration.
        device: The device to use for initialization.
        args: The command-line arguments.
    """
    dummy_input = torch.randn(
        args.batch_size,
        *tuple(data_config["input_size"]),
    ).to(device)

    with amp_autocast():
        model(dummy_input)


def train(
    num_epochs: int,
    model: nn.Module,
    optimizer: Optimizer,
    train_loss_fn: Callable,
    lr_scheduler: Scheduler | None,
    train_loader: DataLoader | PrefetchLoader,
    saver: CheckpointSaver,
    amp_autocast: Callable,
    loss_scaler: NativeScaler | None,
    id_eval_loader: DataLoader | PrefetchLoader,
    eval_metric: str,
    device: torch.device,
    storage_device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[float, int]:
    """Trains the model for the specified number of epochs.

    Args:
        num_epochs: The number of epochs to train for.
        model: The model to train.
        optimizer: The optimizer to use.
        train_loss_fn: The loss function to use during training.
        lr_scheduler: The learning rate scheduler.
        train_loader: The data loader for the training set.
        saver: The checkpoint saver.
        amp_autocast: The autocast function to use.
        loss_scaler: The loss scaler to use.
        id_eval_loader: The data loader for the in-distribution evaluation set.
        eval_metric: The evaluation metric to use.
        device: The device to use for training.
        storage_device: The device to use for storing evaluation metrics.
        output_dir: The path to the output directory.
        args: The command-line arguments.

    Returns:
        A tuple containing the best evaluation metric achieved and the epoch
        at which it was achieved.
    """
    eval_accuracy = "id_eval_hard_bma_accuracy_original"
    best_eval_metric = -float("inf")
    best_eval_metrics = None
    best_epoch = None

    for epoch in range(num_epochs):
        time_start_epoch = time.perf_counter()
        train_metrics = train_one_epoch(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=train_loss_fn,
            args=args,
            device=device,
            lr_scheduler=lr_scheduler,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
        )

        if not isinstance(model, SWAGWrapper):
            eval_metrics = evaluate(
                model=model,
                loader=id_eval_loader,
                loader_name=args.dataset_id.replace("/", "_"),
                device=device,
                storage_device=storage_device,
                amp_autocast=amp_autocast,
                key_prefix="id_eval",
                output_dir=output_dir,
                is_upstream_dataset=True,
                is_test_dataset=False,
                is_soft_dataset="soft" in args.dataset_id,
                args=args,
            )
            logger.info(f"Eval accuracy: {eval_metrics[eval_accuracy]:.4f}")
            logger.info(f"Eval metric: {eval_metrics[eval_metric]:.4f}")

            if eval_metrics[eval_accuracy] < min_accuracy(args):
                # Random AUROC for poor models to meaningfully aid hyperparam sweep
                eval_metrics[eval_metric] = 0.5

            is_new_best = (
                epoch >= args.best_save_start_epoch
                and eval_metrics[eval_metric] > best_eval_metric
            )

            if is_new_best:
                best_eval_metric = eval_metrics[eval_metric]
                best_eval_metrics = eval_metrics
                best_epoch = epoch

            if args.log_wandb:
                log_wandb(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                    best_eval_metrics=best_eval_metrics,
                    optimizer=optimizer,
                )

            if saver is not None and epoch >= args.best_save_start_epoch:
                # Save proper checkpoint with eval metric
                metric = eval_metrics[eval_metric]
                saver.save_checkpoint(epoch=epoch, metric=metric)
        else:
            # Add placeholder value for SWAGWrapper: this method does not support
            # plateau-based LR scheduling
            eval_metrics = {eval_accuracy: 1.0}

        if lr_scheduler is not None:
            # Step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics[eval_accuracy])

        time_end_epoch = time.perf_counter()
        logger.info(f"Epoch {epoch} took {time_end_epoch - time_start_epoch} seconds.")

    return best_eval_metric, best_epoch


def load_best_checkpoint(saver: CheckpointSaver, model: nn.Module) -> None:
    """Loads the best checkpoint for the model.

    Args:
        saver: The checkpoint saver.
        model: The model to load the checkpoint into.
    """
    best_save_path = (
        saver.checkpoint_dir / f"{saver.checkpoint_prefix}_best.{saver.extension}"
    )
    checkpoint = torch.load(best_save_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=True)


def test(
    num_epochs: int,
    model: nn.Module,
    train_loader: DataLoader | PrefetchLoader,
    hard_id_eval_loader: DataLoader | PrefetchLoader,
    varied_s2_eval_loader: DataLoader | PrefetchLoader,
    id_test_loader: DataLoader | PrefetchLoader,
    ood_uniform_test_loaders: dict[str, dict[str, DataLoader | PrefetchLoader]],
    ood_varied_test_loaders: dict[str, DataLoader | PrefetchLoader],
    saver: CheckpointSaver,
    amp_autocast: Callable,
    device: torch.device,
    storage_device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Performs final tests on the trained model.

    Args:
        num_epochs: The number of epochs the model was trained for.
        model: The trained model.
        optimizer: The optimizer used during training.
        train_loader: The data loader for the training set.
        hard_id_eval_loader: The data loader for the hard in-distribution
            evaluation set.
        varied_s2_eval_loader: The data loader for the varied S2 evaluation set.
        id_test_loader: The data loader for the in-distribution test set.
        ood_uniform_test_loaders: The data loaders for the uniform out-of-distribution
            test sets.
        ood_varied_test_loaders: The data loaders for the varied out-of-distribution
            test sets.
        saver: The checkpoint saver.
        amp_autocast: The autocast function to use.
        device: The device to use for testing.
        storage_device: The device to use for storing evaluation metrics.
        output_dir: The path to the output directory.
        args: The command-line arguments.
    """
    logger.info("Starting final tests.")

    if num_epochs > 0 and not isinstance(model, SWAGWrapper):
        # No post-hoc method, load best checkpoint first
        load_best_checkpoint(saver, model)

    time_start_test = time.perf_counter()

    model.eval()

    update_post_hoc_method(
        model=model,
        train_loader=train_loader,
        hard_id_eval_loader=hard_id_eval_loader,
        varied_s2_eval_loader=varied_s2_eval_loader,
        args=args,
    )

    best_test_metrics = evaluate_on_test_sets(
        model=model,
        id_test_loader=id_test_loader,
        ood_uniform_test_loaders=ood_uniform_test_loaders,
        ood_varied_test_loaders=ood_varied_test_loaders,
        device=device,
        storage_device=storage_device,
        amp_autocast=amp_autocast,
        output_dir=output_dir,
        discard_ood_test_sets=args.discard_ood_test_sets,
        discard_uniform_ood_test_sets=args.discard_uniform_ood_test_sets,
        args=args,
    )

    if args.log_wandb:
        log_wandb(best_test_metrics=best_test_metrics)

    time_end_test = time.perf_counter()
    logger.info(f"Tests took {time_end_test - time_start_test:.4f} seconds.")


def main() -> None:
    """Runs the main training and testing pipeline."""
    time_start_setup = time.perf_counter()
    args = parse_args()
    setup_logging(args)
    device, storage_device = setup_devices(args)

    set_random_seed(args.seed)
    data_config = resolve_data_config(vars(args))
    amp_autocast, loss_scaler = setup_amp(device, args)

    model = create_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=data_config["input_size"][0],
        model_kwargs=args.model_kwargs,
    )

    model = wrap_model(
        model=model,
        model_wrapper_name=args.method_name,
        reset_classifier=args.reset_classifier,
        weight_paths=args.weight_paths,
        num_hidden_features=args.num_hidden_features,
        mlp_depth=args.mlp_depth,
        stopgrad=args.stopgrad,
        num_hooks=args.num_hooks,
        module_type=args.module_type,
        module_name_regex=args.module_name_regex,
        dropout_probability=args.dropout_probability,
        use_filterwise_dropout=args.use_filterwise_dropout,
        num_mc_samples=args.num_mc_samples,
        num_mc_samples_integral=args.num_mc_samples_integral,
        num_mc_samples_cv=args.num_mc_samples_cv,
        rbf_length_scale=args.rbf_length_scale,
        ema_momentum=args.ema_momentum,
        matrix_rank=args.matrix_rank,
        use_het=args.use_het,
        temperature=args.temperature,
        pred_type=args.pred_type,
        hessian_structure=args.hessian_structure,
        use_low_rank_cov=args.use_low_rank_cov,
        max_rank=args.max_rank,
        magnitude=args.magnitude,
        num_heads=args.num_heads,
        likelihood=args.likelihood,
        use_spectral_normalization=args.use_spectral_normalization,
        spectral_normalization_iteration=args.spectral_normalization_iteration,
        spectral_normalization_bound=args.spectral_normalization_bound,
        use_spectral_normalized_batch_norm=args.use_spectral_normalized_batch_norm,
        use_tight_norm_for_pointwise_convs=args.use_tight_norm_for_pointwise_convs,
        num_random_features=args.num_random_features,
        gp_kernel_scale=args.gp_kernel_scale,
        gp_output_bias=args.gp_output_bias,
        gp_random_feature_type=args.gp_random_feature_type,
        use_input_normalized_gp=args.use_input_normalized_gp,
        gp_cov_momentum=args.gp_cov_momentum,
        gp_cov_ridge_penalty=args.gp_cov_ridge_penalty,
        gp_input_dim=args.gp_input_dim,
        latent_dim=args.latent_dim,
        num_density_components=args.num_density_components,
        use_batched_flow=args.use_batched_flow,
        edl_activation=args.edl_activation,
        checkpoint_path=args.initial_checkpoint_path,
    )

    # Move model to device
    model.to(device=device)

    if args.channels_last:
        if args.method_name == "laplace":
            msg = "--channels-last not supported for Laplace"
            raise ValueError(msg)

        # Initialize LazyModules in model before switching memory format
        if isinstance(model, SNGPWrapper | DDUWrapper):
            initialize_lazy_modules(model, amp_autocast, data_config, device, args)

        model.to(memory_format=torch.channels_last)

    setup_learning_rate(args)
    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(args=args),
    )
    setup_compile(model, args)

    (
        train_loader,
        id_eval_loader,
        hard_id_eval_loader,
        id_test_loader,
        ood_uniform_test_loaders,
        ood_varied_test_loaders,
        varied_s2_eval_loader,
    ) = create_loaders(
        data_config=data_config,
        args=args,
        device=device,
    )

    setup_wrapper(model, train_loader)

    train_loss_fn = create_loss_fn(args=args, num_batches=len(train_loader))
    train_loss_fn = train_loss_fn.to(device=device)

    # Setup checkpoint saver and eval metric tracking
    eval_metric = verify_eval_metric(args)

    output_dir = setup_output_dir(data_config, args)

    saver = CheckpointSaver(
        model=model,
        optimizer=optimizer,
        amp_scaler=loss_scaler,
        decreasing=False,
        max_history=args.checkpoint_history,
        checkpoint_dir=output_dir,
    )

    lr_scheduler, num_epochs = setup_scheduler(optimizer, train_loader, args)

    time_end_setup = time.perf_counter()
    logger.info(f"Setup took {time_end_setup - time_start_setup:.4f} seconds.")

    try:
        if num_epochs > 0:
            best_eval_metric, best_epoch = train(
                num_epochs=num_epochs,
                model=model,
                optimizer=optimizer,
                train_loss_fn=train_loss_fn,
                lr_scheduler=lr_scheduler,
                train_loader=train_loader,
                saver=saver,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                id_eval_loader=id_eval_loader,
                eval_metric=eval_metric,
                device=device,
                storage_device=storage_device,
                output_dir=output_dir,
                args=args,
            )

            if not isinstance(model, SWAGWrapper):
                logger.info(
                    f"Best eval metric: {best_eval_metric:.4f} (epoch {best_epoch})."
                )

        if args.evaluate_on_test_sets:
            test(
                num_epochs=num_epochs,
                model=model,
                train_loader=train_loader,
                hard_id_eval_loader=hard_id_eval_loader,
                varied_s2_eval_loader=varied_s2_eval_loader,
                id_test_loader=id_test_loader,
                ood_uniform_test_loaders=ood_uniform_test_loaders,
                ood_varied_test_loaders=ood_varied_test_loaders,
                saver=saver,
                amp_autocast=amp_autocast,
                device=device,
                storage_device=storage_device,
                output_dir=output_dir,
                args=args,
            )
    except KeyboardInterrupt:
        pass


def evaluate_on_test_sets(
    model: nn.Module,
    id_test_loader: DataLoader | PrefetchLoader,
    ood_uniform_test_loaders: dict[str, dict[str, DataLoader | PrefetchLoader]],
    ood_varied_test_loaders: dict[str, DataLoader | PrefetchLoader],
    device: torch.device,
    storage_device: torch.device,
    amp_autocast: Callable,
    output_dir: Path,
    discard_ood_test_sets: bool,
    discard_uniform_ood_test_sets: bool,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Evaluates the model on various test sets.

    Args:
        model: The model to evaluate.
        id_test_loader: The data loader for the in-distribution test set.
        ood_uniform_test_loaders: The data loaders for the uniform out-of-distribution
            test sets.
        ood_varied_test_loaders: The data loaders for the varied out-of-distribution
            test sets.
        device: The device to use for evaluation.
        storage_device: The device to use for storing evaluation metrics.
        amp_autocast: The autocast function to use.
        output_dir: The path to the output directory.
        discard_ood_test_sets: Whether to discard out-of-distribution test sets.
        discard_uniform_ood_test_sets: Whether to discard only uniform
            out-of-distribution test sets.
        args: The command-line arguments.

    Returns:
        A dictionary containing the best test metrics.
    """
    if discard_ood_test_sets and not discard_uniform_ood_test_sets:
        msg = (
            "Cannot discard OOD test sets and not discard the uniform ones at the "
            "same time"
        )
        raise ValueError(msg)

    best_test_metrics = evaluate(
        model=model,
        loader=id_test_loader,
        loader_name=args.dataset_id.replace("/", "_"),
        device=device,
        storage_device=storage_device,
        amp_autocast=amp_autocast,
        key_prefix="id_test",
        output_dir=output_dir,
        is_upstream_dataset=True,
        is_test_dataset=True,
        is_soft_dataset="soft" in args.dataset_id,
        args=args,
    )

    if discard_ood_test_sets:
        return best_test_metrics

    if not discard_uniform_ood_test_sets:
        best_test_metrics |= evaluate_on_ood_uniform_test_loaders(
            model=model,
            loaders=ood_uniform_test_loaders,
            device=device,
            storage_device=storage_device,
            amp_autocast=amp_autocast,
            key_prefix="ood_test",
            output_dir=output_dir,
            is_soft_dataset="soft" in args.dataset_id,
            args=args,
        )

    best_test_metrics |= evaluate_on_ood_varied_test_loaders(
        model=model,
        loaders=ood_varied_test_loaders,
        device=device,
        storage_device=storage_device,
        amp_autocast=amp_autocast,
        key_prefix="ood_test",
        output_dir=output_dir,
        is_soft_dataset="soft" in args.dataset_id,
        args=args,
    )

    return best_test_metrics


def create_datasets(
    args: argparse.Namespace, data_config: dict[str, Any]
) -> tuple[
    Dataset,
    Dataset,
    Dataset,
    Dataset,
    dict[str, dict[str, Dataset]],
    dict[str, Dataset],
    Dataset,
]:
    """Creates datasets for training, evaluation, and testing.

    Args:
        args: The command-line arguments.
        data_config: The data configuration.

    Returns:
        A tuple containing various datasets:
        - The training dataset.
        - The in-distribution (hard or soft) evaluation dataset.
        - The in-distribution evaluation dataset that is enforced to have hard labels.
        - The in-distribution (hard or soft) test dataset.
        - The uniform out-of-distribution (hard or soft) test datasets. Each key of the
            outer dict corresponds to one severity level. Each key of the inner dict
            corresponds to one type of image perturbation.
        - The varied out-of-distribution (hard or soft) test datasets. Each key of the
            dict corresponds to one severity level. Each dataset samples the
            perturbation types uniformly at random.
        - The varied out-of-distribution (hard or soft) evaluation dataset
            with severity level two.
    """
    # Create the train dataset
    train_dataset = create_dataset(
        name=args.dataset,
        root=args.data_dir,
        label_root=args.soft_imagenet_label_dir,
        split=args.train_split,
        download=args.dataset_download,
        seed=args.seed,
        subset=args.train_subset,
        input_size=data_config["input_size"],
        padding=args.padding,
        is_training_dataset=True,
        use_prefetcher=args.prefetcher,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],  # From --mean
        std=data_config["std"],  # From --std
        crop_pct=data_config["crop_pct"],
        ood_transform_type=None,
        severity=0,
        convert_soft_labels_to_hard=False,
    )

    # Create the eval datasets
    if not args.discard_ood_test_sets and not args.ood_transforms_eval:
        msg = "A non-empty list of OOD transforms must be specified"
        raise ValueError(msg)

    if not args.ood_transforms_test:
        args.ood_transforms_test = args.ood_transforms_eval

    id_eval_dataset = create_dataset(
        name=args.dataset_id,
        root=args.data_dir_id,
        label_root=args.soft_imagenet_label_dir,
        split=args.val_split,
        download=args.dataset_download,
        seed=args.seed,
        subset=1.0,
        input_size=data_config["input_size"],
        padding=args.padding,
        is_training_dataset=False,
        use_prefetcher=args.prefetcher,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        crop_pct=data_config["crop_pct"],
        ood_transform_type=None,
        severity=0,
        convert_soft_labels_to_hard=False,
    )

    hard_id_eval_dataset = create_dataset(
        name=args.dataset_id,
        root=args.data_dir_id,
        label_root=args.soft_imagenet_label_dir,
        split=args.val_split,
        download=args.dataset_download,
        seed=args.seed,
        subset=1.0,
        input_size=data_config["input_size"],
        padding=args.padding,
        is_training_dataset=False,
        use_prefetcher=args.prefetcher,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        crop_pct=data_config["crop_pct"],
        ood_transform_type=None,
        severity=0,
        convert_soft_labels_to_hard=True,
    )

    varied_s2_eval_dataset = create_dataset(
        name=args.dataset_id,
        root=args.data_dir_id,
        label_root=args.soft_imagenet_label_dir,
        split=args.val_split,
        download=args.dataset_download,
        seed=args.seed,
        subset=1.0,
        input_size=data_config["input_size"],
        padding=args.padding,
        is_training_dataset=False,
        use_prefetcher=args.prefetcher,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        crop_pct=data_config["crop_pct"],
        ood_transform_type=args.ood_transforms_eval,
        severity=2,
        convert_soft_labels_to_hard=True,
    )

    dataset_locations_ood_test = {}
    for severity in args.severities:
        dataset_name = args.dataset_id.replace("/", "_")
        dataset_locations_ood_test[f"{dataset_name}_s{severity}"] = args.data_dir_id

    id_test_dataset = create_dataset(
        name=args.dataset_id,
        root=args.data_dir_id,
        label_root=args.soft_imagenet_label_dir,
        split=args.test_split,
        download=args.dataset_download,
        seed=args.seed,
        subset=1.0,
        input_size=data_config["input_size"],
        padding=args.padding,
        is_training_dataset=False,
        use_prefetcher=args.prefetcher,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        color_jitter=args.color_jitter,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        crop_pct=data_config["crop_pct"],
        ood_transform_type=None,
        severity=0,
        convert_soft_labels_to_hard=False,
    )

    ood_uniform_test_datasets = {}
    for name, location in dataset_locations_ood_test.items():
        ood_uniform_test_datasets[name] = {}

        for ood_transform_type in args.ood_transforms_test:
            ood_uniform_test_datasets[name][ood_transform_type] = create_dataset(
                name=args.dataset_id,
                root=location,
                label_root=args.soft_imagenet_label_dir,
                split=args.test_split,
                download=args.dataset_download,
                seed=args.seed,
                subset=1.0,
                input_size=data_config["input_size"],
                padding=args.padding,
                is_training_dataset=False,
                use_prefetcher=args.prefetcher,
                scale=args.scale,
                ratio=args.ratio,
                hflip=args.hflip,
                color_jitter=args.color_jitter,
                interpolation=data_config["interpolation"],
                mean=data_config["mean"],
                std=data_config["std"],
                crop_pct=data_config["crop_pct"],
                ood_transform_type=ood_transform_type,
                severity=int(name[-1]),
                convert_soft_labels_to_hard=False,
            )

    ood_varied_test_datasets = {}
    for name, location in dataset_locations_ood_test.items():
        ood_varied_test_datasets[name] = create_dataset(
            name=args.dataset_id,
            root=location,
            label_root=args.soft_imagenet_label_dir,
            split=args.test_split,
            download=args.dataset_download,
            seed=args.seed,
            subset=1.0,
            input_size=data_config["input_size"],
            padding=args.padding,
            is_training_dataset=False,
            use_prefetcher=args.prefetcher,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            color_jitter=args.color_jitter,
            interpolation=data_config["interpolation"],
            mean=data_config["mean"],
            std=data_config["std"],
            crop_pct=data_config["crop_pct"],
            ood_transform_type=args.ood_transforms_test,
            severity=int(name[-1]),
            convert_soft_labels_to_hard=False,
        )

    return (
        train_dataset,
        id_eval_dataset,
        hard_id_eval_dataset,
        id_test_dataset,
        ood_uniform_test_datasets,
        ood_varied_test_datasets,
        varied_s2_eval_dataset,
    )


def create_loaders(
    args: argparse.Namespace, data_config: dict[str, Any], device: torch.device
) -> tuple[
    DataLoader | PrefetchLoader,
    DataLoader | PrefetchLoader,
    DataLoader | PrefetchLoader,
    DataLoader | PrefetchLoader,
    dict[str, dict[str, DataLoader | PrefetchLoader]],
    dict[str, DataLoader | PrefetchLoader],
    DataLoader | PrefetchLoader,
]:
    """Creates data loaders for training, evaluation, and testing.

    Args:
        args: The command-line arguments.
        data_config: The data configuration.
        device: The device to use for loading data.

    Returns:
        A tuple containing various data loaders:
        - The training data loader.
        - The in-distribution (hard or soft) evaluation data loader.
        - The in-distribution evaluation data loader that is enforced to have hard
            labels.
        - The in-distribution (hard or soft) test data loader.
        - The uniform out-of-distribution (hard or soft) test data loaders. Each key of
            the outer dict corresponds to one severity level. Each key of the inner dict
            corresponds to one type of image perturbation.
        - The varied out-of-distribution (hard or soft) test data loaders. Each key of
            the dict corresponds to one severity level. Each data loader samples the
            perturbation types uniformly at random.
        - The varied out-of-distribution (hard or soft) evaluation data loader
            with severity level two.
    """
    (
        train_dataset,
        id_eval_dataset,
        hard_id_eval_dataset,
        id_test_dataset,
        ood_uniform_test_datasets,
        ood_varied_test_datasets,
        varied_s2_eval_dataset,
    ) = create_datasets(args, data_config)

    train_loader = create_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        is_training_dataset=True,
        use_prefetcher=args.prefetcher,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True,
        device=device,
    )

    id_eval_loader = create_loader(
        dataset=id_eval_dataset,
        batch_size=args.validation_batch_size or args.batch_size,
        is_training_dataset=False,
        use_prefetcher=args.prefetcher,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.num_eval_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True,
        device=device,
    )

    hard_id_eval_loader = create_loader(
        dataset=hard_id_eval_dataset,
        batch_size=args.validation_batch_size or args.batch_size,
        is_training_dataset=False,
        use_prefetcher=args.prefetcher,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.num_eval_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True,
        device=device,
    )

    varied_s2_eval_loader = create_loader(
        dataset=varied_s2_eval_dataset,
        batch_size=args.validation_batch_size or args.batch_size,
        is_training_dataset=False,
        use_prefetcher=args.prefetcher,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.num_eval_workers,
        pin_memory=args.pin_memory,
        persistent_workers=True,
        device=device,
    )

    id_test_loader = create_loader(
        dataset=id_test_dataset,
        batch_size=args.validation_batch_size or args.batch_size,
        is_training_dataset=False,
        use_prefetcher=args.prefetcher,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.num_eval_workers,
        pin_memory=args.pin_memory,
        persistent_workers=False,
        device=device,
    )

    ood_uniform_test_loaders = {}
    for name, dataset_subset in ood_uniform_test_datasets.items():
        ood_uniform_test_loaders[name] = {}

        for ood_transform_type, dataset in dataset_subset.items():
            ood_uniform_test_loaders[name][ood_transform_type] = create_loader(
                dataset=dataset,
                batch_size=args.validation_batch_size or args.batch_size,
                is_training_dataset=False,
                use_prefetcher=args.prefetcher,
                mean=data_config["mean"],
                std=data_config["std"],
                num_workers=args.num_eval_workers,
                pin_memory=args.pin_memory,
                persistent_workers=False,
                device=device,
            )

    ood_varied_test_loaders = {}
    for name, dataset in ood_varied_test_datasets.items():
        ood_varied_test_loaders[name] = create_loader(
            dataset=dataset,
            batch_size=args.validation_batch_size or args.batch_size,
            is_training_dataset=False,
            use_prefetcher=args.prefetcher,
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=args.num_eval_workers,
            pin_memory=args.pin_memory,
            persistent_workers=False,
            device=device,
        )

    return (
        train_loader,
        id_eval_loader,
        hard_id_eval_loader,
        id_test_loader,
        ood_uniform_test_loaders,
        ood_varied_test_loaders,
        varied_s2_eval_loader,
    )


def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader | PrefetchLoader,
    optimizer: Optimizer,
    loss_fn: Callable,
    args: argparse.Namespace,
    device: torch.device,
    lr_scheduler: Scheduler | None,
    amp_autocast: Callable,
    loss_scaler: NativeScaler | None,
) -> dict[str, float]:
    """Trains the model for one epoch.

    Args:
        epoch: The current epoch number.
        model: The model to train.
        loader: The data loader for the training set.
        optimizer: The optimizer to use.
        loss_fn: The loss function to use.
        args: The command-line arguments.
        device: The device to use for training.
        lr_scheduler: The learning rate scheduler.
        amp_autocast: The autocast function to use.
        loss_scaler: The loss scaler to use.

    Returns:
        A dictionary containing the average loss for the epoch.
    """
    update_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    accumulation_steps = args.accumulation_steps
    current_accumulation_steps = accumulation_steps
    num_batches = len(loader)
    updates_per_epoch = ceil(num_batches / accumulation_steps)
    num_updates = epoch * updates_per_epoch
    last_batch_idx = num_batches - 1
    last_accumulation_steps = num_batches % accumulation_steps
    first_batch_idx_of_last_accumulation = num_batches - last_accumulation_steps

    data_start_time = update_start_time = time.perf_counter()
    optimizer.zero_grad()

    if isinstance(model, SNGPWrapper) and args.gp_cov_momentum < 0:
        model.reset_covariance_matrix()

    if isinstance(model, SWAGWrapper):
        checkpoint_batches = model.calculate_checkpoint_batches(
            num_batches=num_batches,
            num_checkpoints_per_epoch=args.num_checkpoints_per_epoch,
            accumulation_steps=accumulation_steps,
        )

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accumulation_steps == 0
        update_idx = batch_idx // accumulation_steps

        if batch_idx == first_batch_idx_of_last_accumulation:
            current_accumulation_steps = last_accumulation_steps

        if not args.prefetcher:
            input, target = input.to(device), target.to(device)

        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        data_time_m.update(time.perf_counter() - data_start_time)

        if isinstance(model, DUQWrapper):
            input, target = model.prepare_data(input, target)

        loss = forward(
            model=model,
            input=input,
            target=target,
            loss_fn=loss_fn,
            lambda_gradient_penalty=args.lambda_gradient_penalty,
            amp_autocast=amp_autocast,
            accumulation_steps=current_accumulation_steps,
        )

        backward(
            model=model,
            input=input,
            target=target,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            need_update=need_update,
            loss=loss,
        )

        losses_m.update(loss.item() * current_accumulation_steps, input.shape[0])

        if not need_update:
            data_start_time = time.perf_counter()
            continue

        num_updates += 1
        optimizer.zero_grad()

        time_now = time.perf_counter()
        update_time_m.update(time_now - update_start_time)
        update_start_time = time_now

        if isinstance(model, SWAGWrapper) and batch_idx in checkpoint_batches:
            model.update_stats()

        if (update_idx + 1) % args.log_interval == 0 or update_idx in {
            0,
            updates_per_epoch - 1,
        }:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            pad_len = len(str(updates_per_epoch))

            logger.info(
                f"Train: {epoch} [{update_idx + 1:>{pad_len}d}/{updates_per_epoch} "
                f"({100 * (update_idx + 1) / updates_per_epoch:>5.1f}%)]  "
                f"Loss: {losses_m.avg:#.3g}  "
                f"Update Time: {update_time_m.avg:.3f}s  "
                f"Data Time: {data_time_m.avg:.3f}s  "
                f"Learning Rate: {lr:.3e}  "
            )

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        data_start_time = time.perf_counter()

    return {"loss": losses_m.avg}


@torch.no_grad()
def update_post_hoc_method(
    model: nn.Module,
    train_loader: DataLoader | PrefetchLoader,
    hard_id_eval_loader: DataLoader | PrefetchLoader,
    varied_s2_eval_loader: DataLoader | PrefetchLoader,
    args: argparse.Namespace,
) -> None:
    """Updates post-hoc methods if applicable.

    Args:
        model: The model to update.
        train_loader: The data loader for the training set.
        hard_id_eval_loader: The data loader for the hard in-distribution
            evaluation set.
        varied_s2_eval_loader: The data loader for the varied evaluation set of
            severity level two.
        args: The command-line arguments.

    Raises:
        ValueError: If the required data loaders are not specified for certain methods.
    """
    if isinstance(model, LaplaceWrapper):
        model.perform_laplace_approximation(
            train_loader=train_loader, val_loader=hard_id_eval_loader
        )
    elif isinstance(model, MahalanobisWrapper):
        model.train_logistic_regressor(
            train_loader=train_loader,
            id_loader=hard_id_eval_loader,
            ood_loader=varied_s2_eval_loader,
            max_num_training_samples=args.max_num_covariance_samples,
            max_num_id_ood_samples=args.max_num_id_ood_train_samples,
            channels_last=args.channels_last,
        )
    elif isinstance(model, DDUWrapper):
        model.fit_gmm(
            train_loader=train_loader,
            max_num_training_samples=args.max_num_id_train_samples,
            channels_last=args.channels_last,
        )
        model.set_temperature_loader(
            val_loader=hard_id_eval_loader, channels_last=args.channels_last
        )
    elif isinstance(model, TemperatureWrapper) and args.use_temperature_scaling:
        model.set_temperature_loader(
            val_loader=hard_id_eval_loader, channels_last=args.channels_last
        )
    elif isinstance(model, SWAGWrapper):
        model.get_mc_samples(
            train_loader=train_loader,
            num_mc_samples=args.num_mc_samples,
            channels_last=args.channels_last,
        )


def forward(
    model: nn.Module,
    input: Tensor,
    target: Tensor,
    loss_fn: Callable,
    lambda_gradient_penalty: float,
    amp_autocast: Callable,
    accumulation_steps: int,
) -> Tensor:
    """Performs a forward pass through the model.

    Args:
        model: The model to use.
        input: The input data.
        target: The target labels.
        loss_fn: The loss function to use.
        lambda_gradient_penalty: The gradient penalty coefficient.
        amp_autocast: The autocast function to use.
        accumulation_steps: The number of gradient accumulation steps.

    Returns:
        The calculated loss.
    """
    with amp_autocast():
        output = model(input)
        loss = loss_fn(output, target)

        if isinstance(model, DUQWrapper):
            gradient_penalty = model.calc_gradient_penalty(input, output)
            loss += lambda_gradient_penalty * gradient_penalty

    if accumulation_steps > 1:
        loss /= accumulation_steps
    return loss


def backward(
    model: nn.Module,
    input: Tensor,
    target: Tensor,
    optimizer: Optimizer,
    loss_scaler: NativeScaler | None,
    need_update: bool,
    loss: Tensor,
) -> None:
    """Performs a backward pass and update the model parameters.

    Args:
        model: The model to update.
        input: The input data.
        target: The target labels.
        optimizer: The optimizer to use.
        loss_scaler: The loss scaler to use.
        need_update: Whether to update the model parameters.
        loss: The calculated loss.
    """
    if loss_scaler is not None:
        loss_scaler(
            loss=loss,
            optimizer=optimizer,
            need_update=need_update,
        )
    else:
        loss.backward()

        if need_update:
            optimizer.step()

        if isinstance(model, DUQWrapper):
            input.requires_grad_(False)

            with torch.no_grad():
                model.update_centroids(input, target)


if __name__ == "__main__":
    main()
