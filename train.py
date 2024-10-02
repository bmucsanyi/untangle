"""Copyright 2020 Ross Wightman and 2024 Bálint Mucsányi."""

import datetime
import logging
import time
from argparse import Namespace
from functools import partial
from pathlib import Path

import torch
import torch.distributed
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch.nn.parallel import DistributedDataParallel

from test import evaluate, evaluate_bulk
from untangle.utils import (
    AverageMeter,
    CheckpointSaver,
    DefaultContext,
    NativeScaler,
    accuracy,
    create_dataset,
    create_loader,
    create_loss_fn,
    create_model,
    distribute_bn,
    get_activation,
    get_predictive,
    init_distributed_device,
    log_wandb,
    optimizer_kwargs,
    parse_args,
    reduce_tensor,
    resolve_data_config,
    scheduler_kwargs,
    set_random_seed,
    setup_logging,
    wrap_model,
)
from untangle.wrappers import (
    CovariancePushforwardLaplaceWrapper,
    LinearizedSWAGWrapper,
    PostNetWrapper,
    SamplePushforwardLaplaceWrapper,
    SNGPWrapper,
    SWAGWrapper,
)

# TODO(bmucsanyi): Remove Namespace from safe globals once the old checkpoints are not
# used
torch.serialization.add_safe_globals([Namespace])
logger = logging.getLogger(__name__)


def setup_devices(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device, storage_device = init_distributed_device(args)

    if args.distributed:
        logger.info(
            f"Training in distributed mode with {args.world_size} processes, one "
            f"device per process. Process {args.rank}, device {args.device}, storage "
            f"device {args.storage_device}."
        )
    else:
        logger.info(
            f"Training on single device {args.device}, storage device "
            f"{args.storage_device}."
        )

    return device, storage_device


def setup_compile(model, args):
    if args.compile:
        if args.method_name == "deep_ensemble":
            msg = "Compile not supported for deep ensembles"
            raise ValueError(msg)
        model.model = torch.compile(model.model, backend=args.compile)


def setup_amp(device, args):
    # Setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = DefaultContext()  # Do nothing
    loss_scaler = None

    # Resolve AMP arguments based on PyTorch
    if args.amp:
        if args.amp_dtype not in {"float16", "bfloat16"}:
            msg = f"Invalid amp_dtype={args.amp_dtype} provided"
            raise ValueError(msg)

        amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)

        if device.type == "cuda" and amp_dtype == torch.float16:
            # Loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()

    if args.rank == 0:
        action = "Training" if args.epochs > 0 else "Testing"

        if isinstance(amp_autocast, DefaultContext):
            logger.info(f"AMP not enabled. {action} in float32.")
        else:
            logger.info(f"Using native Torch AMP. {action} in mixed precision.")

    return amp_autocast, loss_scaler


def setup_learning_rate(args):
    if args.lr is None:
        global_batch_size = args.batch_size * args.world_size * args.accumulation_steps
        batch_ratio = global_batch_size / 256
        optimizer_name = args.opt.lower()
        lr_base_scale = (
            "sqrt" if any(o in optimizer_name for o in ("ada", "lamb")) else "linear"
        )

        if lr_base_scale == "sqrt":
            batch_ratio **= 0.5

        args.lr = args.lr_base * batch_ratio

        if args.rank == 0:
            logger.info(
                f"Learning rate ({args.lr}) calculated from base learning rate "
                f"({args.lr_base}) and effective global batch size "
                f"({global_batch_size}) with {lr_base_scale} scaling."
            )


def setup_wrapper(model, train_loader):
    if isinstance(model, PostNetWrapper):
        model.calculate_sample_counts(train_loader)


def setup_output_dir(data_config, args):
    experiment_name = (
        f"{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d-%H%M%S-%f')}-"
        f"{args.model_name.replace('/', '_')}-{data_config['input_size'][-1]}"
    )
    output_dir = Path("checkpoints") / experiment_name
    output_dir.mkdir(parents=True)

    if args.rank == 0:
        logger.info(f"Output directory is {output_dir}.")

    return output_dir


def setup_scheduler(optimizer, train_loader, args):
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

        if args.rank == 0:
            logger.info(f"Scheduled epochs: {num_epochs}.")
            logger.info(
                f'LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.'
            )

    return lr_scheduler, num_epochs


def train(
    num_epochs,
    model,
    optimizer,
    train_loss_fn,
    lr_scheduler,
    train_loader,
    saver,
    amp_autocast,
    loss_scaler,
    id_eval_loader,
    device,
    args,
):
    best_eval_metric = -float("inf")
    best_eval_metrics = None
    best_epoch = None
    eval_metric = "top_1_accuracy"

    for epoch in range(num_epochs):
        time_start_epoch = time.perf_counter()

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

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

        if args.distributed:
            if args.rank == 0:
                logger.info("Distributing batch norm statistics.")
            distribute_bn(model, args.world_size)

        if not isinstance(model, SWAGWrapper | LinearizedSWAGWrapper):
            eval_metrics = validate(
                model=model,
                loader=id_eval_loader,
                args=args,
                device=device,
                amp_autocast=amp_autocast,
            )
            logger.info(f"{eval_metric}: {eval_metrics[eval_metric]}")

            is_new_best = (
                epoch >= args.best_save_start_epoch
                and eval_metrics[eval_metric] > best_eval_metric
            )

            if is_new_best:
                best_eval_metric = eval_metrics[eval_metric]
                best_eval_metrics = eval_metrics
                best_epoch = epoch

            if args.rank == 0 and args.log_wandb:
                log_wandb(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                    best_eval_metrics=best_eval_metrics,
                    optimizer=optimizer,
                )

            if args.rank == 0 and epoch >= args.best_save_start_epoch:
                # Save proper checkpoint with eval metric
                metric = eval_metrics[eval_metric]
                saver.save_checkpoint(epoch=epoch, metric=metric)
        else:
            # Add placeholder value for [Linearized]SWAGWrapper: this method does not
            # support plateau-based LR scheduling
            eval_metrics = {"top_1_accuracy": 1.0}

        if lr_scheduler is not None:
            # Step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics["top_1_accuracy"])

        time_end_epoch = time.perf_counter()
        logger.info(
            f"Epoch {epoch} took " f"{time_end_epoch - time_start_epoch} seconds."
        )

    return best_eval_metric, best_epoch


def load_best_checkpoint(saver, model):
    best_save_path = (
        saver.checkpoint_dir / f"{saver.checkpoint_prefix}_best.{saver.extension}"
    )
    checkpoint = torch.load(best_save_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, strict=True)


def test(
    num_epochs,
    model,
    optimizer,
    train_loader,
    hard_id_eval_loader,
    id_test_loader,
    ood_test_loaders,
    saver,
    amp_autocast,
    device,
    storage_device,
    args,
):
    logger.info("Starting final tests.")

    if num_epochs > 0 and not isinstance(model, SWAGWrapper | LinearizedSWAGWrapper):
        # No post-hoc method, load best checkpoint first
        load_best_checkpoint(saver, model)

    time_start_test = time.perf_counter()

    model.eval()

    update_post_hoc_method(
        model=model,
        train_loader=train_loader,
        hard_id_eval_loader=hard_id_eval_loader,
        args=args,
    )

    best_test_metrics = evaluate_on_test_sets(
        model=model,
        id_test_loader=id_test_loader,
        ood_test_loaders=ood_test_loaders,
        device=device,
        storage_device=storage_device,
        amp_autocast=amp_autocast,
        discard_ood_test_sets=args.discard_ood_test_sets,
        args=args,
    )

    if args.log_wandb:
        log_wandb(
            best_test_metrics=best_test_metrics,
            optimizer=optimizer,
        )

    time_end_test = time.perf_counter()
    logger.info(f"Tests took " f"{time_end_test - time_start_test} seconds.")


def main():
    time_start_setup = time.perf_counter()
    args = parse_args()

    device, storage_device = setup_devices(args)

    if args.rank == 0:
        setup_logging(args)

    if args.distributed and args.evaluate_on_test_sets:
        msg = "Distributed setting is not supported"
        raise ValueError(msg)

    set_random_seed(args.seed, args.rank)
    data_config = resolve_data_config(vars(args))

    (
        train_loader,
        id_eval_loader,
        hard_id_eval_loader,
        id_test_loader,
        ood_test_loaders,
    ) = create_loaders(
        data_config=data_config,
        args=args,
        device=device,
    )
    train_loss_fn = create_loss_fn(args=args, num_batches=len(train_loader))
    train_loss_fn = train_loss_fn.to(device=device)

    model = create_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=data_config["input_size"][0],
        model_kwargs=args.model_kwargs,
        verbose=args.rank == 0,
        model_checkpoint_path=args.initial_model_checkpoint_path,
    )

    model = wrap_model(
        model=model,
        model_wrapper_name=args.method_name,
        reset_classifier=args.reset_classifier,
        weight_paths=args.weight_paths,
        num_hidden_features=args.num_hidden_features,
        num_mc_samples=args.num_mc_samples,
        matrix_rank=args.matrix_rank,
        mask_regex=args.mask_regex,
        use_sampling=args.use_sampling,
        temperature=args.temperature,
        use_low_rank_cov=args.use_low_rank_cov,
        max_rank=args.max_rank,
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
        loss_function=train_loss_fn,
        predictive_fn=get_predictive(
            args.predictive,
            use_correction=args.use_correction,
            num_mc_samples=args.num_mc_samples,
        ),
        use_eigval_prior=args.use_eigval_prior,
        gp_likelihood=args.gp_likelihood,
        verbose=args.rank == 0,
    )

    # Move model to device
    model.to(device=device)

    # Setup distributed training
    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[device], broadcast_buffers=True
        )

    setup_learning_rate(args)
    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(args=args),
    )
    amp_autocast, loss_scaler = setup_amp(device, args)
    setup_compile(model, args)

    setup_wrapper(model, train_loader)

    saver = None
    if args.rank == 0:
        # Setup checkpoint saver
        output_dir = setup_output_dir(data_config, args)

        saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            max_history=args.checkpoint_history,
            checkpoint_dir=output_dir,
        )

    lr_scheduler, num_epochs = setup_scheduler(optimizer, train_loader, args)

    time_end_setup = time.perf_counter()

    if args.rank == 0:
        logger.info(f"Setup took {time_end_setup - time_start_setup} seconds.")

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
                device=device,
                args=args,
            )

            if not isinstance(model, SWAGWrapper | LinearizedSWAGWrapper):
                logger.info(
                    f"Best eval metric: {best_eval_metric} (epoch {best_epoch})."
                )

        if args.evaluate_on_test_sets:
            test(
                num_epochs=num_epochs,
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                hard_id_eval_loader=hard_id_eval_loader,
                id_test_loader=id_test_loader,
                ood_test_loaders=ood_test_loaders,
                saver=saver,
                amp_autocast=amp_autocast,
                device=device,
                storage_device=storage_device,
                args=args,
            )
    except KeyboardInterrupt:
        pass


def evaluate_on_test_sets(
    model,
    id_test_loader,
    ood_test_loaders,
    device,
    storage_device,
    amp_autocast,
    discard_ood_test_sets,
    args,
):
    best_test_metrics = evaluate(
        model=model,
        loader=id_test_loader,
        device=device,
        storage_device=storage_device,
        amp_autocast=amp_autocast,
        key_prefix="id_test",
        is_upstream_dataset=True,
        is_soft_dataset="soft" in args.dataset_id,
        args=args,
    )

    if discard_ood_test_sets:
        return best_test_metrics

    best_test_metrics |= evaluate_bulk(
        model=model,
        loaders=ood_test_loaders,
        device=device,
        storage_device=storage_device,
        amp_autocast=amp_autocast,
        key_prefix="ood_test",
        is_upstream_dataset=False,
        is_soft_dataset="soft" in args.dataset_id,
        args=args,
    )

    return best_test_metrics


def create_datasets(args, data_config):
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

    dataset_locations_ood_test = {}
    for severity in args.severities:
        dataset_locations_ood_test[f"{args.dataset_id}S{severity}"] = args.data_dir_id

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

    ood_test_datasets = {}
    for name, location in dataset_locations_ood_test.items():
        ood_test_datasets[name] = {}

        for ood_transform_type in args.ood_transforms_test:
            ood_test_datasets[name][ood_transform_type] = create_dataset(
                name=name[:-2],
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

    return (
        train_dataset,
        id_eval_dataset,
        hard_id_eval_dataset,
        id_test_dataset,
        ood_test_datasets,
    )


def create_loaders(args, data_config, device):
    (
        train_dataset,
        id_eval_dataset,
        hard_id_eval_dataset,
        id_test_dataset,
        ood_test_datasets,
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
        distributed=args.distributed,
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
        distributed=args.distributed,
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
        distributed=False,
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
        distributed=False,
    )

    ood_test_loaders = {}
    for name, dataset_subset in ood_test_datasets.items():
        ood_test_loaders[name] = {}

        for ood_transform_type, dataset in dataset_subset.items():
            ood_test_loaders[name][ood_transform_type] = create_loader(
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
                distributed=False,
            )

    return (
        train_loader,
        id_eval_loader,
        hard_id_eval_loader,
        id_test_loader,
        ood_test_loaders,
    )


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    device,
    lr_scheduler,
    amp_autocast,
    loss_scaler,
):
    update_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    accumulation_steps = args.accumulation_steps
    num_batches = len(loader)
    last_accumulation_steps = num_batches % accumulation_steps
    updates_per_epoch = (num_batches + accumulation_steps - 1) // accumulation_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = num_batches - 1
    last_batch_idx_to_accumulate = num_batches - last_accumulation_steps

    update_start_time = time.perf_counter()
    optimizer.zero_grad()

    if isinstance(model, SNGPWrapper) and args.gp_cov_momentum < 0:
        model.reset_covariance_matrix()

    if isinstance(model, SWAGWrapper | LinearizedSWAGWrapper):
        checkpoint_batches = model.calculate_checkpoint_batches(
            num_batches=num_batches,
            num_checkpoints_per_epoch=args.num_checkpoints_per_epoch,
            accumulation_steps=accumulation_steps,
        )

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accumulation_steps == 0
        update_idx = batch_idx // accumulation_steps

        if batch_idx >= last_batch_idx_to_accumulate:
            accumulation_steps = last_accumulation_steps

        if not args.prefetcher:
            input, target = input.to(device), target.to(device)

        loss = forward(
            model=model,
            input=input,
            target=target,
            loss_fn=loss_fn,
            amp_autocast=amp_autocast,
            accumulation_steps=accumulation_steps,
        )

        if loss.isnan():
            msg = "NaN detected in loss"
            raise ValueError(msg)

        backward(
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            need_update=need_update,
            loss=loss,
        )

        if not args.distributed:
            losses_m.update(loss.item() * accumulation_steps, input.shape[0])

        if not need_update:
            continue

        num_updates += 1
        optimizer.zero_grad()

        time_now = time.perf_counter()
        update_time_m.update(time.perf_counter() - update_start_time)
        update_start_time = time_now

        if (
            isinstance(model, SWAGWrapper | LinearizedSWAGWrapper)
            and batch_idx in checkpoint_batches
        ):
            model.update_stats()

        if update_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(
                    reduced_loss.item() * accumulation_steps, input.shape[0]
                )

            if args.rank == 0:
                logger.info(
                    f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} "
                    f"({100 * update_idx / (updates_per_epoch - 1):>3.0f}%)]  "
                    f"Loss: {losses_m.avg:#.3g}  "
                    f"Time: {update_time_m.avg:.3f}s  "
                    f"LR: {lr:.3e}  "
                )

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

    return {"loss": losses_m.avg}


@torch.no_grad()
def validate(
    model,
    loader,
    args,
    device,
    amp_autocast,
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top_1_m = AverageMeter()

    model.eval()

    end = time.time()

    for input, target in loader:
        if not args.prefetcher:
            input = input.to(device)
            target = target.to(device)

        with amp_autocast():
            output = model(input)
            if len(output) == 2:
                predictive_fn = get_predictive(
                    args.predictive, args.use_correction, args.num_mc_samples
                )
                mean, var = output
                prob = predictive_fn(mean, var)
            elif len(output) == 1 and output[0].ndim == 3:
                output = output[0]
                act_fn = get_activation(args.predictive)
                prob = act_fn(output).mean(dim=1)
            elif len(output) == 1 and output[0].ndim == 2:
                output = output[0]
                prob = output / output.sum(dim=-1, keepdim=True)

        if target.ndim == 2:
            target = target[:, -1]

        log_likelihood = (
            prob[torch.arange(target.shape[0]), target]
            .log()
            .clamp(torch.finfo(prob.dtype).min)
            .mean()
        )
        loss = -log_likelihood
        top_1 = accuracy(prob, target)[0]

        if args.distributed:
            reduced_loss = reduce_tensor(loss, args.world_size)
            top_1 = reduce_tensor(top_1, args.world_size)
        else:
            reduced_loss = loss

        if device.type == "cuda":
            torch.cuda.synchronize()

        losses_m.update(reduced_loss.item(), input.shape[0])
        top_1_m.update(top_1.item(), input.shape[0])

        batch_time_m.update(time.time() - end)
        end = time.time()

    metrics = {"loss": losses_m.avg, "top_1_accuracy": top_1_m.avg}

    return metrics


def update_post_hoc_method(model, train_loader, hard_id_eval_loader, args):
    if isinstance(
        model, SamplePushforwardLaplaceWrapper | CovariancePushforwardLaplaceWrapper
    ):
        if hard_id_eval_loader is None:
            msg = "For Laplace approximation, the ID eval loader has to be specified."
            raise ValueError(msg)
        model.perform_laplace_approximation(train_loader, hard_id_eval_loader)
    elif isinstance(model, SWAGWrapper):
        model.get_mc_samples(
            train_loader=train_loader, num_mc_samples=args.num_mc_samples
        )


def forward(
    model,
    input,
    target,
    loss_fn,
    amp_autocast,
    accumulation_steps,
):
    with amp_autocast():
        if isinstance(model, SNGPWrapper):
            output = model(input, target)
        else:
            output = model(input)
        loss = loss_fn(output, target)

    if accumulation_steps > 1:
        loss /= accumulation_steps
    return loss


def backward(optimizer, loss_scaler, need_update, loss):
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


if __name__ == "__main__":
    main()
