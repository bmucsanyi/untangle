"""Compatibility utilities for timm."""


def scheduler_kwargs(args):
    """Argparse to kwargs helper.

    Converts scheduler args in argparse args to keyword args.
    No scheduler keyword argument is exposed directly in `args` because we deem them
    unimportant for hyperparameter sweeps..
    """
    sched_kwargs = args.sched_kwargs
    plateau_mode = "max"

    kwargs = {
        "sched": sched_kwargs.get("sched", "cosine"),
        "num_epochs": args.epochs,
        "decay_epochs": sched_kwargs.get("decay_epochs", args.epochs),
        "decay_milestones": sched_kwargs.get("decay_milestones", [30, 60]),
        "warmup_epochs": sched_kwargs.get("warmup_epochs", 5),
        "cooldown_epochs": sched_kwargs.get("cooldown_epochs", 0),
        "patience_epochs": sched_kwargs.get("patience_epochs", 10),
        "decay_rate": sched_kwargs.get("decay_rate", 0.1),
        "min_lr": sched_kwargs.get("min_lr", 0.0),
        "warmup_lr": sched_kwargs.get("warmup_lr", 1e-5),
        "warmup_prefix": sched_kwargs.get("warmup_prefix", False),
        "noise": sched_kwargs.get("lr_noise", None),
        "noise_pct": sched_kwargs.get("lr_noise_pct", 0.67),
        "noise_std": sched_kwargs.get("lr_noise_std", 1.0),
        "noise_seed": args.seed,
        "cycle_mul": sched_kwargs.get("lr_cycle_mul", 1.0),
        "cycle_decay": sched_kwargs.get("lr_cycle_decay", 0.5),
        "cycle_limit": sched_kwargs.get("lr_cycle_limit", 1),
        "k_decay": sched_kwargs.get("lr_k_decay", 1.0),
        "plateau_mode": plateau_mode,
        "step_on_epochs": not sched_kwargs.get("sched_on_updates", False),
    }

    return kwargs


def optimizer_kwargs(args):
    """Argparse to kwargs helper.

    Converts optimizer args in argparse to keyword args.
    The `args` parameter directly contains values that one might want to sweep over.
    Less important parameters are included in `args.optimizer_kwargs`.
    """
    optimizer_kwargs = args.opt_kwargs

    kwargs = {
        "opt": args.opt,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        **optimizer_kwargs,
    }

    return kwargs
