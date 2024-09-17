"""Dataset utilities."""

import torch
from torchvision.datasets import CIFAR10

from untangle.datasets import (
    DATASET_NAME_TO_PATH,
    ImageNet,
    SoftDataset,
    SoftImageNet,
    Subset,
)
from untangle.utils.transform import create_transform, hard_target_transform


def create_dataset(
    name,
    root,
    label_root,
    split,
    download,
    seed,
    subset,
    input_size,
    padding,
    is_training_dataset,
    use_prefetcher,
    scale,
    ratio,
    hflip,
    color_jitter,
    interpolation,
    mean,
    std,
    crop_pct,
    ood_transform_type,
    severity,
    convert_soft_labels_to_hard,
):
    transform = create_transform(
        input_size=input_size,
        dataset_name=name,
        padding=padding,
        is_training_dataset=is_training_dataset,
        use_prefetcher=use_prefetcher,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        color_jitter=color_jitter,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        ood_transform_type=ood_transform_type,
        severity=severity,
    )

    target_transform = None
    if convert_soft_labels_to_hard:
        target_transform = hard_target_transform

    name = name.lower()
    if name.startswith("hard/"):
        name = name.split("/", 2)[-1]

        if name == "imagenet":
            dataset = ImageNet(
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
            )
        elif name == "cifar10":
            dataset = CIFAR10(
                root=root,
                train=split == "train",
                transform=transform,
                target_transform=target_transform,
                download=download,
            )
        else:
            msg = "Unsupported dataset"
            raise ValueError(msg)
    elif name.startswith("soft/"):
        name = name.split("/", 2)[-1]

        if name == "imagenet":
            if split != "val":
                msg = "Only the val split is supported for SoftImageNet"
                raise ValueError(msg)

            dataset = SoftImageNet(
                root=root,
                label_root=label_root,
                transform=transform,
                target_transform=target_transform,
            )
        elif name in DATASET_NAME_TO_PATH:
            dataset = SoftDataset(
                name=name,
                root=root,
                split=split,
                transform=transform,
                target_transform=target_transform,
            )
        else:
            msg = "Unsupported dataset"
            raise ValueError(msg)
    else:
        msg = "Unsupported dataset type"
        raise ValueError(msg)

    if ood_transform_type is not None and severity > 0:
        dataset.set_ood()

    if subset < 1.0:
        num_samples = len(dataset)
        indices = torch.randperm(
            num_samples, generator=torch.Generator().manual_seed(seed)
        )
        subset_size = int(subset * num_samples)
        subset_indices = indices[:subset_size]
        dataset = Subset(dataset, subset_indices)

    return dataset
