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
    name: str,
    root: str,
    label_root: str,
    split: str,
    download: bool,
    seed: int,
    subset: float,
    input_size: int,
    padding: int,
    is_training_dataset: bool,
    use_prefetcher: bool,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    hflip: float,
    color_jitter: float,
    interpolation: str,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    crop_pct: float,
    ood_transform_type: str | None,
    severity: int,
    convert_soft_labels_to_hard: bool,
) -> CIFAR10 | ImageNet | SoftImageNet | SoftDataset | Subset:
    """Creates and returns a dataset based on the given parameters.

    This function creates a dataset with the specified configuration, applying
    transformations and subset selection as needed.

    Args:
        name: Name of the dataset.
        root: Root directory of the dataset.
        label_root: Root directory for labels (used for soft datasets).
        split: Data split to use ('train' or 'val').
        download: Whether to download the dataset if not present.
        seed: Random seed for subset selection.
        subset: Fraction of the dataset to use (1.0 means use all data).
        input_size: Size of the input images.
        padding: Padding to apply to the images.
        is_training_dataset: Whether this is a training dataset.
        use_prefetcher: Whether to use a prefetcher.
        scale: Scale range for transforms.
        ratio: Aspect ratio range for transforms.
        hflip: Horizontal flip probability.
        color_jitter: Color jitter factor.
        interpolation: Interpolation method for resizing.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        crop_pct: Crop percentage for transforms.
        ood_transform_type: Type of out-of-distribution transform to apply.
        severity: Severity of the OOD transform.
        convert_soft_labels_to_hard: Whether to convert soft labels to hard labels.

    Returns:
        The created dataset.

    Raises:
        ValueError: If an unsupported dataset or configuration is specified.
    """
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
