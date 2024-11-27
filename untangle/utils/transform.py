"""Transform utilities."""

import math
from collections.abc import Callable

import numpy as np
import torch
from numpy.random import Generator
from PIL import Image
from torch import Tensor
from torchvision import transforms

from untangle.transforms import (
    STR_TO_INTERPOLATION,
    RandomResizedCropAndInterpolation,
    Resize,
    ResizeKeepRatio,
    ToNumpy,
)
from untangle.transforms.ood_transforms_cifar import OOD_TRANSFORM_DICT_CIFAR
from untangle.transforms.ood_transforms_imagenet import OOD_TRANSFORM_DICT_IMAGENET


class OODTransform:
    """ImageNet-C and CIFAR-10C OOD transform class."""

    def __init__(
        self,
        ood_transform_type: str | tuple[str, ...],
        severity: int,
        dataset_name: str,
    ) -> None:
        if not any(name in dataset_name for name in ["cifar", "imagenet"]):
            msg = "Corruptions are only implemented for CIFAR-10(H) and ImageNet(-ReaL)"
            raise ValueError(msg)

        self.ood_transform_type = ood_transform_type
        self.severity = severity

        transform_dict = (
            OOD_TRANSFORM_DICT_CIFAR
            if "cifar" in dataset_name
            else OOD_TRANSFORM_DICT_IMAGENET
        )

        self.has_transform_sequence = not isinstance(ood_transform_type, str)

        if self.has_transform_sequence:
            self.transform = transform_dict
        else:
            self.transform = transform_dict[ood_transform_type]

    def __call__(self, img: Image.Image, rng: Generator) -> Image.Image:
        """Applies the OOD transform to the given image.

        Args:
            img: The input image to transform.
            rng: Random number generator for reproducibility.

        Returns:
            The transformed image.
        """
        if self.has_transform_sequence:
            idx = rng.integers(low=0, high=len(self.ood_transform_type))
            transform = self.transform[self.ood_transform_type[idx]]
        else:
            transform = self.transform

        return transform(img, self.severity, rng)


class CustomCompose(transforms.Compose):
    """Transform composer that supports passing in rng objects for reproducibility."""

    def __init__(self, transforms: list[Callable]) -> None:
        super().__init__(transforms)

    def __call__(self, img: Image.Image, rng: Generator | None = None) -> Image.Image:
        """Apply a series of transforms to the input image.

        Args:
            img: The input image to transform.
            rng: Optional random number generator for reproducibility.

        Returns:
            The transformed image.
        """
        for t in self.transforms:
            img = t(img, rng) if isinstance(t, OODTransform) else t(img)

        return img


def create_transform(
    input_size: tuple[int, int, int],
    dataset_name: str,
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
    ood_transform_type: str | tuple[str, ...] | None,
    severity: int,
) -> Callable:
    """Creates a transform based on the given parameters.

    Args:
        input_size: Size of the input image.
        dataset_name: Name of the dataset.
        padding: Padding size for random crop.
        is_training_dataset: Whether it's a training dataset.
        use_prefetcher: Whether to use prefetcher.
        scale: Scale range for random resized crop.
        ratio: Aspect ratio range for random resized crop.
        hflip: Probability of horizontal flip.
        color_jitter: Strength of color jitter.
        interpolation: Interpolation method.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        crop_pct: Percentage of image to crop.
        ood_transform_type: Type of OOD transform.
        severity: Severity of OOD transform.

    Returns:
        A callable transform.
    """
    if is_training_dataset and ood_transform_type is not None and severity > 0:
        msg = "OOD transformations cannot be applied during training."
        raise ValueError(msg)

    img_size = input_size[-2:]

    if is_training_dataset:
        if "imagenet" in dataset_name:
            transform = transforms_imagenet_train(
                img_size=img_size,
                scale=scale,
                ratio=ratio,
                hflip=hflip,
                color_jitter=color_jitter,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
            )
        elif "cifar" in dataset_name:
            transform = transforms_cifar_train(
                img_size=img_size,
                interpolation=interpolation,
                padding=padding,
                hflip=hflip,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
            )
        else:
            msg = (
                "Please implement the transforms you want to use "
                f"for dataset {dataset_name}."
            )
            raise ValueError(msg)
    elif "imagenet" in dataset_name:
        transform = transforms_imagenet_eval(
            img_size=img_size,
            crop_pct=crop_pct,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            ood_transform_type=ood_transform_type,
            severity=severity,
        )
    elif "cifar" in dataset_name:
        transform = transforms_cifar_eval(
            img_size=img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            ood_transform_type=ood_transform_type,
            severity=severity,
        )
    else:
        msg = (
            "Please implement the transforms you want to use "
            f"for dataset {dataset_name}"
        )
        raise ValueError(msg)

    return transform


def hard_target_transform(
    target: np.ndarray | Tensor | int,
) -> np.ndarray | Tensor:
    """Transforms the target to a hard label if it's a soft label.

    Args:
        target: The input target, which can be a soft label.

    Returns:
        The hard label.
    """
    if isinstance(target, np.ndarray | Tensor):  # Soft dataset
        return target[-1]  # Last entry contains hard label

    return target


def transforms_cifar_train(
    img_size: tuple[int, int],
    interpolation: str,
    padding: int,
    hflip: float,
    use_prefetcher: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    """Creates a transform for CIFAR training.

    Args:
        img_size: Size of the image.
        interpolation: Interpolation method.
        padding: Padding size for random crop.
        hflip: Probability of horizontal flip.
        use_prefetcher: Whether to use prefetcher.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        A composition of transforms.
    """
    tfl = []

    if img_size != (32, 32):
        tfl += [
            Resize(img_size, interpolation),
        ]

    tfl += [
        transforms.RandomCrop(img_size, padding=padding),
    ]

    if hflip > 0:
        tfl += [transforms.RandomHorizontalFlip(p=hflip)]

    if use_prefetcher:
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]

    return transforms.Compose(tfl)


def transforms_imagenet_train(
    img_size: tuple[int, int],
    scale: tuple[float, float],
    ratio: tuple[float, float],
    hflip: float,
    color_jitter: float,
    interpolation: str,
    use_prefetcher: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> transforms.Compose:
    """Creates a transform for ImageNet training.

    Args:
        img_size: Size of the image.
        scale: Scale range for random resized crop.
        ratio: Aspect ratio range for random resized crop.
        hflip: Probability of horizontal flip.
        color_jitter: Strength of color jitter.
        interpolation: Interpolation method.
        use_prefetcher: Whether to use prefetcher.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.

    Returns:
        A composition of transforms.
    """
    primary_tfl = [
        RandomResizedCropAndInterpolation(
            img_size, scale=scale, ratio=ratio, interpolation=interpolation
        )
    ]
    if hflip > 0.0:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]

    secondary_tfl = []

    if color_jitter is not None:
        # Duplicate for brightness, contrast, and saturation, no hue
        color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # Prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]

    return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def transforms_cifar_eval(
    img_size: tuple[int, int],
    interpolation: str,
    use_prefetcher: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    ood_transform_type: str | tuple[str, ...] | None,
    severity: int,
) -> CustomCompose:
    """Creates a transform for CIFAR evaluation.

    Args:
        img_size: Size of the image.
        interpolation: Interpolation method.
        use_prefetcher: Whether to use prefetcher.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        ood_transform_type: Type of OOD transform.
        severity: Severity of OOD transform.

    Returns:
        A CustomCompose of transforms.
    """
    tfl = []

    if ood_transform_type is not None and severity > 0:
        ood_transform = OODTransform(ood_transform_type, severity, dataset_name="cifar")
        tfl += [ood_transform]

    if img_size != (32, 32):
        tfl += [
            Resize(img_size, interpolation),
        ]

    if use_prefetcher:
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]

    return CustomCompose(tfl)


def transforms_imagenet_eval(
    img_size: int | tuple[int, int],
    crop_pct: float,
    interpolation: str,
    use_prefetcher: bool,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    ood_transform_type: str | tuple[str, ...] | None,
    severity: int,
) -> CustomCompose:
    """Creates a transform for ImageNet evaluation.

    Args:
        img_size: Size of the image.
        crop_pct: Percentage of image to crop.
        interpolation: Interpolation method.
        use_prefetcher: Whether to use prefetcher.
        mean: Mean values for normalization.
        std: Standard deviation values for normalization.
        ood_transform_type: Type of OOD transform.
        severity: Severity of OOD transform.

    Returns:
        A CustomCompose of transforms.
    """
    tfl = []

    if isinstance(img_size, tuple | list):
        if len(img_size) != 2:
            msg = "Invalid image size provided"
            raise ValueError(msg)
        scale_size = tuple(math.floor(x / crop_pct) for x in img_size)
    else:
        scale_size = math.floor(img_size / crop_pct)
        scale_size = (scale_size, scale_size)

    # Default crop model is center
    # Aspect ratio is preserved, crops center within image, no borders are added,
    # image is lost
    if scale_size[0] == scale_size[1]:
        # Simple case, use torchvision built-in Resize w/ shortest edge mode
        # (scalar size arg)
        tfl += [
            transforms.Resize(
                scale_size[0], interpolation=STR_TO_INTERPOLATION[interpolation]
            )
        ]
    else:
        # Resize shortest edge to matching target dim for non-square target
        tfl += [ResizeKeepRatio(scale_size)]
    tfl += [transforms.CenterCrop(img_size)]

    # Add OOD transformations
    if ood_transform_type is not None and severity > 0:
        ood_transform = OODTransform(
            ood_transform_type, severity, dataset_name="imagenet"
        )
        tfl += [ood_transform]

    if use_prefetcher:
        # Prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std),
            ),
        ]

    return CustomCompose(tfl)
