"""Resize transform."""

import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode

STR_TO_INTERPOLATION = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


class Resize:
    """Resize image transform.

    Args:
        img_size: Desired output size.
        interpolation: Desired interpolation method.
    """

    def __init__(self, img_size: int | tuple[int, int], interpolation: str) -> None:
        self.img_size = img_size

        self.interpolation = STR_TO_INTERPOLATION[interpolation]

    def __call__(self, img: Tensor) -> Tensor:
        """Applies the transform.

        Args:
            img: Image tensor to be resized.

        Returns:
            Resized image tensor.
        """
        return F.resize(img=img, size=self.img_size, interpolation=self.interpolation)
