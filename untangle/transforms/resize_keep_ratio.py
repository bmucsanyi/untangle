"""Resize transform that keeps the ratio of the input."""

import torchvision.transforms.functional as F
from torch import Tensor

from .resize import STR_TO_INTERPOLATION


class ResizeKeepRatio:
    """Resizes the input while keeping its ratio.

    Args:
        size: Desired output size.
        longest: Weight for the longest side in ratio calculation.
        interpolation: Desired interpolation method.
        fill: Pixel fill value for the area outside the image.
    """

    def __init__(
        self,
        size: tuple[int, int],
        longest: float = 0.0,
        interpolation: str = "bilinear",
        fill: int = 0,
    ) -> None:
        self.size = size
        self.interpolation = STR_TO_INTERPOLATION[interpolation]
        self.longest = longest
        self.fill = fill

    @staticmethod
    def get_params(
        img: Tensor, target_size: tuple[int, int], longest: float
    ) -> list[int]:
        """Calculates the new size while keeping the aspect ratio.

        Args:
            img: Input image tensor.
            target_size: Desired output size.
            longest: Weight for the longest side in ratio calculation.

        Returns:
            New size that keeps the aspect ratio.
        """
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (
            1.0 - longest
        )
        size = [round(x / ratio) for x in source_size]

        return size

    def __call__(self, img: Tensor) -> Tensor:
        """Applies the transform.

        Args:
            img: Image tensor to be resized.

        Returns:
            Resized image tensor.
        """
        size = self.get_params(img, self.size, self.longest)
        img = F.resize(img, size, self.interpolation)

        return img
