"""Random resized crop and interpolation transform."""

import math
import random

import torchvision.transforms.functional as F
from torch import Tensor

from .resize import STR_TO_INTERPOLATION


class RandomResizedCropAndInterpolation:
    """Crops the PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size of the original size and a random
    aspect ratio of the original aspect ratio is made. This
    crop is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: Expected output size of each edge.
        scale: Range of size of the origin size cropped.
        ratio: Range of aspect ratio of the origin aspect ratio cropped.
        interpolation: Desired interpolation method.
    """

    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float],
        ratio: tuple[float, float],
        interpolation: str | list[str],
    ) -> None:
        self.size = size
        self.interpolation = STR_TO_INTERPOLATION[interpolation]
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
        img: Tensor, scale: tuple[float, float], ratio: tuple[float, float]
    ) -> tuple[int, int, int, int]:
        """Gets parameters for ``crop`` for a random sized crop.

        Args:
            img: Image to be cropped.
            scale: Range of size of the origin size cropped.
            ratio: Range of aspect ratio of the origin aspect ratio cropped.

        Returns:
            Tuple of params (i, j, h, w) to be passed to ``crop`` for a random sized
            crop.
        """
        area = img.size[0] * img.size[1]

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]

        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]

        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2

        return i, j, h, w

    def __call__(self, img: Tensor) -> Tensor:
        """Applies the transform.

        Args:
            img: Image to be cropped and resized.

        Returns:
            Cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        if isinstance(self.interpolation, tuple | list):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation

        return F.resized_crop(img, i, j, h, w, self.size, interpolation)
