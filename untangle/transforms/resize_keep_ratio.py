"""Resize transform that keeps the ratio of the input."""

import torchvision.transforms.functional as F

from .resize import STR_TO_INTERPOLATION


class ResizeKeepRatio:
    """Resize and keep ratio."""

    def __init__(
        self,
        size,
        longest=0.0,
        interpolation="bilinear",
        fill=0,
    ):
        if isinstance(size, list | tuple):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = STR_TO_INTERPOLATION[interpolation]
        self.longest = longest
        self.fill = fill

    @staticmethod
    def get_params(img, target_size, longest):
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

    def __call__(self, img):
        size = self.get_params(img, self.size, self.longest)
        img = F.resize(img, size, self.interpolation)
        return img
