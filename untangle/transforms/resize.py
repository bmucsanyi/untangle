"""Resize transform."""

import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

STR_TO_INTERPOLATION = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


class Resize:
    """Resize image transform."""

    def __init__(self, img_size, interpolation):
        self.img_size = img_size

        self.interpolation = STR_TO_INTERPOLATION[interpolation]

    def __call__(self, img):
        return F.resize(img=img, size=self.img_size, interpolation=self.interpolation)
