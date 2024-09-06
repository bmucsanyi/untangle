"""ImageNet dataset."""

from pathlib import Path
from typing import Any

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg


class ImageNet(ImageFolder):
    """Minimal ImageNet 2012 Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    """

    def __init__(self, root: Path, split: str = "train", **kwargs: Any) -> None:
        root = self.root = root.resolve()
        self.split = verify_str_arg(split, "split", ("train", "val"))

        super().__init__(self.split_folder, **kwargs)
        self.root = root

    @property
    def split_folder(self) -> str:
        return self.root / self.split

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)
