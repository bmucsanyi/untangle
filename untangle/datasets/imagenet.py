"""ImageNet dataset."""

from pathlib import Path
from typing import Any

from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import verify_str_arg


class ImageNet(ImageFolder):
    """Minimal ImageNet 2012 Classification Dataset.

    Args:
        root: Root directory of the ImageNet Dataset.
        split: The dataset split, supports ``train``, or ``val``.
            Defaults to "train".
        **kwargs: Additional keyword arguments passed to the ImageFolder constructor.
    """

    def __init__(self, root: Path, split: str = "train", **kwargs: Any) -> None:
        root = self.root = root.resolve()
        self.split = verify_str_arg(split, "split", ("train", "val"))

        super().__init__(self.split_folder, **kwargs)
        self.root = root

    @property
    def split_folder(self) -> Path:
        """Gets the path to the split folder.

        Returns:
            The path to the split folder.
        """
        return self.root / self.split

    def extra_repr(self) -> str:
        """Gets an extra representation string for the dataset.

        Returns:
            A string representation of the dataset split.
        """
        return "Split: {split}".format(**self.__dict__)
