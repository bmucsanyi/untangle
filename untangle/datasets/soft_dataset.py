"""Dataset that supports the 'Is one annotation enough?' datasets."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils import data
from torchvision.datasets.folder import pil_loader

DATASET_NAME_TO_PATH = {
    # Closest datasets to ImageNet (containing natural objects)
    "cifar10": "CIFAR10H",
    "treeversity1": "Treeversity#1",
    "turkey": "Turkey",
    "pig": "Pig",
    "benthic": "Benthic",
    # Medical datasets (bit larger shift from pretraining)
    "micebone": "MiceBone",
    "planktion": "Planktion",
    "qualitymri": "QualityMRI",
    # Synthetic dataset, currently unused
    "synthetic": "Synthetic",
    # Same as Treeversity#6, but with 6 tags per image instead of one class
    # (doesn't make sense to use both Treeversity#1 and Treeversity#6)
    "treeversity6": "Treeversity#6",
}


class SoftDataset(data.Dataset):
    """A dataset class for handling soft-labeled image data.

    This class extends PyTorch's Dataset class to work with image datasets that have
    soft labels (multiple annotations per image). It supports loading images and their
    corresponding soft labels, and can be used for training, validation, or testing.

    Args:
        name: Name of the dataset.
        root: Root directory where the dataset is stored.
        split: Which split of the data to use. Defaults to 'train'.
        transform: A function/transform that takes in a PIL image
            and returns a transformed version. Defaults to None.
        target_transform: A function/transform that takes in the
            target and transforms it. Defaults to None.

    Raises:
        RuntimeError: If no images are found in the specified root directory.
    """

    def __init__(
        self,
        name: str,
        root: Path,
        split: str = "test",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        dataset_path = DATASET_NAME_TO_PATH[name]
        root /= dataset_path

        # Load the soft labels
        self.load_raw_annotations(root / "annotations.json")

        self.root = root.parent
        self.samples = self.file_path_to_img_id.keys()

        # Restrict self.samples to val/test
        current_folds = []
        if split == "val":
            current_folds = [f"fold{i}" for i in range(1, 3)]
        elif split == "test":
            current_folds = [f"fold{i}" for i in range(3, 6)]
        elif split == "all":
            current_folds = [f"fold{i}" for i in range(1, 6)]
        self.samples = [s for s in self.samples if any(f in s for f in current_folds)]

        if len(self.samples) == 0:
            msg = f"Found 0 images in subfolders of {root}"
            raise RuntimeError(msg)

        self.transform = None
        self.target_transform = None
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.is_ood = False

    def __getitem__(self, index: int) -> tuple[Image.Image, Tensor]:
        """Retrieves an item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A tuple containing the image and its soft label.
        """
        path_str = self.samples[index]
        full_path_str = str(self.root / path_str)
        target = self.soft_labels[self.file_path_to_img_id[path_str], :]
        img = pil_loader(full_path_str)

        if self.transform is not None and self.is_ood:
            rng = np.random.default_rng(seed=index)
            img = self.transform(img, rng)
        elif self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def set_ood(self) -> None:
        """Sets the dataset to use out-of-distribution transform."""
        self.is_ood = True

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.samples)

    def load_raw_annotations(self, path: Path) -> None:
        """Loads and processes raw annotations from a JSON file.

        Args:
            path: Path to the JSON file containing annotations.
        """
        with path.open() as f:
            raw = json.load(f)

            # Collect all annotations
            img_filepath = []
            labels = []
            for annotator in raw:
                for entry in annotator["annotations"]:
                    # Add only valid annotations to table
                    if (label := entry["class_label"]) is not None:
                        img_filepath.append(entry["image_path"])
                        labels.append(label)

            # Summarize the annotations
            unique_img_file_path = sorted(set(img_filepath))
            file_path_to_img_id = {
                filepath: i for i, filepath in enumerate(unique_img_file_path)
            }

            unique_labels = sorted(set(labels))
            class_name_to_label_id = {label: i for i, label in enumerate(unique_labels)}

            soft_labels = np.zeros(
                (len(unique_img_file_path), len(unique_labels)), dtype=np.int64
            )
            for filepath, classname in zip(img_filepath, labels, strict=True):
                soft_labels[
                    file_path_to_img_id[filepath], class_name_to_label_id[classname]
                ] += 1

            soft_labels = np.concatenate(
                (soft_labels, soft_labels.argmax(axis=-1, keepdims=True)), axis=-1
            )
            self.soft_labels = torch.from_numpy(soft_labels)
            self.file_path_to_img_id = file_path_to_img_id
