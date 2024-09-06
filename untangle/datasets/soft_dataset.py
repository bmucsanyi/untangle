"""Dataset that supports the 'Is one annotation enough?' datasets."""

import json

import numpy as np
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
        name (str): Name of the dataset.
        root (Path): Root directory where the dataset is stored.
        split (str, optional): Which split of the data to use. Defaults to 'train'.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. Defaults to None.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it. Defaults to None.

    Raises:
        RuntimeError: If no images are found in the specified root directory.

    Example:
        >>> dataset = SoftDataset('my_dataset', Path('/path/to/data'), split='train')
        >>> image, label = dataset[0]
    """

    def __init__(
        self,
        name,
        root,
        split="test",
        transform=None,
        target_transform=None,
    ):
        dataset_path = DATASET_NAME_TO_PATH[name]
        root /= dataset_path

        # Load the soft labels
        (
            self.soft_labels,
            self.class_to_idx,
            self.filepath_to_imgid,
        ) = self.load_raw_annotations(root / "annotations.json")
        self.soft_labels = np.concatenate(
            [self.soft_labels, self.soft_labels.argmax(axis=-1, keepdims=True)], axis=-1
        )

        self.root = root.parent
        self.samples = self.filepath_to_imgid.keys()

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

    def __getitem__(self, index):
        path = self.samples[index]
        full_path_str = str(self.root / path)

        target = self.soft_labels[self.filepath_to_imgid[full_path_str], :]
        img = pil_loader(full_path_str)

        if self.transform is not None and self.is_ood:
            rng = np.random.default_rng(seed=index)
            img = self.transform(img, rng)
        elif self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def set_ood(self):
        self.is_ood = True

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_raw_annotations(path):
        """Casts the raw annotations into a numpy array of label votes per image."""
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
            unique_img_filepath = list(np.unique(np.array(img_filepath)))
            filepath_to_imgid = dict(
                zip(
                    unique_img_filepath,
                    list(np.arange(0, len(unique_img_filepath))),
                    strict=False,
                )
            )
            unique_labels = list(np.unique(np.array(labels)))
            classname_to_labelid = dict(
                zip(unique_labels, list(np.arange(0, len(unique_labels))), strict=False)
            )
            soft_labels = np.zeros(
                (len(unique_img_filepath), len(unique_labels)), dtype=np.int64
            )
            for filepath, classname in zip(img_filepath, labels, strict=False):
                soft_labels[
                    filepath_to_imgid[filepath], classname_to_labelid[classname]
                ] += 1

            return soft_labels, classname_to_labelid, filepath_to_imgid
