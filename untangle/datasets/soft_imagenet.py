"ImageNet-ReaL dataset."

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .imagenet import ImageNet


class SoftImageNet(ImageNet):
    """A dataset class for handling ImageNet data with soft labels.

    Extends the ImageNet dataset to work with soft labels, which are
    multiple annotations per image. Supports loading images and their
    corresponding soft labels for validation or testing.

    Args:
        root: Root directory where the ImageNet dataset is stored.
        label_root: Directory containing the soft labels.
            If None, it defaults to the same as `root`.
        **kwargs: Additional arguments to be passed to the ImageNet constructor.

    Raises:
        FileNotFoundError: If the required label files are not found in the specified
            directories.
    """

    def __init__(
        self, root: Path, label_root: Path | None = None, **kwargs: Any
    ) -> None:
        super().__init__(root, split="val", **kwargs)

        if label_root is None:
            label_root = root

        self.soft_labels = self.load_raw_annotations(
            path_soft_labels=label_root / "raters.npz",
            path_real_labels=label_root / "real.json",
        )

        self.is_ood = False

    def __getitem__(self, index: int) -> tuple[Image.Image, np.ndarray]:
        """Retrieves an item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            A tuple containing the image and its augmented soft label.
        """
        path, original_target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None and self.is_ood:
            rng = np.random.default_rng(seed=index)
            img = self.transform(img, rng)
        elif self.transform is not None:
            img = self.transform(img)

        converted_index = int(path[-13:-5]) - 1
        soft_target = self.soft_labels[converted_index, :]
        augmented_target = np.concatenate([soft_target, [original_target]])

        if self.target_transform is not None:
            augmented_target = self.target_transform(augmented_target)

        return img, augmented_target

    def set_ood(self) -> None:
        """Sets the dataset to use out-of-distribution transform."""
        self.is_ood = True

    @staticmethod
    def load_raw_annotations(
        path_soft_labels: Path, path_real_labels: Path
    ) -> np.ndarray:
        """Loads the raw annotations from raters.npz from reassessed-imagenet.

        Adapted from uncertainty-baselines/baselines/jft/data_uncertainty_utils.py#L87.

        Args:
            path_soft_labels: Path to the soft labels file (raters.npz).
            path_real_labels: Path to the real labels file (real.json).

        Returns:
            The soft labels array.
        """
        data = np.load(path_soft_labels)

        summed_ratings = np.sum(data["tensor"], axis=0)  # 0 is the annotator axis
        yes_prob = summed_ratings[:, 2]
        # This gives a [questions] np array.
        # It gives how often the question "Is image X of class Y" was
        # answered with "yes".

        # We now need to summarize these questions across the images and labels
        num_labels = 1000
        soft_labels = {}
        for idx, (file_name, label_id) in enumerate(data["info"]):
            if file_name not in soft_labels:
                soft_labels[file_name] = np.zeros(num_labels, dtype=np.int64)
            added_label = np.zeros(num_labels, dtype=np.int64)
            added_label[int(label_id)] = yes_prob[idx]
            soft_labels[file_name] += added_label

        # Questions were only asked about 24889 images, and of those 1067 have no single
        # yes vote at any label
        # We will fill up (some of) the missing ones by taking the ImageNet Real Labels
        new_soft_labels = {}
        with path_real_labels.open() as f:
            real_labels = json.load(f)
        for idx, label in enumerate(real_labels):
            key = "ILSVRC2012_val_"
            key += (8 - len(str(idx + 1))) * "0" + str(idx + 1) + ".JPEG"
            if len(label) > 0:
                one_hot_label = np.zeros(num_labels, dtype=np.int64)
                one_hot_label[label] = 1
                new_soft_labels[key] = one_hot_label
            else:
                new_soft_labels[key] = np.zeros(num_labels)

        # Merge soft and hard labels
        unique_img_filepath = list(new_soft_labels.keys())
        soft_labels_array = np.zeros((len(unique_img_filepath), 1000), dtype=np.int64)
        for idx, img in enumerate(unique_img_filepath):
            if img in soft_labels and soft_labels[img].sum() > 0:
                final_soft_label = soft_labels[img]
            else:
                final_soft_label = new_soft_labels[img]
            soft_labels_array[idx, :] = final_soft_label

        # Note that 750 of the 50000 images in soft_labels_array will still not have a
        # label at all. These are ones where the old imagenet label was false and also
        # the raters could not determine any new one. We hand 0 matrices out for them.
        # They should be ignored in computing the metrics

        return soft_labels_array
