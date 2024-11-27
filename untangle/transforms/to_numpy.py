"""PIL image to NumPy array converter."""

import numpy as np
from PIL import Image


class ToNumpy:
    """Transform that converts PIL images into NumPy arrays."""

    def __call__(self, pil_img: Image.Image) -> np.ndarray:
        """Converts a PIL image to a NumPy array.

        Args:
            pil_img: PIL image to be converted.

        Returns:
            NumPy array representation of the image.
        """
        np_img = np.array(pil_img, dtype=np.uint8)

        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)

        np_img = np.rollaxis(np_img, 2)  # HWC to CHW

        return np_img
