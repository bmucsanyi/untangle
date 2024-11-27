"""Fast collate implementation."""

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


def fast_collate(
    batch: Sequence[tuple[np.ndarray | Tensor | tuple, int]],
) -> tuple[Tensor, Tensor]:
    """A fast collation function optimized for uint8 images and int64 targets.

    Args:
        batch: A sequence of tuples containing image data and target labels.

    Returns:
        A tuple containing the collated image tensor and target tensor.

    Raises:
        TypeError: If the input is not a sequence of tuples.
        ValueError: If the input tensors are not of consistent shape or type.
    """
    if not isinstance(batch[0], tuple):
        msg = f"Tuple expected at batch[0], got {type(batch[0])}"
        raise TypeError(msg)

    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into one
        # tensor ordered by position such that all tuple of position n will end up in a
        # torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.int64)
        tensor = torch.zeros(
            (flattened_batch_size, *batch[0][0][0].shape), dtype=torch.uint8
        )
        for i in range(batch_size):
            if len(batch[i][0]) != inner_tuple_size:
                msg = "All input tensor tuples must be the same length"
                raise ValueError(msg)

            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets

    if isinstance(batch[0][0], np.ndarray):
        targets = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.int64))

        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)

        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])

        return tensor, targets

    if isinstance(batch[0][0], Tensor):
        targets = torch.from_numpy(np.array([b[1] for b in batch], dtype=np.int64))

        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.uint8)

        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])

        return tensor, targets

    msg = f"batch[0][0] has an invalid type {type(batch[0][0])}"
    raise ValueError(msg)
