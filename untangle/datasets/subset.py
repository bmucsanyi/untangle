"""Dataset subset utility."""

from collections.abc import Sequence
from typing import Any

from torch.utils.data import Dataset


class Subset(Dataset):
    """Represents a subset of a dataset at specified indices.

    Adapted from torch.utils.data.Subset.

    Args:
        dataset: The whole Dataset.
        indices: Indices in the whole set selected for subset.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int | list[int]) -> Any:
        """Retrieves item(s) from the dataset at the specified index/indices.

        Args:
            idx: Index or list of indices of the item(s) to retrieve.

        Returns:
            The item(s) at the specified index/indices.
        """
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: Sequence[int]) -> list[Any]:  # noqa: PLW3201
        """Retrieves multiple items from the dataset.

        Supports batched sampling when the parent dataset supports it.
        See torch.utils.data._utils.fetch._MapDatasetFetcher.

        Args:
            indices: Sequence of indices of the items to retrieve.

        Returns:
            A list of items at the specified indices.
        """
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])

        return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self) -> int:
        """Returns the length of the subset."""
        return len(self.indices)

    def __getattr__(self, name: str) -> Any:
        """Retrieves an attribute from the dataset.

        Args:
            name: Name of the attribute to retrieve.

        Returns:
            The value of the requested attribute.
        """
        if name in {"dataset", "indices"}:
            return object.__getattribute__(self, name)
        return getattr(self.dataset, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets an attribute in the dataset.

        Args:
            name: Name of the attribute to set.
            value: Value to assign to the attribute.
        """
        if name in {"dataset", "indices"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.dataset, name, value)
