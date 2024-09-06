"""Dataset subset utility."""

from torch.utils.data import Dataset


class Subset(Dataset):
    """Subset of a dataset at specified indices. Adapted from torch.utils.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices):  # noqa: PLW3201
        # Add batched sampling support when parent dataset supports it.
        # See torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])

        return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name: str):
        if name in {"dataset", "indices"}:
            return object.__getattribute__(self, name)
        return getattr(self.dataset, name)

    def __setattr__(self, name: str, value):
        if name in {"dataset", "indices"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.dataset, name, value)
