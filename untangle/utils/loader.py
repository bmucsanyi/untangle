"""Loader utilities."""

from collections.abc import Iterator
from functools import partial

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from untangle.utils import DefaultContext

from .collate import fast_collate


class PrefetchLoader:
    """Data loader that prefetches and preprocesses data on GPU for faster training."""

    def __init__(
        self,
        loader: DataLoader,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        device: torch.device,
    ) -> None:
        normalization_shape = (1, 3, 1, 1)

        self.loader = loader
        self.device = device
        self.mean = torch.tensor(
            [x * 255 for x in mean], device=device, dtype=torch.float32
        ).view(normalization_shape)
        self.std = torch.tensor(
            [x * 255 for x in std], device=device, dtype=torch.float32
        ).view(normalization_shape)
        self.has_cuda = torch.cuda.is_available() and device.type == "cuda"

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Returns an iterator over the data, prefetching and preprocessing batches."""
        first = True
        if self.has_cuda:
            stream = torch.cuda.Stream()
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = DefaultContext()

        for next_input, next_target in self.loader:
            with stream_context():
                next_input = next_input.to(device=self.device, non_blocking=True)
                next_target = next_target.to(device=self.device, non_blocking=True)
                next_input = next_input.to(torch.float32).sub_(self.mean).div_(self.std)

            if not first:
                yield input, target  # noqa: F821, F823
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)

            input = next_input
            target = next_target

        yield input, target

    def __len__(self) -> int:
        """Returns the number of batches in the loader."""
        return len(self.loader)

    @property
    def sampler(self) -> Sampler:
        """Returns the sampler used by the underlying loader."""
        return self.loader.sampler

    @property
    def dataset(self) -> Dataset:
        """Returns the dataset used by the underlying loader."""
        return self.loader.dataset

    @property
    def batch_size(self) -> int:
        """Returns the batch size used by the underlying loader."""
        return self.loader.batch_size

    @property
    def drop_last(self) -> bool:
        """Returns whether the underlying loader drops the last incomplete batch."""
        return self.loader.drop_last


def create_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    is_training_dataset: bool,
    use_prefetcher: bool,
    mean: list[float],
    std: list[float],
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    device: torch.device,
) -> DataLoader | PrefetchLoader:
    """Creates a DataLoader or PrefetchLoader based on the given parameters."""
    if use_prefetcher:
        collate_fn = fast_collate
    else:
        collate_fn = torch.utils.data.dataloader.default_collate

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_training_dataset,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training_dataset,
        persistent_workers=persistent_workers,
    )

    if use_prefetcher:
        loader = PrefetchLoader(
            loader=loader,
            mean=mean,
            std=std,
            device=device,
        )

    return loader
