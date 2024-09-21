"""Loader utilities."""

from functools import partial

import torch

from untangle.utils import DefaultContext

from .collate import fast_collate


class PrefetchLoader:
    """Fast prefetch loader."""

    def __init__(
        self,
        loader,
        mean,
        std,
        device,
    ):
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

    def __iter__(self):
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

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def batch_size(self):
        return self.loader.batch_size

    @property
    def drop_last(self):
        return self.loader.drop_last


def create_loader(
    dataset,
    batch_size,
    is_training_dataset,
    use_prefetcher,
    mean,
    std,
    num_workers,
    pin_memory,
    persistent_workers,
    device,
    distributed,
):
    if use_prefetcher:
        collate_fn = fast_collate
    else:
        collate_fn = torch.utils.data.dataloader.default_collate

    # NOTE: IterableDataset not supported
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=is_training_dataset, drop_last=is_training_dataset
        )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=None if distributed else is_training_dataset,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training_dataset,  # TODO(bmucsanyi): Check for distributed case
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
