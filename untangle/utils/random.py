"""Random seed utilities."""

import random

import numpy as np
import torch


def set_random_seed(seed: int = 42, rank: int = 0) -> None:
    """Sets seed to ``random_seed`` in random, numpy and torch."""
    seed += rank
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
