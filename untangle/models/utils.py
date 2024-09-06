"""Model utils."""

import torch
import torch.nn.functional as F
from torch import nn


class FlattenAdaptiveAvgPool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size."""

    def __init__(
        self,
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d()
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x


class AvgPoolShortCut(nn.Module):
    """Average pooling shortcut module used by the DDU method."""

    def __init__(self, stride, out_c, in_c):
        super().__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(
            x.shape[0],
            self.out_c - self.in_c,
            x.shape[2],
            x.shape[3],
            device=x.device,
        )
        x = torch.cat((x, pad), dim=1)
        return x


class BinaryClassifier(nn.Module):
    """Simple binary classifier MLP."""

    def __init__(
        self,
        in_channels: int,
        width: int,
        depth: int,
    ):
        super().__init__()
        layers = [nn.Linear(in_channels, width), nn.LeakyReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.LeakyReLU()])
        layers.extend([nn.Linear(width, 1)])
        self.unc_module = nn.Sequential(*layers)

    def forward(self, x):
        return self.unc_module(x)


class NonNegativeRegressor(nn.Module):
    """Simple non-negative regressor MLP."""

    def __init__(
        self,
        in_channels: int,
        width: int,
        depth: int,
        eps: float = 1e-10,
    ):
        super().__init__()
        layers = [nn.Linear(in_channels, width), nn.LeakyReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.LeakyReLU()])
        layers.extend([nn.Linear(width, 1), nn.Softplus()])
        self.unc_module = nn.Sequential(*layers)
        self.eps = eps

    def forward(self, x):
        return self.unc_module(x).clamp(min=self.eps)
