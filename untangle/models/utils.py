"""Model utils."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FlattenAdaptiveAvgPool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size."""

    def __init__(self) -> None:
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the FlattenAdaptiveAvgPool2d.

        Args:
            x: Input tensor.

        Returns:
            Flattened output tensor after global average pooling.
        """
        x = self.pool(x)
        x = self.flatten(x)
        return x


class PoolPad(nn.Module):
    """Shortcut module with average/bed-of-nails pooling and padding.

    Args:
        stride: Stride for pooling.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        downsample_type: Type of downsampling ('bed_of_nails_pad' or 'avg_pad').

    Raises:
        ValueError: If an invalid downsample_type is provided.
    """

    def __init__(
        self, stride: int, in_channels: int, out_channels: int, downsample_type: str
    ) -> None:
        super().__init__()

        if downsample_type not in {"bed_of_nails_pad", "avg_pad"}:
            msg = f"Invalid downsample_type '{downsample_type}' provided"
            raise ValueError(msg)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 1 if downsample_type == "bed_of_nails_pad" else self.stride

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the PoolPad.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after pooling and padding.
        """
        x = F.avg_pool2d(input=x, kernel_size=self.kernel_size, stride=self.stride)
        pad = torch.zeros(
            size=(
                x.shape[0],
                self.out_channels - self.in_channels,
                x.shape[2],
                x.shape[3],
            ),
            dtype=x.dtype,
            device=x.device,
        )
        x = torch.cat([x, pad], dim=1)
        return x


class BinaryClassifier(nn.Module):
    """Simple binary classifier MLP.

    Args:
        in_channels: Number of input channels.
        width: Width of hidden layers.
        depth: Number of hidden layers.
    """

    def __init__(
        self,
        in_channels: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()

        layers = [nn.Linear(in_channels, width), nn.LeakyReLU()]

        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.LeakyReLU()])

        layers.extend([nn.Linear(width, 1)])
        self.unc_module = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the BinaryClassifier.

        Args:
            x: Input tensor.

        Returns:
            Output tensor representing binary classification logits.
        """
        return self.unc_module(x)


class NonNegativeRegressor(nn.Module):
    """Simple non-negative regressor MLP.

    Args:
        in_channels: Number of input channels.
        width: Width of hidden layers.
        depth: Number of hidden layers.
        eps: Small value to ensure non-negative output.
    """

    def __init__(
        self,
        in_channels: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()

        layers = [nn.Linear(in_channels, width), nn.LeakyReLU()]

        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.LeakyReLU()])
        layers.extend([nn.Linear(width, 1), nn.Softplus()])

        self.unc_module = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the NonNegativeRegressor.

        Args:
            x: Input tensor.

        Returns:
            Output tensor representing non-negative regression values.
        """
        return self.unc_module(x)
