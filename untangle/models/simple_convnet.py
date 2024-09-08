"""Simple sequential ConvNet class."""

from torch import nn


class Lambda(nn.Module):
    """Lambda module wrapper that executes `fn` on input `x`."""

    def __init__(self, fn):
        super().__init__()

        self._fn = fn

    def forward(self, x):
        return self._fn(x)


class SimpleConvNet(nn.Sequential):
    """Simple sequential ConvNet with a constant hidden dimensionality."""

    def __init__(
        self,
        num_classes: int = 10,
        in_chans: int = 3,
        hidden_dim: int = 256,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()

        self._make_conv_block(
            in_channels=in_chans,
            out_channels=hidden_dim,
            kernel_size=(5, 5),
            padding=2,
            index=0,
        )

        for i in range(1, num_blocks):
            self._make_conv_block(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=(3, 3),
                padding=1,
                index=i,
            )
        self.add_module("global_pool", Lambda(lambda x: x.mean(dim=(-2, -1))))
        self.add_module("fc", nn.Linear(hidden_dim, num_classes))

    def _make_conv_block(
        self, in_channels, out_channels, kernel_size, padding, index, *, add_pool=True
    ):
        self.add_module(
            f"conv_{index}",
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        )
        self.add_module(f"act_{index}", nn.ReLU())

        if add_pool:
            self.add_module(f"pool_{index}", nn.MaxPool2d(kernel_size=(2, 2), stride=2))
