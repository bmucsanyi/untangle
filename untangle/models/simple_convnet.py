"""Simple sequential ConvNet class."""

from torch import nn


def simple_convnet_3_256(num_classes=10, in_chans=3):
    """Constructs a SimpleConvnet-3-256 model."""
    model = SimpleConvNet(
        num_classes=num_classes,
        in_chans=in_chans,
        hidden_dim=256,
        num_blocks=3,
    )

    return model


def simple_convnet_3_32(num_classes=10, in_chans=3):
    """Constructs a SimpleConvnet-3-32 model."""
    model = SimpleConvNet(
        num_classes=num_classes,
        in_chans=in_chans,
        hidden_dim=32,
        num_blocks=3,
    )

    return model


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

        self.num_classes = num_classes
        self.num_features = hidden_dim

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
        self.add_module(
            "fc", nn.Linear(in_features=hidden_dim, out_features=num_classes)
        )

        self.num_feature_modules = len(self) - 2  # Exclude global_pool and fc

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

    def get_classifier(self):
        return self.fc

    def forward_features(self, x):
        for i, module in enumerate(self):
            if i == self.num_feature_modules:
                break
            x = module(x)
        return x

    def forward_head(self, x, *, pre_logits: bool = False):
        x = self.global_pool(x)
        if pre_logits:
            return x
        return self.fc(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.forward_head(x)
