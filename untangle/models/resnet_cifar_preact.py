"""CIFAR (Wide-)ResNet implementation using the pre-activation variant."""

from torch import Tensor, nn

from .utils import FlattenAdaptiveAvgPool2d, PoolPad


class BasicBlockCPreAct(nn.Module):
    """Pre-activation version of the BasicBlock for CIFAR ResNets.

    Args:
        in_planes: Number of input planes.
        planes: Number of output planes.
        stride: Stride for convolution.
        act_layer: Activation layer to use.
        downsample_type: Type of downsampling to use.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        act_layer: nn.Module,
        downsample_type: str,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = act_layer(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.downsample = None
        out_planes = self.expansion * planes
        if stride != 1 or in_planes != out_planes:
            if downsample_type == "conv":
                self.downsample = nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            else:
                self.downsample = PoolPad(
                    stride=stride,
                    in_channels=in_planes,
                    out_channels=out_planes,
                    downsample_type=downsample_type,
                )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the BasicBlockCPreAct.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the block.
        """
        shortcut = x

        out = self.act1(self.bn1(x))
        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        out += shortcut
        return out


class ResNetCPreAct(nn.Module):
    """Pre-activation version of the CIFAR ResNet.

    Args:
        block_fn: Block function to use.
        depth: Depth of the network.
        width_multiplier: Width multiplier for the network.
        num_classes: Number of output classes.
        in_chans: Number of input channels.
        downsample_type: Type of downsampling to use.
        act_layer: Activation layer to use.
    """

    def __init__(
        self,
        block_fn: BasicBlockCPreAct,
        depth: int,
        width_multiplier: int,
        num_classes: int,
        in_chans: int,
        downsample_type: str,
        act_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        self.num_features = 64 * block_fn.expansion * width_multiplier
        self.downsample_type = downsample_type

        if (depth - 2) % 6 != 0:
            msg = "Depth should be 6n+2 (e.g., 20, 32, 44, 56, 110, 1202)"
            raise ValueError(msg)

        n = (depth - 2) // 6

        self.conv1 = nn.Conv2d(
            in_chans, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self.make_layer(
            block_fn, 16 * width_multiplier, n, stride=1, act_layer=act_layer
        )
        self.layer2 = self.make_layer(
            block_fn, 32 * width_multiplier, n, stride=2, act_layer=act_layer
        )
        self.layer3 = self.make_layer(
            block_fn, 64 * width_multiplier, n, stride=2, act_layer=act_layer
        )
        self.bn = nn.BatchNorm2d(self.num_features)
        self.act = act_layer(inplace=True)
        self.global_pool = FlattenAdaptiveAvgPool2d()
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def make_layer(
        self,
        block: type[BasicBlockCPreAct],
        planes: int,
        num_blocks: int,
        stride: int,
        act_layer: type[nn.Module],
    ) -> nn.Sequential:
        """Create a layer of blocks.

        Args:
            block: Block type to use.
            planes: Number of output planes.
            num_blocks: Number of blocks in the layer.
            stride: Stride for the first block.
            act_layer: Activation layer to use.

        Returns:
            A sequential container of blocks.
        """
        blocks = nn.Sequential(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                act_layer=act_layer,
                downsample_type=self.downsample_type,
            )
        )

        self.in_planes = planes * block.expansion

        for _ in range(num_blocks - 1):
            blocks.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    stride=1,
                    act_layer=act_layer,
                    downsample_type=self.downsample_type,
                )
            )

        return blocks

    def get_classifier(self, *, name_only: bool = False) -> str | nn.Linear:
        """Get the classifier of the network.

        Args:
            name_only: If True, return only the name of the classifier.

        Returns:
            The classifier or its name.
        """
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes: int) -> None:
        """Reset the classifier with a new number of classes.

        Args:
            num_classes: New number of classes for the classifier.
        """
        self.num_classes = num_classes
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward pass through the feature extraction layers.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after feature extraction.
        """
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.act(out)

        return out

    def forward_head(self, x: Tensor, *, pre_logits: bool = False) -> Tensor:
        """Forward pass through the head of the network.

        Args:
            x: Input tensor.
            pre_logits: If True, return features before the final linear layer.

        Returns:
            Output tensor after passing through the head.
        """
        out = self.global_pool(x)

        return out if pre_logits else self.fc(out)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the entire network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor of the network.
        """
        out = self.forward_features(x)
        out = self.forward_head(out)

        return out


def wide_resnet_c_preact_26_10(
    num_classes: int = 10,
    in_chans: int = 3,
    downsample_type: str = "conv",
    act_layer: nn.Module = nn.ReLU,
) -> ResNetCPreAct:
    """Constructs a WideResNet-28-10 model."""
    model = ResNetCPreAct(
        block_fn=BasicBlockCPreAct,
        depth=26,
        width_multiplier=10,
        num_classes=num_classes,
        in_chans=in_chans,
        downsample_type=downsample_type,
        act_layer=act_layer,
    )

    return model


def resnet_c_preact_26(
    num_classes: int = 10,
    in_chans: int = 3,
    downsample_type: str = "conv",
    act_layer: type[nn.Module] = nn.ReLU,
) -> ResNetCPreAct:
    """Constructs a ResNet-26 model.

    Args:
        num_classes: Number of output classes.
        in_chans: Number of input channels.
        downsample_type: Type of downsampling to use.
        act_layer: Activation layer to use.

    Returns:
        A ResNet-26 model.
    """
    model = ResNetCPreAct(
        block_fn=BasicBlockCPreAct,
        depth=26,
        width_multiplier=1,
        num_classes=num_classes,
        in_chans=in_chans,
        downsample_type=downsample_type,
        act_layer=act_layer,
    )

    return model