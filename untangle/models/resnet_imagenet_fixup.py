"""ImageNet ResNet implementation."""

import logging
import math

import torch
from torch import Tensor, nn

from .utils import FlattenAdaptiveAvgPool2d, PoolPad

logger = logging.getLogger(__name__)


class BasicBlockFixup(nn.Module):
    """BasicBlock for ImageNet Fixup ResNets.

    Args:
        in_planes: Number of input channels.
        planes: Number of output channels.
        stride: Stride for the first convolutional layer.
        downsample: Optional downsampling module.
        act_layer: Activation layer constructor.
    """

    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        downsample: nn.Module | None,
        act_layer: nn.Module,
    ) -> None:
        super().__init__()

        out_planes = planes * self.expansion

        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.act1 = act_layer(inplace=True)

        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=out_planes,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the BasicBlockFixup.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the block.
        """
        shortcut = x

        x = self.conv1(x + self.bias1a)
        x = self.act1(x + self.bias1b)

        x = self.conv2(x + self.bias2a)
        x = x * self.scale + self.bias2b

        if self.downsample is not None:
            shortcut = self.downsample(shortcut + self.bias1a)

        x += shortcut
        x = self.act2(x)

        return x


class BottleneckFixup(nn.Module):
    """Bottleneck module for ImageNet Fixup ResNets.

    Args:
        in_planes: Number of input channels.
        planes: Number of intermediate channels.
        stride: Stride for the second convolutional layer.
        downsample: Optional downsampling module.
        act_layer: Activation layer constructor.
    """

    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        downsample: nn.Module | None,
        act_layer: nn.Module,
    ) -> None:
        super().__init__()

        out_planes = planes * self.expansion

        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=planes, kernel_size=1, bias=False
        )
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.act1 = act_layer(inplace=True)

        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            bias=False,
        )
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.act2 = act_layer(inplace=True)

        self.bias3a = nn.Parameter(torch.zeros(1))
        self.conv3 = nn.Conv2d(
            in_channels=planes, out_channels=out_planes, kernel_size=1, bias=False
        )
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3b = nn.Parameter(torch.zeros(1))

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the BottleneckFixup.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the block.
        """
        shortcut = x

        x = self.conv1(x + self.bias1a)
        x = self.act1(x + self.bias1b)

        x = self.conv2(x + self.bias2a)
        x = self.act2(x + self.bias2b)

        x = self.conv3(x + self.bias3a)
        x = x * self.scale + self.bias3b

        if self.downsample is not None:
            shortcut = self.downsample(shortcut + self.bias1a)

        x += shortcut
        x = self.act3(x)

        return x


class ResNetFixup(nn.Module):
    """ImageNet Fixup ResNet.

    Args:
        block_fn: Type of residual block to use.
        layers: List of number of blocks in each layer.
        num_classes: Number of classes for classification.
        in_chans: Number of input channels.
        downsample_type: Type of downsampling to use.
        act_layer: Activation layer constructor.
    """

    def __init__(
        self,
        block_fn: nn.Module,
        layers: list[int],
        num_classes: int,
        in_chans: int,
        downsample_type: str,
        act_layer: nn.Module,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Stem
        in_planes = 64
        self.num_layers = sum(layers)
        self.conv1 = nn.Conv2d(
            in_chans, in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.act1 = act_layer(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules = make_blocks(
            block_fn=block_fn,
            channels=channels,
            block_repeats=layers,
            in_planes=in_planes,
            downsample_type=downsample_type,
            act_layer=act_layer,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc

        # Head
        self.num_features = 512 * block_fn.expansion
        self.global_pool = FlattenAdaptiveAvgPool2d()
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(self.num_features, self.num_classes, bias=True)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of the network."""
        for module in self.modules():
            if isinstance(module, BasicBlockFixup):
                weight1 = module.conv1.weight
                base_std1 = math.sqrt(
                    2 / (weight1.shape[0] * math.prod(weight1.shape[2:]))
                )
                multiplier = self.num_layers ** (-0.5)
                nn.init.normal_(
                    weight1,
                    mean=0,
                    std=base_std1 * multiplier,
                )
                nn.init.constant_(module.conv2.weight, 0.0)

                if isinstance(module.downsample, nn.Conv2d):
                    weight_downsample = module.downsample.weight
                    std_downsample = math.sqrt(
                        2
                        / (
                            weight_downsample.shape[0]
                            * math.prod(weight_downsample.shape[2:])
                        )
                    )
                    nn.init.normal_(
                        weight_downsample,
                        mean=0,
                        std=std_downsample,
                    )
            elif isinstance(module, BottleneckFixup):
                weight1 = module.conv1.weight
                base_std1 = math.sqrt(
                    2 / (weight1.shape[0] * math.prod(weight1.shape[2:]))
                )
                weight2 = module.conv2.weight
                base_std2 = math.sqrt(
                    2 / (weight2.shape[0] * math.prod(weight2.shape[2:]))
                )
                multiplier = self.num_layers ** (-0.5)
                nn.init.normal_(
                    weight1,
                    mean=0,
                    std=base_std1 * multiplier,
                )
                nn.init.normal_(
                    weight2,
                    mean=0,
                    std=base_std2 * multiplier,
                )
                nn.init.constant_(module.conv3.weight, 0.0)

                if isinstance(module.downsample, nn.Conv2d):
                    weight_downsample = module.downsample.weight
                    std_downsample = math.sqrt(
                        2
                        / (
                            weight_downsample.shape[0]
                            * math.prod(weight_downsample.shape[2:])
                        )
                    )
                    nn.init.normal_(
                        weight_downsample,
                        mean=0,
                        std=std_downsample,
                    )
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0)
                nn.init.constant_(module.bias, 0)

    def get_classifier(self, *, name_only: bool = False) -> None:
        """Get the classifier.

        Args:
            name_only: If True, return only the name of the classifier.

        Returns:
            The classifier function or its name.
        """
        return "fc" if name_only else lambda x: self.fc(x + self.bias2)

    def reset_classifier(self, num_classes: int) -> None:
        """Reset the classifier.

        Args:
            num_classes: New number of classes for classification.
        """
        self.num_classes = num_classes
        self.global_pool = FlattenAdaptiveAvgPool2d()
        self.fc = nn.Linear(self.num_features, self.num_classes, bias=True)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward pass through the feature extraction layers.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the feature layers.
        """
        x = self.conv1(x)
        x = self.act1(x + self.bias1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward_head(self, x: Tensor, *, pre_logits: bool = False) -> Tensor:
        """Forward pass through the head of the network.

        Args:
            x: Input tensor.
            pre_logits: If True, return features before the final linear layer.

        Returns:
            Output tensor after passing through the head.
        """
        x = self.global_pool(x)

        return x if pre_logits else self.fc(x + self.bias2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the entire network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the network.
        """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def make_blocks(
    block_fn: nn.Module,
    channels: list[int],
    block_repeats: list[int],
    in_planes: int,
    downsample_type: str,
    act_layer: nn.Module,
) -> list[tuple[str, nn.Sequential]]:
    """Create blocks for ResNet stages.

    Args:
        block_fn: The block function to use (BasicBlockFixup or BottleneckFixup).
        channels: List of channel sizes for each stage.
        block_repeats: List of number of blocks in each stage.
        in_planes: Number of input channels.
        downsample_type: Type of downsampling to use ('conv' or other).
        act_layer: Activation layer to use.

    Returns:
        List of tuples containing stage names and corresponding sequential blocks.
    """
    stages = []

    for stage_idx, (planes, num_blocks) in enumerate(
        zip(channels, block_repeats, strict=False)
    ):
        stage_name = f"layer{stage_idx + 1}"
        stride = 1 if stage_idx == 0 else 2

        downsample = None
        out_planes = planes * block_fn.expansion
        if stride != 1 or in_planes != out_planes:
            if downsample_type == "conv":
                downsample = nn.Conv2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            else:
                downsample = PoolPad(
                    stride=stride,
                    in_channels=in_planes,
                    out_channels=out_planes,
                    downsample_type=downsample_type,
                )

        blocks = []
        for block_idx in range(num_blocks):
            if block_idx > 0:
                downsample = None
                stride = 1

            blocks.append(
                block_fn(
                    in_planes=in_planes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    act_layer=act_layer,
                )
            )
            in_planes = planes * block_fn.expansion

        stages.append((stage_name, nn.Sequential(*blocks)))

    return stages


def resnet_fixup_50(
    num_classes: int = 1000,
    in_chans: int = 3,
    downsample_type: str = "conv",
    act_layer: nn.Module = nn.ReLU,
) -> ResNetFixup:
    """Constructs a ResNet-Fixup-50 model.

    Args:
        num_classes: Number of classes for classification.
        in_chans: Number of input channels.
        downsample_type: Type of downsampling to use.
        act_layer: Activation layer to use.

    Returns:
        ResNetFixup model instance.
    """
    model = ResNetFixup(
        block_fn=BottleneckFixup,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_chans=in_chans,
        downsample_type=downsample_type,
        act_layer=act_layer,
    )

    return model
