"""ImageNet ResNet implementation."""

import logging

import torch
from huggingface_hub import hf_hub_download
from torch import Tensor, nn

from .utils import FlattenAdaptiveAvgPool2d, PoolPad

logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    """BasicBlock for ImageNet ResNets.

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

        self.conv1 = nn.Conv2d(
            in_channels=in_planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=out_planes,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def zero_init_last(self) -> None:
        """Initialize the last BatchNorm layer with zeros."""
        if getattr(self.bn2, "weight", None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the BasicBlock.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the block.
        """
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    """Bottleneck module for ImageNet ResNets.

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

        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(
            in_channels=planes, out_channels=out_planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def zero_init_last(self) -> None:
        """Initialize the last BatchNorm layer with zeros."""
        if getattr(self.bn3, "weight", None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the Bottleneck.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the block.
        """
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.act3(x)

        return x


class ResNet(nn.Module):
    """ImageNet ResNet.

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
        self.conv1 = nn.Conv2d(
            in_chans, in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_planes)
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
        self.fc = nn.Linear(self.num_features, self.num_classes, bias=True)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize the weights of the network."""
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

        for module in self.modules():
            if hasattr(module, "zero_init_last"):
                module.zero_init_last()

    def get_classifier(self, *, name_only: bool = False) -> str | nn.Linear:
        """Get the classifier.

        Args:
            name_only: If True, return only the name of the classifier.

        Returns:
            The classifier name or the classifier module.
        """
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes: int) -> None:
        """Reset the classifier.

        Args:
            num_classes: New number of classes for classification.
        """
        self.num_classes = num_classes
        self.global_pool = FlattenAdaptiveAvgPool2d()
        self.fc = nn.Linear(self.num_features, self.num_classes, bias=True)

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward pass through the feature extraction layers.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after passing through the feature layers.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
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

        return x if pre_logits else self.fc(x)

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
        block_fn: The block function to use (BasicBlock or Bottleneck).
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
                downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_planes,
                        out_channels=out_planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_planes),
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


def resnet_50(
    num_classes: int = 1000,
    in_chans: int = 3,
    downsample_type: str = "conv",
    act_layer: nn.Module = nn.ReLU,
    *,
    pretrained: bool = False,
) -> ResNet:
    """Constructs a ResNet-50 model.

    Args:
        num_classes: Number of classes for classification.
        in_chans: Number of input channels.
        downsample_type: Type of downsampling to use.
        act_layer: Activation layer to use.
        pretrained: Whether to load pretrained weights.

    Returns:
        ResNet model instance.
    """
    model = ResNet(
        block_fn=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_chans=in_chans,
        downsample_type=downsample_type,
        act_layer=act_layer,
    )

    if pretrained:
        pretrained_strict = downsample_type == "conv"
        cached_file = hf_hub_download(
            "timm/resnet50.a1_in1k",
            filename="pytorch_model.bin",
            revision=None,
            library_name="timm",
            library_version="0.9.8dev0",
        )
        state_dict = torch.load(cached_file, map_location="cpu", weights_only=True)
        mismatches = model.load_state_dict(state_dict, strict=pretrained_strict)

        if mismatches.missing_keys:
            logger.warning(
                f"The following keys are missing: {mismatches.missing_keys}."
            )

        if mismatches.unexpected_keys:
            logger.warning(
                f"The following keys are unexpected: {mismatches.unexpected_keys}."
            )

    return model
