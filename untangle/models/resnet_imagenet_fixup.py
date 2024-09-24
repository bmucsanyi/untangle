"""ImageNet ResNet implementation."""

import logging
import math

import torch
from torch import nn

from .utils import FlattenAdaptiveAvgPool2d, PoolPad

logger = logging.getLogger(__name__)


def resnet_fixup_50(
    num_classes=1000,
    in_chans=3,
    downsample_type="conv",
    act_layer=nn.ReLU,
    *,
    init_bias_minus_log_c=False,
):
    """Constructs a ResNet-Fixup-50 model."""
    model = ResNetFixup(
        block_fn=BottleneckFixup,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_chans=in_chans,
        downsample_type=downsample_type,
        act_layer=act_layer,
        init_bias_minus_log_c=init_bias_minus_log_c,
    )

    return model


class BasicBlockFixup(nn.Module):
    """BasicBlock for ImageNet Fixup ResNets."""

    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride,
        downsample,
        act_layer,
    ):
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

    def forward(self, x):
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
    """Bottleneck module for ImageNet Fixup ResNets."""

    expansion = 4

    def __init__(
        self,
        in_planes,
        planes,
        stride,
        downsample,
        act_layer,
    ):
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

    def forward(self, x):
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
    """ImageNet Fixup ResNet."""

    def __init__(
        self,
        block_fn,
        layers,
        num_classes,
        in_chans,
        downsample_type,
        act_layer,
        init_bias_minus_log_c,
    ):
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

        self.init_weights(init_bias_minus_log_c=init_bias_minus_log_c)

    def init_weights(self, *, init_bias_minus_log_c):
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

        if init_bias_minus_log_c:
            nn.init.constant_(self.fc.bias, -math.log(self.num_classes))

    def get_classifier(self, *, name_only=False):
        return "fc" if name_only else lambda x: self.fc(x + self.bias2)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.global_pool = FlattenAdaptiveAvgPool2d()
        self.fc = nn.Linear(self.num_features, self.num_classes, bias=True)
        nn.init.constant_(self.fc.weight, 0)
        nn.init.constant_(self.fc.bias, 0)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.act1(x + self.bias1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward_head(self, x, *, pre_logits: bool = False):
        x = self.global_pool(x)

        return x if pre_logits else self.fc(x + self.bias2)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def make_blocks(
    block_fn,
    channels,
    block_repeats,
    in_planes,
    downsample_type,
    act_layer,
):
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
