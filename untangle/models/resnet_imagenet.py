"""ImageNet ResNet implementation."""

import logging
import math

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from .utils import FlattenAdaptiveAvgPool2d, PoolPad

logger = logging.getLogger(__name__)


def resnet_50(
    num_classes=1000,
    in_chans=3,
    downsample_type="conv",
    act_layer=nn.ReLU,
    *,
    pretrained=False,
    init_bias_minus_log_c=False,
):
    """Constructs a ResNet-50 model."""
    model = ResNet(
        block_fn=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_chans=in_chans,
        downsample_type=downsample_type,
        act_layer=act_layer,
        init_bias_minus_log_c=init_bias_minus_log_c,
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


class BasicBlock(nn.Module):
    """BasicBlock for ImageNet ResNets."""

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

    def zero_init_last(self):
        if getattr(self.bn2, "weight", None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
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
    """Bottleneck module for ImageNet ResNets."""

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

    def zero_init_last(self):
        if getattr(self.bn3, "weight", None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
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
    """ImageNet ResNet."""

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

        self.init_weights(init_bias_minus_log_c=init_bias_minus_log_c)

    def init_weights(self, *, init_bias_minus_log_c):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

        for module in self.modules():
            if hasattr(module, "zero_init_last"):
                module.zero_init_last()

        if init_bias_minus_log_c:
            nn.init.constant_(self.fc.bias, -math.log(self.num_classes))

    def get_classifier(self, *, name_only=False):
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.global_pool = FlattenAdaptiveAvgPool2d()
        self.fc = nn.Linear(self.num_features, self.num_classes, bias=True)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward_head(self, x, *, pre_logits: bool = False):
        x = self.global_pool(x)

        return x if pre_logits else self.fc(x)

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
