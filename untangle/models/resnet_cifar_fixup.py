"""CIFAR (Wide-)ResNet Fixup implementation."""

import math

import torch
from torch import nn

from .utils import FlattenAdaptiveAvgPool2d, PoolPad


def wide_resnet_c_fixup_26_10(
    num_classes=10,
    in_chans=3,
    downsample_type="conv",
    act_layer=nn.ReLU,
    *,
    init_bias_minus_log_c=False,
    no_zero_init_fc_weight=False,
):
    """Constructs a WideResNet-Fixup-26-10 model."""
    model = ResNetCFixup(
        block_fn=BasicBlockCFixup,
        depth=26,
        width_multiplier=10,
        num_classes=num_classes,
        in_chans=in_chans,
        downsample_type=downsample_type,
        act_layer=act_layer,
        init_bias_minus_log_c=init_bias_minus_log_c,
        no_zero_init_fc_weight=no_zero_init_fc_weight,
    )

    return model


class BasicBlockCFixup(nn.Module):
    """BasicBlock for CIFAR Fixup ResNets."""

    expansion = 1

    def __init__(self, in_planes, planes, stride, act_layer, downsample_type):
        super().__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.act1 = act_layer(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.act2 = act_layer(inplace=True)

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

    def forward(self, x):
        shortcut = x

        out = self.conv1(x + self.bias1a)
        out = self.act1(out + self.bias1b)
        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            shortcut = self.downsample(shortcut + self.bias1a)

        out += shortcut
        out = self.act2(out)
        return out


class ResNetCFixup(nn.Module):
    """CIFAR Fixup ResNet."""

    def __init__(
        self,
        block_fn,
        depth,
        width_multiplier,
        num_classes,
        in_chans,
        downsample_type,
        act_layer,
        init_bias_minus_log_c,
        no_zero_init_fc_weight,
    ):
        super().__init__()

        self.in_planes = 16
        self.num_classes = num_classes
        self.num_features = 64 * block_fn.expansion * width_multiplier
        self.downsample_type = downsample_type

        if (depth - 2) % 6 != 0:
            msg = "Depth should be 6n+2 (e.g., 20, 32, 44, 56, 110, 1202)"
            raise ValueError(msg)

        n = (depth - 2) // 6
        self.num_layers = 3 * n

        if downsample_type not in {"conv", "bed_of_nails_pad", "avg_pad"}:
            msg = f"Invalid option '{downsample_type}' provided"
            raise ValueError(msg)
        self.downsample_type = downsample_type

        self.conv1 = nn.Conv2d(
            in_chans, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.act1 = act_layer(inplace=True)
        self.layer1 = self.make_layer(
            block_fn, 16 * width_multiplier, n, stride=1, act_layer=act_layer
        )
        self.layer2 = self.make_layer(
            block_fn, 32 * width_multiplier, n, stride=2, act_layer=act_layer
        )
        self.layer3 = self.make_layer(
            block_fn, 64 * width_multiplier, n, stride=2, act_layer=act_layer
        )
        self.global_pool = FlattenAdaptiveAvgPool2d()
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.fc = nn.Linear(self.num_features, self.num_classes)
        self.init_weights(
            init_bias_minus_log_c=init_bias_minus_log_c,
            no_zero_init_fc_weight=no_zero_init_fc_weight,
        )

    def init_weights(self, *, init_bias_minus_log_c, no_zero_init_fc_weight):
        for module in self.modules():
            if isinstance(module, BasicBlockCFixup):
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
            elif isinstance(module, nn.Linear):
                if not no_zero_init_fc_weight:
                    nn.init.constant_(module.weight, 0.0)
                nn.init.constant_(module.bias, 0.0)

        if init_bias_minus_log_c:
            nn.init.constant_(self.fc.bias, -math.log(self.num_classes))

    def make_layer(self, block, planes, num_blocks, stride, act_layer):
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

    def get_classifier(self, *, name_only=False):
        return "fc" if name_only else lambda x: self.fc(x + self.bias2)

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def forward_features(self, x):
        out = self.conv1(x)
        out = self.act1(out + self.bias1)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out

    def forward_head(self, x, *, pre_logits: bool = False):
        out = self.global_pool(x)

        return out if pre_logits else self.fc(out + self.bias2)

    def forward(self, x):
        out = self.forward_features(x)
        out = self.forward_head(out)

        return out
