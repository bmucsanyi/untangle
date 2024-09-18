"""CIFAR (Wide-)ResNet implementation."""

import math

from torch import nn

from .utils import FlattenAdaptiveAvgPool2d, PoolPad


def wide_resnet_c_26_10(
    num_classes=10,
    in_chans=3,
    downsample_type="conv",
    act_layer=nn.ReLU,
    *,
    init_bias_minus_log_c=False,
):
    """Constructs a WideResNet-26-10 model."""
    model = ResNetC(
        block_fn=BasicBlockC,
        depth=26,
        width_multiplier=10,
        num_classes=num_classes,
        in_chans=in_chans,
        downsample_type=downsample_type,
        act_layer=act_layer,
        init_bias_minus_log_c=init_bias_minus_log_c,
    )

    return model


class BasicBlockC(nn.Module):
    """BasicBlock for CIFAR ResNets."""

    expansion = 1

    def __init__(self, in_planes, planes, stride, act_layer, downsample_type):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = act_layer(inplace=True)

        self.downsample = None
        out_planes = self.expansion * planes
        if stride != 1 or in_planes != out_planes:
            if downsample_type == "conv":
                self.downsample = nn.Sequential(
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
                self.downsample = PoolPad(
                    stride=stride,
                    in_channels=in_planes,
                    out_channels=out_planes,
                    downsample_type=downsample_type,
                )

    def forward(self, x):
        shortcut = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        out += shortcut
        out = self.act2(out)
        return out


class ResNetC(nn.Module):
    """CIFAR ResNet."""

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

        if downsample_type not in {"conv", "bed_of_nails_pad", "avg_pad"}:
            msg = f"Invalid option '{downsample_type}' provided"
            raise ValueError(msg)
        self.downsample_type = downsample_type

        self.conv1 = nn.Conv2d(
            in_chans, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
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
        self.fc = nn.Linear(self.num_features, self.num_classes)

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
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def forward_features(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out

    def forward_head(self, x, *, pre_logits: bool = False):
        out = self.global_pool(x)

        return out if pre_logits else self.fc(out)

    def forward(self, x):
        out = self.forward_features(x)
        out = self.forward_head(out)

        return out
