"""CIFAR (Wide-)ResNet implementation."""

from torch import nn

from .utils import AvgPoolShortCut


def wide_resnet_c_28_10(
    num_classes=10,
    in_chans=3,
    down_type="conv",
    act_layer=nn.ReLU,
):
    """Constructs a WideResNet-28-10 model."""
    model = ResNetC(
        block_fn=BasicBlockC,
        depth=28,
        width_multiplier=10,
        num_classes=num_classes,
        in_chans=in_chans,
        down_type=down_type,
        act_layer=act_layer,
    )

    return model


class BasicBlockC(nn.Module):
    """BasicBlock for CIFAR ResNets."""

    expansion = 1

    def __init__(self, in_planes, planes, stride, act_layer, down_type):
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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if down_type == "conv":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            elif down_type == "ddu":
                self.shortcut = nn.Sequential(
                    AvgPoolShortCut(
                        stride=stride, out_c=self.expansion * planes, in_c=in_planes
                    )
                )
            else:
                msg = "Invalid downsample type provided."
                raise ValueError(msg)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


class BottleneckC(nn.Module):
    """Bottleneck module for CIFAR ResNets."""

    expansion = 4

    def __init__(self, in_planes, planes, stride, act_layer, down_type):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.act3 = act_layer(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if down_type == "conv":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            elif down_type == "ddu":
                self.shortcut = nn.Sequential(
                    AvgPoolShortCut(
                        stride=stride, out_c=self.expansion * planes, in_c=in_planes
                    )
                )
            else:
                msg = "Invalid downsample type provided"
                raise ValueError(msg)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act3(out)
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
        down_type,
        act_layer,
    ):
        super().__init__()
        self.in_planes = 16
        self.num_classes = num_classes
        self.num_features = 64 * block_fn.expansion * width_multiplier
        self.down_type = down_type

        if (depth - 4) % 6 != 0:
            msg = "Depth should be 6n+4 (e.g., 22, 34, 46, 58, 112, 1204)"
            raise ValueError(msg)

        n = (depth - 2) // 6

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
        self.global_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(self.num_features, self.num_classes)

    def make_layer(self, block, planes, num_blocks, stride, act_layer):
        blocks = nn.Sequential(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                act_layer=act_layer,
                down_type=self.down_type,
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
                    down_type=self.down_type,
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
        out = out.view(out.size(0), -1)

        return out if pre_logits else self.fc(out)

    def forward(self, x):
        out = self.forward_features(x)
        out = self.forward_head(out)

        return out
