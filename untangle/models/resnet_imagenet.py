"""ImageNet ResNet implementation."""

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from .utils import AvgPoolShortCut, FlattenAdaptiveAvgPool2d


def resnet50(
    num_classes=1000,
    in_chans=3,
    down_type="conv",
    act_layer=nn.ReLU,
    *,
    pretrained=False,
    pretrained_strict=True,
):
    """Constructs a ResNet-50 model."""
    model = ResNet(
        block_fn=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_chans=in_chans,
        down_type=down_type,
        act_layer=act_layer,
    )

    if pretrained:
        cached_file = hf_hub_download(
            "timm/resnet50.a1_in1k",
            filename="pytorch_model.bin",
            revision=None,
            library_name="timm",
            library_version="0.9.8dev0",
        )
        state_dict = torch.load(cached_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=pretrained_strict)

    return model


class BasicBlock(nn.Module):
    """BasicBlock for ImageNet ResNets."""

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride,
        downsample,
        act_layer,
        first_dilation,
        dilation,
    ):
        super().__init__()

        reduce_first = 1

        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(
            inplanes,
            first_planes,
            kernel_size=3,
            stride=stride,
            padding=first_dilation,
            dilation=first_dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes,
            outplanes,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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
        inplanes,
        planes,
        stride,
        downsample,
        act_layer,
        first_dilation,
        dilation,
    ):
        super().__init__()

        width = planes
        first_planes = width
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes,
            width,
            kernel_size=3,
            stride=stride,
            padding=first_dilation,
            dilation=first_dilation,
            groups=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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
        down_type,
        act_layer,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Stem
        inplanes = 64
        self.conv1 = nn.Conv2d(
            in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.act1 = act_layer(inplace=True)
        self.feature_info = [{"num_chs": inplanes, "reduction": 2, "module": "act1"}]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Feature Blocks
        channels = [64, 128, 256, 512]
        stage_modules, stage_feature_info = make_blocks(
            block_fn=block_fn,
            channels=channels,
            block_repeats=layers,
            inplanes=inplanes,
            down_type=down_type,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        # Head
        self.num_features = 512 * block_fn.expansion
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes
        )

        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        for m in self.modules():
            if hasattr(m, "zero_init_last"):
                m.zero_init_last()

    def get_classifier(self, *, name_only=False):
        return "fc" if name_only else self.fc

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(
            self.num_features, self.num_classes
        )

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
    inplanes,
    down_type,
    act_layer,
):
    stages = []
    feature_info = []
    net_block_idx = 0
    net_stride = 4
    output_stride = 32
    down_kernel_size = 1

    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks) in enumerate(
        zip(channels, block_repeats, strict=False)
    ):
        stage_name = f"layer{stage_idx + 1}"
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = {
                "in_channels": inplanes,
                "out_channels": planes * block_fn.expansion,
                "kernel_size": down_kernel_size,
                "stride": stride,
                "dilation": dilation,
                "first_dilation": prev_dilation,
            }

            if down_type == "conv":
                downsample = downsample_conv(**down_kwargs)
            elif down_type == "ddu":
                downsample = downsample_ddu(**down_kwargs)
            else:
                msg = f"Invalid down_type '{down_type}' provided."
                raise ValueError(msg)

        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            blocks.append(
                block_fn(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    act_layer=act_layer,
                    first_dilation=prev_dilation,
                    dilation=dilation,
                )
            )
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append({
            "num_chs": inplanes,
            "reduction": net_stride,
            "module": stage_name,
        })

    return stages, feature_info


def create_pool():
    global_pool = FlattenAdaptiveAvgPool2d()

    return global_pool


def create_fc(num_features, num_classes):
    fc = nn.Linear(num_features, num_classes, bias=True)

    return fc


def create_classifier(
    num_features: int,
    num_classes: int,
):
    global_pool = create_pool(
        num_features=num_features,
    )
    fc = create_fc(
        num_features=num_features,
        num_classes=num_classes,
    )

    return global_pool, fc


def get_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def downsample_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    dilation,
    first_dilation,
):
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=p,
            dilation=first_dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    )


def downsample_ddu(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    dilation,
    first_dilation,
):
    del kernel_size, dilation, first_dilation
    return nn.Sequential(
        AvgPoolShortCut(stride=stride, out_c=out_channels, in_c=in_channels)
    )
