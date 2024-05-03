__copyright__ = """
    SLAMcore Limited
    All Rights Reserved.
    (C) Copyright 2024

    NOTICE:

    All information contained herein is, and remains the property of SLAMcore
    Limited and its suppliers, if any. The intellectual and technical concepts
    contained herein are proprietary to SLAMcore Limited and its suppliers and
    may be covered by patents in process, and are protected by trade secret or
    copyright law.
"""

__license__ = "CC BY-NC-SA 3.0"

import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Sequence
from collections import OrderedDict

import torch
from torch import nn, Tensor


__all__ = ["MobileNetBackbone", "mobilenet_v3_large", "mobilenet_v3_small"]


model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406])
PIXEL_STD = torch.tensor([0.229, 0.224, 0.225])


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNormActivation(torch.nn.Sequential):
    """
    Configurable block used for Convolution-Normalzation-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalzation-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolutiuon layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: Optional[bool] = None,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class SElayer(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
            self,
            input_channels: int,
            squeeze_channels: int,
            activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
            scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class SqueezeExcitation(SElayer):
    """DEPRECATED"""

    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        super().__init__(input_channels, squeeze_channels, scale_activation=nn.Hardsigmoid)
        self.relu = self.activation
        delattr(self, "activation")
        warnings.warn(
            "This SqueezeExcitation class is deprecated since 0.12 and will be removed in 0.14. "
            "Use torchvision.ops.SqueezeExcitation instead.",
            FutureWarning,
        )


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
            self,
            input_channels: int,
            kernel: int,
            expanded_channels: int,
            out_channels: int,
            use_se: bool,
            activation: str,
            stride: int,
            dilation: int,
            width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
            self,
            cnf: InvertedResidualConfig,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        # only use residual connection when there is no dimension change (both spatial and channel-wise)
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand along channel dimension
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetBackbone(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[InvertedResidualConfig],
            input_channels=3,
            **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            input_channels 3 for RGB, 1 for depth
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(inverted_residual_setting, Sequence)
                and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        assert len(inverted_residual_setting) in [11, 15], "Only support small and large"

        block = InvertedResidual
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.firstconv = ConvNormActivation(input_channels,
                                            firstconv_output_channels,
                                            kernel_size=3,
                                            stride=2,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.Hardswish)

        # building inverted residual blocks
        # 5 stages both for small and large
        if len(inverted_residual_setting) == 15:  # large
            self.stage1 = nn.Sequential(*[block(cnf, norm_layer) for cnf in inverted_residual_setting[0:2]])  # 1/4
            self.stage2 = nn.Sequential(*[block(cnf, norm_layer) for cnf in inverted_residual_setting[2:4]])  # 1/8
            self.stage3 = nn.Sequential(*[block(cnf, norm_layer) for cnf in inverted_residual_setting[4:7]])  # 1/16
            self.stage4 = nn.Sequential(*[block(cnf, norm_layer) for cnf in inverted_residual_setting[7:13]])  # 1/16
            self.stage5 = nn.Sequential(*[block(cnf, norm_layer) for cnf in inverted_residual_setting[13:]])  # 1/16
        else:  # small
            self.stage1 = nn.Sequential(*[block(cnf, norm_layer) for cnf in inverted_residual_setting[0:1]])  # 1/4
            self.stage2 = nn.Sequential(*[block(cnf, norm_layer) for cnf in inverted_residual_setting[1:2]])  # 1/8
            self.stage3 = nn.Sequential(*[block(cnf, norm_layer) for cnf in inverted_residual_setting[2:4]])  # 1/16
            self.stage4 = nn.Sequential(*[block(cnf, norm_layer) for cnf in inverted_residual_setting[4:9]])  # 1/16
            self.stage5 = nn.Sequential(*[block(cnf, norm_layer) for cnf in inverted_residual_setting[9:]])  # 1/16

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.lastconv = ConvNormActivation(lastconv_input_channels,
                                           lastconv_output_channels,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=nn.Hardswish)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_first_conv(self, x):
        # Note that bn and activation is already included
        return self.firstconv(x)

    def forward_stage1(self, x):
        return self.stage1(x)

    def forward_stage2(self, x):
        return self.stage2(x)

    def forward_stage3(self, x):
        return self.stage3(x)

    def forward_stage4(self, x):
        return self.stage4(x)

    def forward_stage5(self, x):
        return self.stage5(x)

    def forward_last_conv(self, x):
        # Note that bn and activation is already included
        return self.lastconv(x)

    def forward(self, x: Tensor) -> Tensor:
        x0 = self.forward_first_conv(x)
        x1 = self.forward_stage1(x0)
        x2 = self.forward_stage2(x1)
        x3 = self.forward_stage3(x2)
        x4 = self.forward_stage4(x3)
        x5 = self.forward_stage5(x4)
        return self.lastconv(x5)


def _mobilenet_v3_conf(
        arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1  # default is True

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1  1/4
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2  1/8
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3  1/16
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4 1/16 dilation=2 -> stride=1
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4 dilation=2 -> stride=1
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def mobilenet_v3_large(pretrained: bool = False,
                       progress: bool = True,
                       input_channels: int = 3,
                       dilated: bool = True) -> MobileNetBackbone:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        input_channels (int): input channel dim
        progress (bool): If True, displays a progress bar of the download to stderr
        dilated (bool): whether use dilated or not, default is True
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, _ = _mobilenet_v3_conf(arch, dilated=dilated)
    model = MobileNetBackbone(inverted_residual_setting, input_channels=input_channels)
    if pretrained:
        assert model_urls.get(arch, None) is not None, f"No checkpoint is available for model type {arch}"
        state_dict = torch.load("pretrained/mobilenet_v3_large.pth")  # 312
        state_dict = state_dict_key_mapping_large(state_dict)  # 308
        if input_channels == 1:
            state_dict["firstconv.0.weight"] = torch.sum(state_dict["firstconv.0.weight"], dim=1, keepdim=True)
        # print(state_dict)
        model.load_state_dict(state_dict)

    return model


def state_dict_key_mapping_large(state_dict):
    mapping_dict = {"features.0": "firstconv",
                    "features.1": "stage1.0", "features.2": "stage1.1",
                    "features.3": "stage2.0", "features.4": "stage2.1",
                    "features.5": "stage3.0", "features.6": "stage3.1", "features.7": "stage3.2",
                    "features.8": "stage4.0", "features.9": "stage4.1", "features.10": "stage4.2", "features.11": "stage4.3", "features.12": "stage4.4", "features.13": "stage4.5",
                    "features.14": "stage5.0", "features.15": "stage5.1",
                    "features.16": "lastconv"}

    updated_state_dict = OrderedDict()
    for key in state_dict:
        prefix = ".".join(key.split(".")[:2])
        if prefix in mapping_dict:
            new_key = key.replace(prefix, mapping_dict[prefix])
            updated_state_dict[new_key] = state_dict[key]

    return updated_state_dict


def mobilenet_v3_small(pretrained: bool = False,
                       progress: bool = True,
                       input_channels: int = 3,
                       dilated: bool = True) -> MobileNetBackbone:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        input_channels (int): input channel dim
        progress (bool): If True, displays a progress bar of the download to stderr
        dilated (bool): whether use dilated or not, default is True
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, _ = _mobilenet_v3_conf(arch, dilated=dilated)
    model = MobileNetBackbone(inverted_residual_setting, input_channels=input_channels)
    if pretrained:
        assert model_urls.get(arch, None) is not None, f"No checkpoint is available for model type {arch}"
        state_dict = torch.load("pretrained/mobilenet_v3_large.pth")  # 312
        state_dict = state_dict_key_mapping_small(state_dict)  # 308
        if input_channels == 1:
            state_dict["firstconv.0.weight"] = torch.sum(state_dict["firstconv.0.weight"], dim=1, keepdim=True)
        # print(state_dict)
        model.load_state_dict(state_dict)

    return model


def state_dict_key_mapping_small(state_dict):
    mapping_dict = {"features.0": "firstconv",
                    "features.1": "stage1.0",
                    "features.2": "stage2.0",
                    "features.3": "stage3.0", "features.4": "stage3.1",
                    "features.5": "stage4.0", "features.6": "stage4.1", "features.7": "stage4.2", "features.8": "stage4.3", "features.9": "stage4.4",
                    "features.10": "stage5.0", "features.11": "stage5.1",
                    "features.12": "lastconv"}

    updated_state_dict = OrderedDict()
    for key in state_dict:
        prefix = ".".join(key.split(".")[:2])
        if prefix in mapping_dict:
            new_key = key.replace(prefix, mapping_dict[prefix])
            updated_state_dict[new_key] = state_dict[key]

    return updated_state_dict


if __name__ == "__main__":
    rgb_encoder = mobilenet_v3_large(pretrained=True, input_channels=3, dilated=True)
    mobilenet_v3_small(pretrained=True, input_channels=3, dilated=True)
