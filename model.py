# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
from typing import Any

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "SRDenseNet",
    "srdensenet_x2", "srdensenet_x3", "srdensenet_x4", "srdensenet_x8",
]


class _DenseConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, growth_channels: int):
        super(_DenseConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(int(growth_channels * 1), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(int(growth_channels * 2), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(int(growth_channels * 3), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(int(growth_channels * 4), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(int(growth_channels * 5), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(int(growth_channels * 6), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(int(growth_channels * 7), out_channels, (3, 3), (1, 1), (1, 1))

        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.relu(self.conv1(x))

        out2 = self.relu(self.conv2(out1))
        out2_concat = torch.cat([out1, out2], 1)

        out3 = self.relu(self.conv3(out2_concat))
        out3_concat = torch.cat([out1, out2, out3], 1)

        out4 = self.relu(self.conv4(out3_concat))
        out4_concat = torch.cat([out1, out2, out3, out4], 1)

        out5 = self.relu(self.conv5(out4_concat))
        out5_concat = torch.cat([out1, out2, out3, out4, out5], 1)

        out6 = self.relu(self.conv6(out5_concat))
        out6_concat = torch.cat([out1, out2, out3, out4, out5, out6], 1)

        out7 = self.relu(self.conv7(out6_concat))
        out7_concat = torch.cat([out1, out2, out3, out4, out5, out6, out7], 1)

        out8 = self.relu(self.conv8(out7_concat))
        out8_concat = torch.cat([out1, out2, out3, out4, out5, out6, out7, out8], 1)

        return out8_concat


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.ConvTranspose2d(channels, 256, (upscale_factor, upscale_factor), (upscale_factor, upscale_factor), (0, 0)), nn.ReLU(True),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out


class SRDenseNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            upscale_factor: int
    ) -> None:
        super(SRDenseNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

        # Feature extraction layer
        self.dcb1 = nn.Sequential(_DenseConvBlock(128, 16, 16))
        self.dcb2 = nn.Sequential(_DenseConvBlock(256, 16, 16))
        self.dcb3 = nn.Sequential(_DenseConvBlock(384, 16, 16))
        self.dcb4 = nn.Sequential(_DenseConvBlock(512, 16, 16))
        self.dcb5 = nn.Sequential(_DenseConvBlock(640, 16, 16))
        self.dcb6 = nn.Sequential(_DenseConvBlock(768, 16, 16))
        self.dcb7 = nn.Sequential(_DenseConvBlock(896, 16, 16))
        self.dcb8 = nn.Sequential(_DenseConvBlock(1024, 16, 16))

        self.conv2 = nn.Conv2d(1152, 256, (1, 1), (1, 1), (0, 0))

        # Upscale block
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(256, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(256, 3))
        self.upsampling = nn.Sequential(*upsampling)

        self.conv3 = nn.Conv2d(256, out_channels, (3, 3), (1, 1), (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)

        # Feature
        dcb1 = self.dcb1(out)
        out1 = torch.cat([dcb1, out], 1)

        dcb2 = self.dcb2(out1)
        out2 = torch.cat([dcb2, out1], 1)

        dcb3 = self.dcb3(out2)
        out3 = torch.cat([dcb3, out2], 1)

        dcb4 = self.dcb4(out3)
        out4 = torch.cat([dcb4, out3], 1)

        dcb5 = self.dcb5(out4)
        out5 = torch.cat([dcb5, out4], 1)

        dcb6 = self.dcb6(out5)
        out6 = torch.cat([dcb6, out5], 1)

        dcb7 = self.dcb7(out6)
        out7 = torch.cat([dcb7, out6], 1)

        dcb8 = self.dcb8(out7)
        out8 = torch.cat([dcb8, out7], 1)

        out = self.conv2(out8)
        out = self.upsampling(out)
        out = self.conv3(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


def srdensenet_x2(**kwargs: Any) -> SRDenseNet:
    model = SRDenseNet(upscale_factor=2, **kwargs)

    return model


def srdensenet_x3(**kwargs: Any) -> SRDenseNet:
    model = SRDenseNet(upscale_factor=3, **kwargs)

    return model


def srdensenet_x4(**kwargs: Any) -> SRDenseNet:
    model = SRDenseNet(upscale_factor=4, **kwargs)

    return model


def srdensenet_x8(**kwargs: Any) -> SRDenseNet:
    model = SRDenseNet(upscale_factor=8, **kwargs)

    return model
