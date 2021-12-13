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
import torch
from torch import nn


class DenseConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 16, growth_channels: int = 16):
        super(DenseConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(int(growth_channels * 1), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(int(growth_channels * 2), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(int(growth_channels * 3), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(int(growth_channels * 4), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(int(growth_channels * 5), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(int(growth_channels * 6), out_channels, (3, 3), (1, 1), (1, 1))
        self.conv8 = nn.Conv2d(int(growth_channels * 7), out_channels, (3, 3), (1, 1), (1, 1))

        self.relu = nn.ReLU(True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.lrelu(self.conv1(x))
        out2 = self.lrelu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.lrelu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.lrelu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.lrelu(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out6 = self.lrelu(self.conv5(torch.cat([x, out1, out2, out3, out4, out5], 1)))
        out7 = self.lrelu(self.conv5(torch.cat([x, out1, out2, out3, out4, out5, out6], 1)))
        out8 = self.lrelu(self.conv5(torch.cat([x, out1, out2, out3, out4, out5, out6, out7], 1)))

        return out8


class SRDenseNet(nn.Module):
    def __init__(self):
        super(SRDenseNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
        )

        # Feature extraction layer
        self.dcb1 = nn.Sequential(DenseConvBlock(128))
        self.dcb2 = nn.Sequential(DenseConvBlock(256))
        self.dcb3 = nn.Sequential(DenseConvBlock(384))
        self.dcb4 = nn.Sequential(DenseConvBlock(512))
        self.dcb5 = nn.Sequential(DenseConvBlock(640))
        self.dcb6 = nn.Sequential(DenseConvBlock(768))
        self.dcb7 = nn.Sequential(DenseConvBlock(896))
        self.dcb8 = nn.Sequential(DenseConvBlock(1024))

        self.conv2 = nn.Conv2d(1152, 256, (1, 1), (1, 1), (0, 0))
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (2, 2), (2, 2), (0, 0)),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, (2, 2), (2, 2), (0, 0)),
            nn.ReLU(True),
        )
        self.conv3 = nn.Conv2d(256, 1, (3, 3), (1, 1), (1, 1))

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
