"""The four manually designed Mixed-GGNAS blocks.

Source: official Mixed-GGNAS implementation
(https://github.com/Hmxki/Mixed-GGNAS, ``Mixed_GGNAS/cell_module/cells/raw_cells.py``)
and Table 1 of Hu et al., *Expert Systems with Applications* 289 (2025) 128338.

Faithfully reproduced (layer order, grouped convolutions with ``groups=16``,
dense growth rate 32 / 3 layers, Inception strip-convolution branches, ConvNeXt
inverted bottleneck with 4x expansion, BatchNorm, activations, residual paths).

Convolution *scale* variants (Table 1; the paper names them 3 / 5 / 7):

=============  ======================  ======================  ======================
Block          scale 3                 scale 5                 scale 7
=============  ======================  ======================  ======================
Residual       3x3, dil 1, pad 1       3x3, dil 3, pad 3       5x5, dil 3, pad 6
Dense          3x3, dil 1, pad 1       3x3, dil 3, pad 3       5x5, dil 3, pad 6
Inception      strips (3, 5, 7)        strips (5, 7, 9)        strips (3, 7, 11)
ConvNeXt       depthwise 5x5           depthwise 7x7           depthwise 9x9
=============  ======================  ======================  ======================

QENAS modification (documented in DESIGN_DECISIONS.md): the official code keeps
all three scale variants (plus residual copies) active and learns mixing weights
with *supervised* gradient descent, later collapsing to one variant. QENAS uses a
single, fixed, configured scale in search, candidate training and final training
alike (no scale selection, no collapse), so every block below instantiates
exactly one variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn

GROUPS = 16  # official grouped-convolution setting


def _check_channels(c: int) -> None:
    if c % GROUPS:
        raise ValueError(f"Manual blocks need channels divisible by {GROUPS} (official groups=16); got {c}")


def _residual_dense_conv(cin: int, cout: int, scale: int) -> nn.Conv2d:
    if scale == 3:
        return nn.Conv2d(cin, cout, 3, stride=1, padding=1, bias=False, groups=GROUPS)
    if scale == 5:
        return nn.Conv2d(cin, cout, 3, stride=1, padding=3, dilation=3, bias=False, groups=GROUPS)
    if scale == 7:
        return nn.Conv2d(cin, cout, 5, stride=1, padding=6, dilation=3, bias=False, groups=GROUPS)
    raise ValueError(f"invalid scale {scale}")


class ResidualBlock(nn.Module):
    """He et al. (2016) basic residual block, official Mixed-GGNAS variant (grouped convs)."""

    kind = "Residual"

    def __init__(self, in_channels: int, out_channels: int, scale: int = 3):
        super().__init__()
        _check_channels(in_channels)
        _check_channels(out_channels)
        self.scale = scale
        self.conv1 = _residual_dense_conv(in_channels, out_channels, scale)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = _residual_dense_conv(out_channels, out_channels, scale)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv_res = (nn.Conv2d(in_channels, out_channels, 1, bias=False)
                         if in_channels != out_channels else None)
        self.relu3 = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.conv_res is None else self.conv_res(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return self.relu3(residual + out)

    def describe(self) -> str:
        return {3: "2x[3x3 gconv]", 5: "2x[3x3 d3 gconv]", 7: "2x[5x5 d3 gconv]"}[self.scale] + " + identity"


class DenseBlock(nn.Module):
    """Huang et al. (2017) dense block, official variant: 3 BN-ReLU-gconv layers, growth 32, 1x1 fusion."""

    kind = "Dense"

    def __init__(self, in_channels: int, out_channels: int, scale: int = 3, growth_rate: int = 32,
                 num_layers: int = 3):
        super().__init__()
        _check_channels(in_channels)
        self.scale = scale
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            cin = in_channels + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(cin),
                nn.ReLU(inplace=False),
                _residual_dense_conv(cin, growth_rate, scale),
            ))
        self.adjusts = nn.Conv2d(in_channels + num_layers * growth_rate, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return self.adjusts(torch.cat(features, dim=1))

    def describe(self) -> str:
        k = {3: "3x3", 5: "3x3 d3", 7: "5x5 d3"}[self.scale]
        return f"3 dense layers ({k} gconv, growth 32) + 1x1"


_INCEPTION_KERNELS = {3: (3, 5, 7), 5: (5, 7, 9), 7: (3, 7, 11)}


class InceptionBlock(nn.Module):
    """Inception block (Szegedy et al.) with factorised strip convolutions, official variant."""

    kind = "Inception"

    def __init__(self, in_channels: int, out_channels: int, scale: int = 3):
        super().__init__()
        _check_channels(in_channels)
        _check_channels(out_channels)
        self.scale = scale
        c = out_channels
        self.conv1 = nn.Conv2d(in_channels, c, 3, padding=1, groups=GROUPS)
        self.bn0 = nn.BatchNorm2d(c)
        self.relu0 = nn.ReLU(inplace=False)
        self.branches = nn.ModuleList()
        for k in _INCEPTION_KERNELS[scale]:
            p = k // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(c, c, (1, k), padding=(0, p), groups=GROUPS),
                nn.Conv2d(c, c, (k, 1), padding=(p, 0), groups=GROUPS),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=False),
            ))
        self.adjust = nn.Conv2d(c * 3, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu0(self.bn0(self.conv1(x)))
        return self.adjust(torch.cat([b(x) for b in self.branches], dim=1))

    def describe(self) -> str:
        ks = _INCEPTION_KERNELS[self.scale]
        return "3x3 gconv -> strips " + "/".join(f"1x{k}+{k}x1" for k in ks) + " -> 1x1"


_CONVNEXT_KERNEL = {3: 5, 5: 7, 7: 9}


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style block (Liu et al., 2022; Ying et al., 2023), official Mixed-GGNAS variant.

    depthwise kxk -> BN -> pointwise 4x expansion (Linear) -> GELU -> pointwise projection
    -> BN -> GELU(residual + x) -> 1x1 adjust.
    """

    kind = "ConvNeXt"

    def __init__(self, in_channels: int, out_channels: int, scale: int = 3):
        super().__init__()
        self.scale = scale
        ks = _CONVNEXT_KERNEL[scale]
        self.ks = ks
        self.dwconv = nn.Conv2d(in_channels, in_channels, ks, padding=(ks - 1) // 2, groups=in_channels)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.GELU()
        self.adjust = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.bn0(self.dwconv(x))
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = self.bn1(x.permute(0, 3, 1, 2))
        return self.adjust(self.act1(residual + x))

    def describe(self) -> str:
        return f"dw {self.ks}x{self.ks} -> 4x MLP -> residual -> 1x1"


MANUAL_BLOCKS = {0: ResidualBlock, 1: DenseBlock, 2: InceptionBlock, 3: ConvNeXtBlock}
