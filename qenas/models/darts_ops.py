"""Primitive operations of the DARTS block (NAS-Unet search space).

Source: NAS-Unet (Weng et al., IEEE Access 2019, https://github.com/tianbaochou/NasUnet,
``models/ops`` / ``util/prim_ops_set.py``) as vendored by the official Mixed-GGNAS
code (``Mixed_GGNAS/cell_module/cells/prim_ops_set.py``), which states that it
"adopt[s] the DARTS-based method used in NAS-Unet".

Faithfully reproduced: operation set and naming, ``weight -> norm -> act`` order,
GroupNorm with 2 channels per group, 1x1 preprocessing with ``act -> weight -> norm``,
stride-2 down-sampling / transposed up-sampling operations, channel weighting (SE-style).

QENAS modifications (behaviour-preserving):
* ``ZeroOp`` allocates its output with ``x.new_zeros`` (the original allocates a
  float32 CPU tensor, which breaks float64 SynFlow evaluation and GPU placement).
* Code is restructured into small classes; numerics are unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _gn(channels: int, affine: bool = True) -> nn.GroupNorm:
    groups = 1 if channels % 2 != 0 else channels // 2
    return nn.GroupNorm(groups, channels, affine=affine)


def _same_padding(k: int) -> int:
    if k % 2 == 0:
        raise ValueError("kernel size should be odd")
    return k // 2


class ConvOps(nn.Module):
    """Conv / depthwise-separable / transposed conv with GroupNorm and ReLU (NAS-Unet ``ConvOps``)."""

    def __init__(self, cin: int, cout: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1,
                 use_transpose: bool = False, output_padding: int = 1, use_depthwise: bool = False,
                 affine: bool = True, ops_order: str = "weight_norm_act", dropout_rate: float = 0.0):
        super().__init__()
        self.ops_list = ops_order.split("_")
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.use_transpose, self.use_depthwise = use_transpose, use_depthwise
        norm_before_weight = self.ops_list.index("norm") < self.ops_list.index("weight")
        self.norm = _gn(cin if norm_before_weight else cout, affine)
        self.activation = nn.ReLU(inplace=self.ops_list[0] != "act")
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        pad = _same_padding(kernel_size) * dilation
        if use_transpose:
            if use_depthwise:
                self.depth_conv = nn.ConvTranspose2d(cin, cout, kernel_size, stride=stride, padding=pad,
                                                     output_padding=output_padding, groups=cin, bias=False)
                self.point_conv = nn.Conv2d(cin, cout, 1, bias=False)
            else:
                self.conv = nn.ConvTranspose2d(cin, cout, kernel_size, stride=stride, padding=pad,
                                               output_padding=output_padding, dilation=dilation, bias=False)
        else:
            if use_depthwise:
                self.depth_conv = nn.Conv2d(cin, cin, kernel_size, stride=stride, padding=pad,
                                            dilation=dilation, groups=cin, bias=False)
                self.point_conv = nn.Conv2d(cin, cout, 1, bias=False)
            else:
                self.conv = nn.Conv2d(cin, cout, kernel_size, stride=stride, padding=pad,
                                      dilation=dilation, bias=False)

    def weight_call(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_depthwise:
            return self.point_conv(self.depth_conv(x))
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.ops_list:
            if op == "weight":
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.weight_call(x)
            elif op == "norm":
                x = self.norm(x)
            elif op == "act":
                x = self.activation(x)
        return x


class CWeightOp(nn.Module):
    """Channel weighting (squeeze-excitation); optional stride-2 conv / transposed conv + GroupNorm."""

    def __init__(self, cin: int, cout: int, kernel_size: int = 3, stride: int = 1, use_transpose: bool = False,
                 output_padding: int = 0, affine: bool = True):
        super().__init__()
        self.stride = stride
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(cin, cin // 2), nn.ReLU(inplace=True), nn.Linear(cin // 2, cout), nn.Sigmoid())
        pad = _same_padding(kernel_size)
        if stride >= 2:
            if use_transpose:
                self.conv = nn.ConvTranspose2d(cin, cout, kernel_size, stride=stride, padding=pad,
                                               output_padding=output_padding, bias=False)
            else:
                self.conv = nn.Conv2d(cin, cout, kernel_size, stride=stride, padding=pad, bias=False)
            self.norm = _gn(cout, affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return self.norm(self.conv(x * y)) if self.stride >= 2 else x * y


class PoolingOp(nn.Module):
    def __init__(self, pool_type: str, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        pad = _same_padding(kernel_size) if stride == 1 else 0
        if pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=pad, count_include_pad=False)
        elif pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=pad)
        else:
            raise ValueError(pool_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class IdentityOp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ZeroOp(nn.Module):
    def __init__(self, stride: int = 1):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        return x.new_zeros((n, c, h // self.stride, w // self.stride))


# name -> factory(c). Identical to the official OPS table (dropout 0, affine True).
OPS = {
    "none": lambda c: ZeroOp(1),
    "identity": lambda c: IdentityOp(),
    "cweight": lambda c: CWeightOp(c, c),
    "dil_conv": lambda c: ConvOps(c, c, dilation=2),
    "dep_conv": lambda c: ConvOps(c, c, use_depthwise=True),
    "shuffle_conv": lambda c: ConvOps(c, c),
    "conv": lambda c: ConvOps(c, c),            # official: has_shuffle=True with groups=1 -> no-op shuffle
    "avg_pool": lambda c: PoolingOp("avg"),
    "max_pool": lambda c: PoolingOp("max"),
    "down_cweight": lambda c: CWeightOp(c, c, stride=2),
    "down_dil_conv": lambda c: ConvOps(c, c, stride=2, dilation=2),
    "down_dep_conv": lambda c: ConvOps(c, c, stride=2, use_depthwise=True),
    "down_conv": lambda c: ConvOps(c, c, stride=2),
    "up_cweight": lambda c: CWeightOp(c, c, stride=2, use_transpose=True, output_padding=1),
    "up_dep_conv": lambda c: ConvOps(c, c, stride=2, use_depthwise=True, use_transpose=True),
    "up_conv": lambda c: ConvOps(c, c, stride=2, use_transpose=True),
    "up_dil_conv": lambda c: ConvOps(c, c, stride=2, dilation=2, use_transpose=True),
}

DOWN_OPS = {"avg_pool", "max_pool", "down_cweight", "down_dil_conv", "down_dep_conv", "down_conv"}
UP_OPS = {"up_cweight", "up_dep_conv", "up_conv", "up_dil_conv"}
NORMAL_OPS = {"identity", "none", "cweight", "dil_conv", "dep_conv", "shuffle_conv", "conv"}
