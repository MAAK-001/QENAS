"""Hand-designed baselines of the evaluation study: U-Net and Attention U-Net.

Both are the variants of the Mixed-GGNAS comparison tables (Hu et al., 2025, Tables 4-7), identified by their
parameter and FLOP counts, which these implementations reproduce (tests/test_flops.py):

* **U-Net** (Ronneberger et al., MICCAI 2015) in the widely used PyTorch form (github.com/milesial/Pytorch-UNet,
  ``bilinear=False``): channels 64-128-256-512-1024, double 3x3 conv + BN + ReLU, 2x2 max pooling, 2x2 transposed-
  convolution up-sampling -> 31.04 M parameters, 437.9 G FLOPs per batch of 8 at 256 x 256 (Mixed-GGNAS: 31.04 M,
  437.94 G).
* **Attention U-Net** (Oktay et al., MIDL 2018) in the widely used PyTorch form (github.com/LeeJunHyun/
  Image_Segmentation, ``AttU_Net``) with channels 32-64-128-256-512, i.e. half the reference width -> 8.73 M
  parameters (Mixed-GGNAS: 8.73 M); additive attention gates on every skip connection, nearest up-sampling + 3x3
  conv in the decoder.

Output: one logit per pixel (binary segmentation, the QENAS loss). Inputs are padded by nothing: all QENAS image
sizes are divisible by 16.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------------------------- U-Net
class _DoubleConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _Up(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin // 2, 2, stride=2)
        self.conv = _DoubleConv(cin, cout)

    def forward(self, x, skip):
        return self.conv(torch.cat([skip, self.up(x)], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base: int = 64):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]
        self.inc = _DoubleConv(in_channels, c[0])
        self.downs = nn.ModuleList([nn.Sequential(nn.MaxPool2d(2), _DoubleConv(c[i], c[i + 1])) for i in range(4)])
        self.ups = nn.ModuleList([_Up(c[4 - i], c[3 - i]) for i in range(4)])
        self.outc = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        feats = [self.inc(x)]
        for d in self.downs:
            feats.append(d(feats[-1]))
        y = feats[-1]
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            y = up(y, skip)
        return self.outc(y)


# ---------------------------------------------------------------------------------------------- Attention U-Net
class _ConvBlock(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=True), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=True), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _UpConv(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(nn.Upsample(scale_factor=2), nn.Conv2d(cin, cout, 3, padding=1, bias=True),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _AttentionGate(nn.Module):
    """Additive attention gate: x * sigmoid(psi(relu(W_g g + W_x x)))."""

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.w_g(g) + self.w_x(x)))


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base: int = 32):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]
        self.pool = nn.MaxPool2d(2, 2)
        self.enc = nn.ModuleList([_ConvBlock(in_channels, c[0])] + [_ConvBlock(c[i], c[i + 1]) for i in range(4)])
        self.up = nn.ModuleList([_UpConv(c[4 - i], c[3 - i]) for i in range(4)])
        self.att = nn.ModuleList([_AttentionGate(c[3 - i], c[3 - i], c[3 - i] // 2) for i in range(4)])
        self.dec = nn.ModuleList([_ConvBlock(c[4 - i], c[3 - i]) for i in range(4)])
        self.out = nn.Conv2d(c[0], num_classes, 1)

    def forward(self, x):
        feats = [self.enc[0](x)]
        for k in range(1, 5):
            feats.append(self.enc[k](self.pool(feats[-1])))
        d = feats[4]
        for i in range(4):
            d = self.up[i](d)
            skip = self.att[i](g=d, x=feats[3 - i])
            d = self.dec[i](torch.cat([skip, d], dim=1))
        return self.out(d)


BASELINES = {"unet": UNet, "attention_unet": AttentionUNet}


def build_baseline(name: str, in_channels: int = 3, num_classes: int = 1) -> nn.Module:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline '{name}'. Available: {sorted(BASELINES)}")
    return BASELINES[name](in_channels=in_channels, num_classes=num_classes)
