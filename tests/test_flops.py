"""The FLOPs counter reproduces the Mixed-GGNAS tables (thop 0.1.1 convention, per training batch)."""

import pytest
import torch
import torch.nn as nn

from qenas.config import build_config
from qenas.models.network import build_model
from qenas.models.summary import model_summary, thop_flops


class _DoubleConv(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))


class _Up(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.up = nn.ConvTranspose2d(i, i // 2, 2, stride=2)
        self.conv = _DoubleConv(i, o)

    def forward(self, x1, x2):
        return self.conv(torch.cat([x2, self.up(x1)], 1))


class UNet(nn.Module):
    """The standard U-Net (31.04 M parameters) used as the U-Net baseline in Mixed-GGNAS Tables 4-7."""

    def __init__(self):
        super().__init__()
        self.inc = _DoubleConv(3, 64)
        self.downs = nn.ModuleList([nn.Sequential(nn.MaxPool2d(2), _DoubleConv(a, b))
                                    for a, b in ((64, 128), (128, 256), (256, 512), (512, 1024))])
        self.ups = nn.ModuleList([_Up(a, b) for a, b in ((1024, 512), (512, 256), (256, 128), (128, 64))])
        self.out = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        feats = [self.inc(x)]
        for d in self.downs:
            feats.append(d(feats[-1]))
        y = feats[-1]
        for u, skip in zip(self.ups, reversed(feats[:-1])):
            y = u(y, skip)
        return self.out(y)


def test_unet_reproduces_mixed_ggnas_table_values():
    m = UNet()
    assert sum(p.numel() for p in m.parameters()) / 1e6 == pytest.approx(31.04, abs=0.01)
    # Table 4/5 (256 x 256, batch 8): 437.94 G;  Table 7 (IDRiD 512 x 320, batch 4): 547.42 G
    assert thop_flops(m, [8, 3, 256, 256]) / 1e9 == pytest.approx(437.94, abs=0.01)
    assert thop_flops(m, [4, 3, 320, 512]) / 1e9 == pytest.approx(547.42, abs=0.03)


def test_model_summary_flops_scale_with_batch():
    cfg = build_config("BUSI", overrides={"data.image_size": [64, 64], "model.base_channels": 16})
    m = build_model([0, 1, 2, 3, 4, 3, 2, 1], cfg)
    s = model_summary(m, [64, 64], 3, 8)
    assert s["gflops_mixed_ggnas"] == pytest.approx(8 * s["gflops_per_image"])
    assert s["total_params"] == sum(p.numel() for p in m.parameters())
