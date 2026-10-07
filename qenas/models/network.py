"""QENAS segmentation network: a U-shaped network assembled from a chromosome.

Topology (b = base channels, H x W input)::

    input ─ stem 3x3 (b, H) ─────────────────────────────────────────────── skip E0 ─┐
             enc1 (2b, H/2) ───────────────────────────────────────── skip E1 ─┐     │
              enc2 (4b, H/4) ─────────────────────────────── skip E2 ─┐        │     │
               enc3 (8b, H/8) ───────────────────── skip E3 ─┐        │        │     │
                enc4 (16b, H/16) ─ dec1 (8b, H/8) ◄─────────-┘        │        │     │
                                    dec2 (4b, H/4) ◄─────────────────-┘        │     │
                                     dec3 (2b, H/2) ◄─────────────────────────-┘     │
                                      dec4 (b, H) ◄─────────────────────────────────-┘
                                       1x1 conv -> 1 logit per pixel

Each of the 8 positions is either a manual block wrapped by the official
Mixed-GGNAS transition, or a DARTS down/up cell:

* encoder transition (official): BN -> AvgPool 2x2 -> 3x3 grouped conv (C_in -> C_out) -> block
* decoder transition (official, without the ViT branch): BN -> 2x2 transposed conv (C_in -> C_out);
  skip -> BN -> spatial-channel attention; concat -> 1x1 conv -> block
* DARTS positions: NAS-Unet cells (see ``darts_cell.py``)

The *same* class and constructor arguments are used for SynFlow evaluation,
candidate training and final training — there is no search/train mismatch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import MANUAL_BLOCKS
from .chromosome import BLOCK_NAMES, POSITION_NAMES, validate_chromosome
from .darts_cell import GENOTYPES, DartsCell, resolve_genotype_name


class SpatialAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.squeeze = nn.Conv2d(dim, 1, 1, bias=False)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.squeeze(x))


class ChannelAttention(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1)
        self.conv2 = nn.Conv2d(dim // reduction, dim, 1)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv2(F.relu(self.conv1(self.pool(x)))))


class EncoderPosition(nn.Module):
    def __init__(self, block_id: int, cin: int, cout: int, scale: int):
        super().__init__()
        self.block_id = block_id
        self.transition = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cin, cout, 3, padding=1, groups=16),
        )
        self.block = MANUAL_BLOCKS[block_id](cout, cout, scale=scale)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        return self.block(self.transition(s1))


class DecoderPosition(nn.Module):
    def __init__(self, block_id: int, cin: int, cout: int, scale: int):
        super().__init__()
        self.block_id = block_id
        self.bn_in = nn.BatchNorm2d(cin)
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2)
        self.bn_skip = nn.BatchNorm2d(cout)
        self.satt = SpatialAttention(cout)
        self.catt = ChannelAttention(cout, reduction=4)
        self.fuse = nn.Conv2d(2 * cout, cout, 1)
        self.block = MANUAL_BLOCKS[block_id](cout, cout, scale=scale)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(self.bn_in(x))
        if up.shape[-2:] != skip.shape[-2:]:
            dy, dx = skip.shape[-2] - up.shape[-2], skip.shape[-1] - up.shape[-1]
            if dy < 0 or dx < 0 or dy > 1 or dx > 1:
                raise RuntimeError(f"decoder/skip size mismatch {tuple(up.shape)} vs {tuple(skip.shape)}")
            up = F.pad(up, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        s = self.bn_skip(skip)
        # Spatial-channel attention enhancement: s + s * sigma_s(s) * sigma_c(s).
        # (The official code multiplies two attended copies, s^2 * sigma_s * sigma_c; see DESIGN_DECISIONS.md D7.)
        s = s + s * self.satt.gate(s) * self.catt.gate(s)
        return self.block(self.fuse(torch.cat([s, up], dim=1)))


class DartsEncoderPosition(nn.Module):
    block_id = 4

    def __init__(self, genotype, genotype_name: str, c_prev_prev: int, c_prev: int, cout: int, s0_stride: int):
        super().__init__()
        if cout % 4:
            raise ValueError("DARTS cell output channels must be divisible by 4")
        self.cell = DartsCell(genotype, c_prev_prev, c_prev, cout // 4, "down", s0_stride=s0_stride,
                              genotype_name=genotype_name)

    def forward(self, s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        return self.cell(s0, s1)


class DartsDecoderPosition(nn.Module):
    block_id = 4

    def __init__(self, genotype, genotype_name: str, c_skip: int, cin: int, cout: int):
        super().__init__()
        self.cell = DartsCell(genotype, c_skip, cin, cout // 4, "up", s0_stride=1, genotype_name=genotype_name)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.cell(skip, x)


class QENASNet(nn.Module):
    def __init__(self, chromosome: Sequence[int], in_channels: int = 3, num_classes: int = 1,
                 base_channels: int = 32, scale: int = 3, genotype_name: str = "darts_cell_busi"):
        super().__init__()
        self.chromosome = validate_chromosome(chromosome)
        self.base_channels = b = int(base_channels)
        self.scale = int(scale)
        self.genotype_name = genotype_name
        genotype = GENOTYPES[genotype_name]
        # Feature channels: E0 (stem) .. E4 (bottleneck)
        self.enc_channels = [b, 2 * b, 4 * b, 8 * b, 16 * b]
        self.stem = nn.Sequential(nn.Conv2d(in_channels, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU())
        self.encoder = nn.ModuleList()
        for k in range(4):
            g = self.chromosome[k]
            cin, cout = self.enc_channels[k], self.enc_channels[k + 1]
            if g == 4:
                cpp = self.enc_channels[k - 1] if k > 0 else self.enc_channels[0]
                self.encoder.append(DartsEncoderPosition(genotype, genotype_name, cpp, cin, cout,
                                                         s0_stride=1 if k == 0 else 2))
            else:
                self.encoder.append(EncoderPosition(g, cin, cout, self.scale))
        self.decoder = nn.ModuleList()
        for j in range(4):
            g = self.chromosome[4 + j]
            cin, cout = self.enc_channels[4 - j], self.enc_channels[3 - j]
            if g == 4:
                self.decoder.append(DartsDecoderPosition(genotype, genotype_name, cout, cin, cout))
            else:
                self.decoder.append(DecoderPosition(g, cin, cout, self.scale))
        self.head = nn.Conv2d(b, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [self.stem(x)]
        for k, pos in enumerate(self.encoder):
            s1 = feats[-1]
            s0 = feats[-2] if len(feats) >= 2 else feats[-1]
            feats.append(pos(s0, s1))
        y = feats[4]
        for j, pos in enumerate(self.decoder):
            y = pos(y, feats[3 - j])
        return self.head(y)

    # ------------------------------------------------------------------ description
    def position_info(self, input_hw: Sequence[int] = (256, 256)) -> List[Dict[str, Any]]:
        h, w = input_hw
        rows = []
        for k, pos in enumerate(self.encoder):
            g = self.chromosome[k]
            module = pos.cell if g == 4 else pos.block
            rows.append({
                "position": POSITION_NAMES[k], "gene": g, "block": BLOCK_NAMES[g],
                "in_channels": self.enc_channels[k], "out_channels": self.enc_channels[k + 1],
                "in_resolution": f"{h >> k}x{w >> k}", "out_resolution": f"{h >> (k + 1)}x{w >> (k + 1)}",
                "kernel_scale": module.describe(),
                "params": count_parameters(pos),
            })
        for j, pos in enumerate(self.decoder):
            g = self.chromosome[4 + j]
            module = pos.cell if g == 4 else pos.block
            rows.append({
                "position": POSITION_NAMES[4 + j], "gene": g, "block": BLOCK_NAMES[g],
                "in_channels": self.enc_channels[4 - j], "out_channels": self.enc_channels[3 - j],
                "in_resolution": f"{h >> (4 - j)}x{w >> (4 - j)}", "out_resolution": f"{h >> (3 - j)}x{w >> (3 - j)}",
                "kernel_scale": module.describe(),
                "params": count_parameters(pos),
            })
        return rows


def count_parameters(module: nn.Module) -> int:
    """Total number of trainable parameters (QENAS objective 1)."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def build_model(chromosome: Sequence[int], cfg: Dict[str, Any]) -> QENASNet:
    m = cfg["model"]
    gname = resolve_genotype_name(m["darts_genotype"], cfg["data"]["dataset"])
    return QENASNet(chromosome, in_channels=int(cfg["data"]["in_channels"]), num_classes=int(m["num_classes"]),
                    base_channels=int(m["base_channels"]), scale=int(m["scale"]), genotype_name=gname)


def architecture_signature(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Everything (besides the chromosome) that determines the architecture."""
    m = cfg["model"]
    gname = resolve_genotype_name(m["darts_genotype"], cfg["data"]["dataset"])
    g = GENOTYPES[gname]
    return {
        "in_channels": int(cfg["data"]["in_channels"]),
        "num_classes": int(m["num_classes"]),
        "base_channels": int(m["base_channels"]),
        "scale": int(m["scale"]),
        "darts_genotype": gname,
        "darts_genotype_def": {"down": [list(e) for e in g.down], "down_concat": list(g.down_concat),
                               "up": [list(e) for e in g.up], "up_concat": list(g.up_concat)},
        "network_version": 1,
    }
