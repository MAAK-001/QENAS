"""Model summary: per-position table, parameter count and multiply-accumulate (MAC) count."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .chromosome import chromosome_to_str, decode
from .network import QENASNet, count_parameters


@torch.no_grad()
def count_macs(model: nn.Module, input_shape: Sequence[int]) -> int:
    """MACs of Conv/ConvTranspose/Linear layers for one input (hook based; norms/activations ignored)."""
    total = [0]
    hooks = []

    def conv_hook(m, inp, out):
        k = m.weight.shape[2] * m.weight.shape[3]
        cin_per_group = m.in_channels // m.groups
        if isinstance(m, nn.ConvTranspose2d):
            # every input position scatters into k positions for each output channel
            total[0] += inp[0].numel() // inp[0].shape[0] * (m.out_channels // m.groups) * k
        else:
            total[0] += out.numel() // out.shape[0] * cin_per_group * k

    def lin_hook(m, inp, out):
        total[0] += out.numel() // out.shape[0] * m.in_features

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    p = next(model.parameters())
    model(torch.zeros([1] + list(input_shape), dtype=p.dtype, device=p.device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return int(total[0])


def model_summary(model: QENASNet, input_hw: Sequence[int], in_channels: int = 3) -> Dict[str, Any]:
    rows = model.position_info(input_hw)
    stem = count_parameters(model.stem)
    head = count_parameters(model.head)
    macs = count_macs(model, [in_channels, int(input_hw[0]), int(input_hw[1])])
    return {
        "chromosome": model.chromosome,
        "blocks": decode(model.chromosome),
        "base_channels": model.base_channels,
        "scale": model.scale,
        "darts_genotype": model.genotype_name,
        "input": f"{in_channels}x{input_hw[0]}x{input_hw[1]}",
        "stem_params": stem,
        "head_params": head,
        "positions": rows,
        "total_params": count_parameters(model),
        "macs": macs,
        "gmacs": macs / 1e9,
    }


def summary_text(s: Dict[str, Any]) -> str:
    lines = [
        f"QENAS architecture  chromosome {chromosome_to_str(s['chromosome'])}",
        f"blocks: {' -> '.join(s['blocks'])}",
        f"input {s['input']} | base channels {s['base_channels']} | convolution scale {s['scale']} | "
        f"DARTS genotype {s['darts_genotype']}",
        "",
        f"{'pos':5s} {'block':10s} {'channels':>11s} {'resolution':>19s} {'params':>10s}  kernel / scale",
        f"{'stem':5s} {'3x3 conv':10s} {'':>11s} {'':>19s} {s['stem_params']:>10,d}",
    ]
    for r in s["positions"]:
        lines.append(f"{r['position']:5s} {r['block']:10s} {r['in_channels']:>4d}->{r['out_channels']:<5d} "
                     f"{r['in_resolution']:>9s}->{r['out_resolution']:<9s} {r['params']:>10,d}  {r['kernel_scale']}")
    lines += [
        f"{'head':5s} {'1x1 conv':10s} {'':>11s} {'':>19s} {s['head_params']:>10,d}",
        "",
        f"total trainable parameters: {s['total_params']:,d} ({s['total_params'] / 1e6:.3f} M)",
        f"multiply-accumulates per image: {s['gmacs']:.3f} GMACs",
    ]
    return "\n".join(lines)
