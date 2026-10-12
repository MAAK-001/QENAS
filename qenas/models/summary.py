"""Model summary: per-position table, parameters and FLOPs.

FLOPs follow the convention of the Mixed-GGNAS tables, reproduced exactly: they are the operation counts of
``thop.profile`` (thop 0.1.1, the de-facto standard of the segmentation-NAS literature) for one *training
batch* - 8 images, IDRiD 4 - in units of 1e9. Verified: the standard U-Net (31.04 M parameters) gives
54.74 G per 256 x 256 image, i.e. 437.94 G per batch of 8 (Mixed-GGNAS Table 4: 437.94) and
54.74 x 2.5 x 4 = 547.42 G per IDRiD batch of four 512 x 320 images (Table 7: 547.42); see tests.

thop 0.1.1 rules reproduced here (``thop/vision/basic_hooks.py``, applied to leaf modules):
* Conv2d / ConvTranspose2d: output elements x (C_in / groups) x kernel area
* Linear: output elements x in_features
* BatchNorm / LayerNorm / InstanceNorm: 2 x input elements, x 2 if affine
* AvgPool2d: output elements; AdaptiveAvgPool2d: (kernel area + 1) x output elements
* ReLU, MaxPool, Dropout, Sequential: 0; modules thop does not know (GELU, Sigmoid, GroupNorm, Identity): 0
  (thop's default for unregistered modules); functional operations (concat, add, multiply): 0.
The per-image value (``gflops_per_image``) is the batch value divided by the batch size.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
import torch.nn as nn

from .chromosome import chromosome_to_str, decode
from .network import QENASNet, count_parameters

_NORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)


@torch.no_grad()
def thop_flops(model: nn.Module, input_shape: Sequence[int]) -> int:
    """Operation count of ``model`` for an input of shape ``input_shape`` (incl. batch), thop 0.1.1 rules."""
    total = [0]

    def hook(m, inp, out):
        x = inp[0]
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            total[0] += out.numel() * (x.shape[1] // m.groups) * m.weight.shape[2] * m.weight.shape[3]
        elif isinstance(m, nn.Linear):
            total[0] += out.numel() * m.in_features
        elif isinstance(m, _NORM):
            affine = getattr(m, "affine", False) or getattr(m, "elementwise_affine", False)
            total[0] += 2 * x.numel() * (2 if affine else 1)
        elif isinstance(m, nn.AvgPool2d):
            total[0] += out.numel()
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            k = (x.shape[2] // out.shape[2]) * (x.shape[3] // out.shape[3])
            total[0] += (k + 1) * out.numel()
        elif isinstance(m, nn.Upsample):
            total[0] += out.numel() * {"nearest": 1, "linear": 5, "bilinear": 11}.get(m.mode, 0)

    hooks = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    was_training = model.training
    model.eval()
    p = next(model.parameters())
    model(torch.zeros(list(input_shape), dtype=p.dtype, device=p.device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return int(total[0])


def cost_summary(model: nn.Module, input_hw: Sequence[int], in_channels: int, batch_size: int) -> Dict[str, Any]:
    """Parameters and FLOPs (thop convention, per training batch and per image) of any network."""
    flops_batch = thop_flops(model, [batch_size, in_channels, int(input_hw[0]), int(input_hw[1])])
    params = count_parameters(model)
    return {"total_params": params, "params_m": params / 1e6, "flops_batch_size": int(batch_size),
            "gflops_mixed_ggnas": flops_batch / 1e9, "gflops_per_image": flops_batch / batch_size / 1e9}


def model_summary(model: QENASNet, input_hw: Sequence[int], in_channels: int, batch_size: int) -> Dict[str, Any]:
    rows = model.position_info(input_hw)
    cost = cost_summary(model, input_hw, in_channels, batch_size)
    return {
        "chromosome": model.chromosome,
        "blocks": decode(model.chromosome),
        "base_channels": model.base_channels,
        "scale": model.scale,
        "darts_genotype": model.genotype_name,
        "input": f"{in_channels}x{input_hw[0]}x{input_hw[1]}",
        "stem_params": count_parameters(model.stem),
        "head_params": count_parameters(model.head),
        "positions": rows,
        **cost,                                            # gflops_mixed_ggnas: per training batch, as Mixed-GGNAS
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
        f"total trainable parameters: {s['total_params']:,d} ({s['params_m']:.3f} M)",
        f"FLOPs (thop): {s['gflops_per_image']:.3f} G per image | {s['gflops_mixed_ggnas']:.2f} G per training "
        f"batch of {s['flops_batch_size']} (Mixed-GGNAS convention)",
    ]
    return "\n".join(lines)
