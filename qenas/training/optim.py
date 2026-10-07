"""Optimiser and learning-rate schedule factories.

Default (Mixed-GGNAS implementation details): AdamW, lr 1e-3, weight decay 5e-5.

Schedules
* ``none``  — constant learning rate (used for the 20-epoch learning-capability runs,
  so that the measured improvement is not an artefact of learning-rate annealing).
* ``poly``  — official Mixed-GGNAS schedule (DeepLab "poly"): linear warm-up from
  1e-3 x lr over ``warmup_epochs``, then ``(1 - t/T) ** 0.9`` decay, stepped every
  iteration. Justification: warm-up stabilises the first AdamW steps of a network
  trained from scratch; the decay lets the final epochs converge instead of
  oscillating at the initial step size.
* ``cosine`` — optional alternative (per-iteration cosine decay after warm-up).
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch


def build_optimizer(model: torch.nn.Module, tcfg: Dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    name = tcfg["optimizer"].lower()
    lr, wd = float(tcfg["lr"]), float(tcfg["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=float(tcfg.get("momentum", 0.9)), nesterov=True)
    raise ValueError(f"Unknown optimizer '{name}' (adamw | adam | sgd)")


def build_scheduler(optimizer: torch.optim.Optimizer, tcfg: Dict[str, Any], steps_per_epoch: int):
    name = (tcfg.get("scheduler") or "none").lower()
    if name == "none":
        return None
    epochs = int(tcfg["epochs"])
    warm = int(tcfg.get("warmup_epochs", 0)) * steps_per_epoch
    total = max(1, epochs * steps_per_epoch)
    power = float(tcfg.get("poly_power", 0.9))

    def factor(step: int) -> float:
        if warm > 0 and step <= warm:
            a = step / warm
            return 1e-3 * (1 - a) + a
        t = min(1.0, (step - warm) / max(1, total - warm))
        if name == "poly":
            return max(0.0, (1 - t)) ** power
        if name == "cosine":
            return 0.5 * (1 + math.cos(math.pi * t))
        raise ValueError(f"Unknown scheduler '{name}' (none | poly | cosine)")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
