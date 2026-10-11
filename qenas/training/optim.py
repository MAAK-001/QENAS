"""Optimiser and learning-rate schedules (Mixed-GGNAS implementation details: AdamW, lr 1e-3, wd 5e-5).

Each supervised stage has one fixed schedule:
* ``none`` - constant learning rate, used by the 20-epoch learning-capability and loss-weight runs, so that
  the measured improvement is not an artefact of learning-rate annealing;
* ``poly`` - final training, official Mixed-GGNAS schedule (DeepLab "poly"): linear warm-up from 1e-3 x lr
  over ``warmup_epochs``, then ``(1 - t/T) ** 0.9`` decay, stepped every iteration.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


def build_optimizer(model: torch.nn.Module, tcfg: Dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))


def build_scheduler(optimizer: torch.optim.Optimizer, tcfg: Dict[str, Any], steps_per_epoch: int):
    name = tcfg["scheduler"]
    if name == "none":
        return None
    if name != "poly":
        raise ValueError(f"Unknown scheduler '{name}'")
    warm = int(tcfg["warmup_epochs"]) * steps_per_epoch
    total = max(1, int(tcfg["epochs"]) * steps_per_epoch)
    power = float(tcfg["poly_power"])

    def factor(step: int) -> float:
        if warm > 0 and step <= warm:
            a = step / warm
            return 1e-3 * (1 - a) + a
        t = min(1.0, (step - warm) / max(1, total - warm))
        return max(0.0, (1 - t)) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
