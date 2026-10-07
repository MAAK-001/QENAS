"""Binary segmentation metrics, computed per image and then averaged over the dataset.

Prediction: foreground where sigmoid(logit) > 0.5 (i.e. logit > 0).

* Dice  = 2|P ∩ G| / (|P| + |G|)           (foreground)
* IoU   = |P ∩ G| / |P ∪ G|                 (foreground Jaccard)
* mIoU  = (IoU_foreground + IoU_background) / 2   (class-mean IoU, as reported by Mixed-GGNAS)

Convention for an empty ground truth *and* empty prediction: Dice = IoU = 1
(a perfect prediction). Counts are accumulated in float64.
"""

from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def per_image_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    pred = (logits.detach() > 0).flatten(1).double()
    gt = (target.detach() > 0.5).flatten(1).double()
    inter = (pred * gt).sum(1)
    ps, gs = pred.sum(1), gt.sum(1)
    union = ps + gs - inter
    n = pred.shape[1]
    both_empty = (ps + gs) == 0
    dice = torch.where(both_empty, torch.ones_like(inter), 2 * inter / (ps + gs).clamp_min(1e-12))
    iou = torch.where(both_empty, torch.ones_like(inter), inter / union.clamp_min(1e-12))
    # background
    inter_bg = n - union                       # pixels that are background in both
    union_bg = n - inter
    iou_bg = torch.where(union_bg == 0, torch.ones_like(inter), inter_bg / union_bg.clamp_min(1e-12))
    return {"dice": dice, "iou": iou, "miou": 0.5 * (iou + iou_bg)}


class MetricAccumulator:
    """Accumulates per-image metrics and per-sample losses; reports dataset means."""

    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.n = 0

    def update(self, per_sample_loss: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> None:
        m = per_image_metrics(logits, target)
        m["loss"] = per_sample_loss.detach().double()
        for k, v in m.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v.sum().item())
        self.n += int(per_sample_loss.shape[0])

    def compute(self) -> Dict[str, float]:
        if self.n == 0:
            raise RuntimeError("no samples were evaluated (empty loader)")
        return {k: v / self.n for k, v in self.sums.items()}
