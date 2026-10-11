"""Binary segmentation metrics, computed per image and then averaged over the dataset.

Prediction: foreground where sigmoid(logit) > 0.5 (i.e. logit > 0). P = prediction, G = ground truth.

Mixed-GGNAS metrics (Hu et al., 2025, Sec. 4.1: Dice, mIoU, parameters, FLOPs - the last two are computed in
``models/summary.py``):

* Dice (= Dice similarity coefficient, DSC) = 2|P ∩ G| / (|P| + |G|)        (foreground)
* IoU  = |P ∩ G| / |P ∪ G|                                                    (foreground Jaccard)
* mIoU = (IoU_foreground + IoU_background) / 2                                (class mean, as Mixed-GGNAS)

Additional metrics of the final evaluation:

* Sensitivity (recall, true-positive rate) = |P ∩ G| / |G|. Undefined for an empty ground truth; such images
  are excluded from its mean (none of the three datasets has empty ground truths).
* HD95 - 95th-percentile Hausdorff distance between the boundaries of P and G, in pixels of the evaluated
  resolution: max(P95 of d(boundary P -> boundary G), P95 of d(boundary G -> boundary P)), the definition of
  the Metrics Reloaded reference implementation (Maier-Hein et al., Nat. Methods 2024). Boundary = mask
  minus its erosion with the 4-connected structuring element; distances are Euclidean (exact distance
  transform). Both masks empty -> 0. Exactly one empty -> the image diagonal (the largest possible
  distance); such images are counted separately (``hd95_empty_pred``).

Convention for an empty ground truth *and* empty prediction: Dice = IoU = 1 (a perfect prediction).
Counts are accumulated in float64.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

_STRUCT = generate_binary_structure(2, 1)


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
    inter_bg = n - union                       # pixels that are background in both
    union_bg = n - inter
    iou_bg = torch.where(union_bg == 0, torch.ones_like(inter), inter_bg / union_bg.clamp_min(1e-12))
    has_gt = gs > 0
    sens = torch.where(has_gt, inter / gs.clamp_min(1e-12), torch.full_like(inter, float("nan")))
    return {"dice": dice, "iou": iou, "miou": 0.5 * (iou + iou_bg), "sensitivity": sens}


def _boundary(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, structure=_STRUCT, border_value=0)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance (pixels) between two binary 2-D masks."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return float(math.hypot(*pred.shape))
    bp, bg = _boundary(pred), _boundary(gt)
    d_to_g = distance_transform_edt(~bg)
    d_to_p = distance_transform_edt(~bp)
    return float(max(np.percentile(d_to_g[bp], 95), np.percentile(d_to_p[bg], 95)))


class MetricAccumulator:
    """Accumulates per-image metrics and per-sample losses; reports dataset means.

    ``boundary=True`` adds HD95 (computed on the CPU per image; used for every evaluation pass, not for the
    augmented training batches)."""

    def __init__(self, boundary: bool = False) -> None:
        self.boundary = boundary
        self.sums: Dict[str, float] = {}
        self.n = 0
        self.n_sens = 0
        self.n_hd_empty_pred = 0

    def update(self, per_sample_loss: torch.Tensor, logits: torch.Tensor, target: torch.Tensor) -> None:
        m = per_image_metrics(logits, target)
        sens = m.pop("sensitivity")
        valid = ~torch.isnan(sens)
        self.sums["sensitivity"] = self.sums.get("sensitivity", 0.0) + float(sens[valid].sum().item())
        self.n_sens += int(valid.sum().item())
        m["loss"] = per_sample_loss.detach().double()
        for k, v in m.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v.sum().item())
        if self.boundary:
            p = (logits.detach() > 0).cpu().numpy()[:, 0]
            g = (target.detach() > 0.5).cpu().numpy()[:, 0]
            for pi, gi in zip(p, g):
                if gi.any() and not pi.any():
                    self.n_hd_empty_pred += 1
                self.sums["hd95"] = self.sums.get("hd95", 0.0) + hd95(pi, gi)
        self.n += int(per_sample_loss.shape[0])

    def compute(self) -> Dict[str, float]:
        if self.n == 0:
            raise RuntimeError("no samples were evaluated (empty loader)")
        out = {k: v / self.n for k, v in self.sums.items() if k != "sensitivity"}
        out["sensitivity"] = self.sums["sensitivity"] / self.n_sens if self.n_sens else float("nan")
        if self.boundary:
            out["hd95_empty_pred"] = self.n_hd_empty_pred
        return out
