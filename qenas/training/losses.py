r"""QENAS loss:  L = alpha * BCE + beta * soft-mIoU loss,  alpha + beta = 1.

* BCE: pixel-wise binary cross-entropy on logits (numerically stable
  ``binary_cross_entropy_with_logits``), averaged over the pixels of each image.
* soft-mIoU loss for binary segmentation: the *mean IoU over the two classes*
  (foreground and background), computed with soft (probabilistic) set operations
  (Rahman & Wang, 2016):

      p = sigmoid(logits),   g = ground truth in {0, 1}
      IoU_fg = (sum p g + s) / (sum (p + g - p g) + s)
      IoU_bg = (sum (1-p)(1-g) + s) / (sum ((1-p) + (1-g) - (1-p)(1-g)) + s)
      L_mIoU = 1 - (IoU_fg + IoU_bg) / 2

  computed per image and averaged over the batch. "mIoU" here is the class-mean IoU
  that the Mixed-GGNAS paper reports. The Laplace smoothing ``s = 1`` keeps the
  loss well defined, and near zero, for a correctly predicted empty mask.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_iou(p: torch.Tensor, g: torch.Tensor, smooth: float) -> torch.Tensor:
    dims = tuple(range(1, p.dim()))
    inter = (p * g).sum(dims)
    union = (p + g - p * g).sum(dims)
    return (inter + smooth) / (union + smooth)


class BCESoftMIoULoss(nn.Module):
    def __init__(self, alpha: float = 0.5, smooth: float = 1.0):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.alpha = float(alpha)
        self.beta = 1.0 - self.alpha
        self.smooth = float(smooth)

    def per_sample(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = logits.float()
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(1).mean(1)
        p = torch.sigmoid(logits)
        iou_fg = soft_iou(p, target, self.smooth)
        iou_bg = soft_iou(1.0 - p, 1.0 - target, self.smooth)
        miou_loss = 1.0 - 0.5 * (iou_fg + iou_bg)
        loss = self.alpha * bce + self.beta * miou_loss
        return loss, {"bce": bce, "soft_miou_loss": miou_loss}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.per_sample(logits, target)[0].mean()
