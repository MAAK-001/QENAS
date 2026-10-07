"""Early stopping that triggers on *overfitting*, never on mere stagnation.

Training is stopped at epoch t only if ALL of the following hold:

1. t >= ``min_epochs``;
2. validation Dice has not improved for ``patience`` epochs (necessary, not sufficient);
3. over those last ``patience`` epochs the validation loss shows a statistically
   significant *increasing* trend (one-sided Kendall tau test, p < ``significance``);
4. over the same epochs the training loss shows a significant *decreasing* trend.

(3)+(4) is the textbook signature of overfitting: the model keeps fitting the
training data while generalisation deteriorates. A plateau (flat validation loss and
Dice) does not satisfy (3), so training continues. The best-validation-Dice
checkpoint is always kept and restored before testing regardless of stopping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import kendalltau


def trend_test(values: List[float], alternative: str) -> Tuple[float, float]:
    """Kendall tau between epoch index and values; returns (tau, one-sided p)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 3 or not np.all(np.isfinite(v)) or np.allclose(v, v[0]):
        return 0.0, 1.0
    res = kendalltau(np.arange(len(v)), v, alternative=alternative)
    tau = float(res.statistic) if np.isfinite(res.statistic) else 0.0
    p = float(res.pvalue) if np.isfinite(res.pvalue) else 1.0
    return tau, p


class OverfittingEarlyStopping:
    def __init__(self, cfg: Optional[Dict[str, Any]]):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.patience = int(cfg.get("patience", 20))
        self.min_epochs = int(cfg.get("min_epochs", 0))
        self.alpha = float(cfg.get("significance", 0.05))
        self.reason = ""

    def should_stop(self, history: List[Dict[str, Any]]) -> bool:
        if not self.enabled or len(history) < max(self.min_epochs, self.patience + 1):
            return False
        dices = [h["val_dice"] for h in history]
        best_epoch = int(np.argmax(dices))
        since = len(history) - 1 - best_epoch
        if since < self.patience:
            return False
        window = history[-self.patience:]
        tau_v, p_v = trend_test([h["val_loss"] for h in window], "greater")
        tau_t, p_t = trend_test([h["train_loss"] for h in window], "less")
        if p_v < self.alpha and p_t < self.alpha:
            self.reason = (f"overfitting: val Dice not improved for {since} epochs (best epoch {best_epoch + 1}); "
                           f"val loss increasing (Kendall tau={tau_v:.2f}, p={p_v:.3g}) while train loss decreasing "
                           f"(tau={tau_t:.2f}, p={p_t:.3g}) over the last {self.patience} epochs")
            return True
        return False
