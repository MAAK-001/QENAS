import numpy as np
import pytest
import torch

from qenas.training.early_stopping import OverfittingEarlyStopping
from qenas.training.losses import BCESoftMIoULoss
from qenas.training.metrics import per_image_metrics
from qenas.training.selection import select_by_learning_capability


def test_loss_perfect_prediction_is_small_and_wrong_is_large():
    t = torch.zeros(2, 1, 16, 16)
    t[:, :, 4:12, 4:12] = 1
    good = (t * 2 - 1) * 20.0
    bad = -good
    for a in (0.0, 0.5, 1.0):
        lf = BCESoftMIoULoss(alpha=a)
        assert lf(good, t).item() < 1e-3
        assert lf(bad, t).item() > 0.4


def test_soft_miou_empty_mask_correct_prediction_near_zero():
    lf = BCESoftMIoULoss(alpha=0.0)
    t = torch.zeros(1, 1, 32, 32)
    assert lf(torch.full_like(t, -20.0), t).item() < 1e-3


def test_metrics_conventions():
    t = torch.zeros(3, 1, 8, 8)
    t[0, :, :4] = 1          # half foreground
    logits = torch.full_like(t, -5.0)
    logits[0, :, :4] = 5.0    # perfect on 0
    logits[1, :, 0, 0] = 5.0  # false positive on empty gt
    m = per_image_metrics(logits, t)
    assert m["dice"][0] == 1 and m["iou"][0] == 1
    assert m["dice"][1] == 0 and m["iou"][1] == 0
    assert m["dice"][2] == 1 and m["iou"][2] == 1   # empty gt + empty prediction
    assert m["miou"][0] == 1


def _hist(val_dice, val_loss, train_loss):
    return [{"epoch": i + 1, "val_dice": d, "val_loss": vl, "train_loss": tl, "val_iou": d * 0.8, "train_dice": d}
            for i, (d, vl, tl) in enumerate(zip(val_dice, val_loss, train_loss))]


def test_early_stopping_ignores_plateau_but_stops_on_overfitting():
    es = OverfittingEarlyStopping({"enabled": True, "patience": 8, "min_epochs": 5, "significance": 0.05})
    plateau = _hist([0.5] + [0.7] + [0.69] * 12, [0.5] * 14, [0.4] * 14)
    assert not es.should_stop(plateau)
    vd = [0.5, 0.7] + [0.65] * 12
    vl = [0.5, 0.3] + list(np.linspace(0.31, 0.6, 12))
    tl = [0.5, 0.3] + list(np.linspace(0.29, 0.05, 12))
    assert es.should_stop(_hist(vd, vl, tl))


def _cand(cid, vd, vl, tl, params=1000):
    return {"candidate_id": cid, "chromosome": [0] * 8, "blocks": "", "params": params, "synflow_log": 50.0,
            "f2_synflow": 0.02, "history": _hist(vd, vl, tl)}


def test_selection_rejects_pathological_large_improvement():
    E = 20
    steady = _cand(1, list(np.linspace(0.30, 0.70, E)), list(np.linspace(0.8, 0.4, E)), list(np.linspace(0.8, 0.3, E)))
    # erratic curve with a lucky final epoch: huge ΔDice but no significant learning trend
    erratic_vd = [0.05, 0.6, 0.1, 0.55, 0.08, 0.5, 0.12, 0.6, 0.05, 0.4, 0.1, 0.5, 0.06, 0.45, 0.1, 0.5, 0.08, 0.4,
                  0.07, 0.90]
    erratic = _cand(2, erratic_vd, list(np.linspace(0.8, 0.5, E)), list(np.linspace(0.8, 0.3, E)))
    # overfitting: val loss rises in the second half while train loss falls
    of_vl = list(np.linspace(0.8, 0.4, 10)) + list(np.linspace(0.42, 0.9, 10))
    overfit = _cand(3, list(np.linspace(0.1, 0.85, E)), of_vl, list(np.linspace(0.8, 0.05, E)))
    diverged = {"candidate_id": 4, "chromosome": [0] * 8, "blocks": "", "params": 1, "synflow_log": 1.0,
                "f2_synflow": 0.5, "diverged": True, "divergence": "nan"}
    sel = select_by_learning_capability([steady, erratic, overfit, diverged], trivial_dice=0.15, significance=0.05,
                                        noise_window=5)
    assert sel["selected_id"] == 1
    by_id = {a["candidate_id"]: a for a in sel["analysis"]}
    assert not by_id[2]["gates"]["G4_learning_trend"]
    assert not by_id[3]["gates"]["G3_no_overfitting"]
    assert by_id[4]["gates"]["G1_finite"] is False
    assert sel["relaxed_gates"] == []
