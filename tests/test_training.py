import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from qenas.training.early_stopping import OverfittingEarlyStopping
from qenas.training.losses import BCESoftMIoULoss
from qenas.training.metrics import MetricAccumulator, hd95, per_image_metrics
from qenas.training.selection import select_by_learning_capability
from qenas.training.trainer import predict_logits


# ---------------------------------------------------------------------------------------------- loss
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


# ---------------------------------------------------------------------------------------------- metrics
def test_metrics_conventions():
    t = torch.zeros(3, 1, 8, 8)
    t[0, :, :4] = 1          # half foreground
    logits = torch.full_like(t, -5.0)
    logits[0, :, :4] = 5.0    # perfect on 0
    logits[1, :, 0, 0] = 5.0  # false positive on empty gt
    m = per_image_metrics(logits, t)
    assert m["dice"][0] == 1 and m["iou"][0] == 1 and m["miou"][0] == 1
    assert m["dice"][1] == 0 and m["iou"][1] == 0
    assert m["dice"][2] == 1 and m["iou"][2] == 1   # empty gt + empty prediction
    assert m["sensitivity"][0] == 1
    assert torch.isnan(m["sensitivity"][1]) and torch.isnan(m["sensitivity"][2])   # undefined for empty gt


def test_sensitivity_is_recall():
    t = torch.zeros(1, 1, 10, 10)
    t[0, 0, :, :4] = 1                      # 40 foreground pixels
    logits = torch.full_like(t, -5.0)
    logits[0, 0, :, :1] = 5.0               # 10 of them found
    logits[0, 0, :, 8:] = 5.0               # 20 false positives do not change recall
    assert per_image_metrics(logits, t)["sensitivity"][0].item() == pytest.approx(0.25)


def test_hd95_simple_cases():
    a = np.zeros((32, 32), bool)
    a[8:16, 8:16] = True
    assert hd95(a, a) == 0.0
    b = np.roll(a, 3, axis=1)                # same square shifted by 3 px
    assert hd95(a, b) == pytest.approx(3.0)
    empty = np.zeros_like(a)
    assert hd95(empty, empty) == 0.0
    assert hd95(empty, a) == pytest.approx(math.hypot(32, 32))   # missed object -> image diagonal


def test_hd95_ignores_the_worst_5_percent_of_boundary_points():
    gt = np.zeros((64, 64), bool)
    gt[10:50, 10:50] = True                  # 156 boundary pixels
    pred = gt.copy()
    pred[60, 60] = True                      # one far outlier pixel
    assert hd95(pred, gt) == pytest.approx(0.0, abs=1e-9)   # the classical Hausdorff distance would be ~14.1


def test_accumulator_reports_all_metrics():
    acc = MetricAccumulator(boundary=True)
    t = torch.zeros(2, 1, 16, 16)
    t[:, :, 4:8, 4:8] = 1
    logits = (t * 2 - 1) * 5
    acc.update(torch.zeros(2), logits, t)
    m = acc.compute()
    assert m["dice"] == 1 and m["miou"] == 1 and m["sensitivity"] == 1 and m["hd95"] == 0
    assert m["hd95_empty_pred"] == 0


def test_tta_averages_probabilities_of_image_and_mirror():
    class Asym(nn.Module):
        def forward(self, x):
            return x[:, :1] * 0 + torch.linspace(-3, 3, x.shape[-1])

    x = torch.zeros(1, 3, 4, 6)
    m = Asym()
    plain = predict_logits(m, x)
    tta = predict_logits(m, x, tta=True)
    expected = 0.5 * (torch.sigmoid(plain) + torch.sigmoid(plain.flip(-1)))
    assert torch.allclose(torch.sigmoid(tta), expected, atol=1e-5)
    conv = nn.Conv2d(3, 1, 1)
    x = torch.randn(2, 3, 8, 8)
    assert torch.allclose(predict_logits(conv, x, tta=True), predict_logits(conv, x), atol=1e-4)


# ---------------------------------------------------------------------------------------------- early stopping
def _hist(val_dice, val_loss, train_loss):
    return [{"epoch": i + 1, "val_dice": d, "val_loss": vl, "train_loss": tl, "val_iou": d * 0.8, "train_dice": d}
            for i, (d, vl, tl) in enumerate(zip(val_dice, val_loss, train_loss))]


def test_early_stopping_ignores_plateau_but_stops_on_overfitting():
    es = OverfittingEarlyStopping({"patience": 8, "min_epochs": 5, "significance": 0.05})
    plateau = _hist([0.5] + [0.7] + [0.69] * 12, [0.5] * 14, [0.4] * 14)
    assert not es.should_stop(plateau)
    vd = [0.5, 0.7] + [0.65] * 12
    vl = [0.5, 0.3] + list(np.linspace(0.31, 0.6, 12))
    tl = [0.5, 0.3] + list(np.linspace(0.29, 0.05, 12))
    assert es.should_stop(_hist(vd, vl, tl))


# ---------------------------------------------------------------------------------------------- selection
def _cand(cid, vd, vl, tl, params=1000):
    return {"candidate_id": cid, "chromosome": [0] * 8, "blocks": "", "params": params, "synflow_log": 50.0,
            "zico": 400.0, "f2_synflow": 0.02, "f3_zico": 0.0025, "history": _hist(vd, vl, tl)}


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
    diverged = {"candidate_id": 4, "chromosome": [0] * 8, "blocks": "", "params": 1, "synflow_log": 1.0, "zico": 1.0,
                "f2_synflow": 0.5, "f3_zico": 0.5, "diverged": True, "divergence": "nan"}
    sel = select_by_learning_capability([steady, erratic, overfit, diverged], trivial_dice=0.15, significance=0.05,
                                        noise_window=5)
    assert sel["selected_id"] == 1
    by_id = {a["candidate_id"]: a for a in sel["analysis"]}
    assert not by_id[2]["gates"]["G4_learning_trend"]
    assert not by_id[3]["gates"]["G3_no_overfitting"]
    assert by_id[4]["gates"]["G1_finite"] is False
    assert sel["relaxed_gates"] == []
