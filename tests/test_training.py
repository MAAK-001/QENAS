import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from qenas.training.early_stopping import OverfittingEarlyStopping
from qenas.training.losses import BCESoftMIoULoss
from qenas.training.metrics import MetricAccumulator, hd95, per_image_metrics
from torch.utils.data import DataLoader

from qenas.datasets.dataset import ArraySegDataset, EpochShuffleSampler
from qenas.training.selection import criterion, promote, rung_sizes, select_by_learning_capability
from qenas.training.trainer import Trainer, predict_logits


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


def test_prediction_has_no_test_time_augmentation():
    conv = nn.Conv2d(3, 1, 1)
    x = torch.randn(2, 3, 8, 8)
    assert torch.equal(predict_logits(conv, x), conv(x).float())


# ---------------------------------------------------------------------------------------------- resumable runs
class _Data:
    """Minimal train/validation data for the Trainer (two 16x16 images per class of blob)."""

    def __init__(self):
        rng = np.random.default_rng(0)
        self.imgs = rng.integers(0, 255, (10, 16, 16, 3), dtype=np.uint8)
        self.masks = (rng.random((10, 16, 16)) > 0.6).astype(np.uint8)

    def train_loader(self, bs, seed):
        ds = ArraySegDataset(self.imgs, self.masks, [0.5] * 3, [0.25] * 3,
                             augment={"hflip": 0.5, "vflip": 0.0, "affine_p": 0.0, "rotate_deg": 0.0,
                                      "scale": [1.0, 1.0], "translate": 0.0, "brightness": 0.1, "contrast": 0.1},
                             seed=seed)
        return DataLoader(ds, batch_size=bs, sampler=EpochShuffleSampler(len(ds), seed))

    def val_loader(self, bs):
        return DataLoader(ArraySegDataset(self.imgs[:4], self.masks[:4], [0.5] * 3, [0.25] * 3), batch_size=bs)


def _trainer(ckpt, epochs=4):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))
    tcfg = {"epochs": epochs, "batch_size": 4, "lr": 1e-2, "weight_decay": 0.0, "scheduler": "none"}
    return Trainer(model=model, data=_Data(), tcfg=tcfg, alpha=0.5, device=torch.device("cpu"), ckpt_dir=ckpt,
                   seed=3, fingerprint="fp", tag="t")


def test_run_continued_rung_by_rung_equals_an_uninterrupted_run(tmp_path):
    """Successive halving: 2 epochs, stop, then continue to 4 = 4 epochs in one go (constant LR, same data order)."""
    straight = _trainer(tmp_path / "a").fit()["history"]
    part = _trainer(tmp_path / "b").fit(until=2)
    assert len(part["history"]) == 2 and not part["completed"]
    cont = _trainer(tmp_path / "b").fit()
    assert cont["completed"] and len(cont["history"]) == 4
    for h1, h2 in zip(straight, cont["history"]):
        for k in ("train_loss", "val_loss", "val_dice"):
            assert h1[k] == pytest.approx(h2[k], rel=1e-6, abs=1e-9), k
    assert _trainer(tmp_path / "b").fit(until=3)["history"] == cont["history"]   # never trains backwards


def test_rung_sizes_and_promotion():
    assert rung_sizes(16, 3, 0.5) == [16, 8, 4]
    assert rung_sizes(5, 3, 0.5) == [5, 3, 2]
    scored = [{"candidate_id": i, "score": s, "params": p} for i, s, p in
              ((1, 0.5, 10), (2, 0.7, 10), (3, 0.7, 5), (4, 0.1, 1))]
    assert promote(scored, 2) == [3, 2]          # equal scores: fewer parameters first
    assert criterion([{"val_dice": v} for v in (0.1, 0.2, 0.6, 0.8)], 2) == pytest.approx(0.7)


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
    return {"candidate_id": cid, "chromosome": [0] * 8, "blocks": "", "params": params, "zico": 400.0,
            "f2_zico": 0.0025, "rung_scores": {}, "history": _hist(vd, vl, tl)}


def test_selection_by_end_of_run_dice_with_gates():
    E = 20
    steady = _cand(1, list(np.linspace(0.30, 0.70, E)), list(np.linspace(0.8, 0.4, E)), list(np.linspace(0.8, 0.3, E)))
    # a poor start and a steep rise: the largest ΔDice but a lower end-of-run Dice -> not selected any more
    late = _cand(2, list(np.linspace(0.01, 0.66, E)), list(np.linspace(0.9, 0.4, E)), list(np.linspace(0.8, 0.3, E)))
    # erratic curve with a lucky final epoch: no significant learning trend (G4)
    erratic_vd = [0.05, 0.6, 0.1, 0.55, 0.08, 0.5, 0.12, 0.6, 0.05, 0.4, 0.1, 0.5, 0.06, 0.45, 0.1, 0.5, 0.08, 0.95,
                  0.97, 0.99]
    erratic = _cand(3, erratic_vd, list(np.linspace(0.8, 0.5, E)), list(np.linspace(0.8, 0.3, E)))
    # overfitting: val loss rises in the second half while train loss falls (G3)
    of_vl = list(np.linspace(0.8, 0.4, 10)) + list(np.linspace(0.42, 0.9, 10))
    overfit = _cand(4, list(np.linspace(0.1, 0.85, E)), of_vl, list(np.linspace(0.8, 0.05, E)))
    eliminated = _cand(5, [0.9] * 5, [0.2] * 5, [0.2] * 5)          # stopped at the first rung: not a finalist
    diverged = {"candidate_id": 6, "chromosome": [0] * 8, "blocks": "", "params": 1, "zico": 1.0, "f2_zico": 0.5,
                "diverged": True, "divergence": "nan"}
    sel = select_by_learning_capability([steady, late, erratic, overfit, eliminated, diverged], trivial_dice=0.15,
                                        significance=0.05, window=5, full_epochs=E)
    assert sel["selected_id"] == 1
    by_id = {a["candidate_id"]: a for a in sel["analysis"]}
    assert by_id[2]["delta_dice"] > by_id[1]["delta_dice"] and by_id[2]["criterion"] < by_id[1]["criterion"]
    assert not by_id[3]["gates"]["G4_learning_trend"] and by_id[3]["criterion"] > by_id[1]["criterion"]
    assert not by_id[4]["gates"]["G3_no_overfitting"]
    assert not by_id[5]["finalist"] and not by_id[5]["eligible"]
    assert by_id[6]["gates"]["G1_finite"] is False
    assert sel["relaxed_gates"] == []
