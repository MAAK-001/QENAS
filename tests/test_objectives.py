import math

import pytest
import torch

from qenas.config import build_config
from qenas.models.network import build_model
from qenas.search.objectives import (ObjectiveError, bypass_normalization, evaluate_architecture, synflow_objective,
                                     synflow_score)

DEV = torch.device("cpu")


def small_cfg():
    return build_config("BUSI", overrides={"data.image_size": [64, 64], "model.base_channels": 16})


def test_objective_direction_and_positivity():
    # larger SynFlow -> smaller (better) objective; always in (0, 1]
    vals = [0.0, 1.0, 1e3, 1e20, 1e60]
    f2 = [synflow_objective(v)[1] for v in vals]
    assert all(0.0 < f <= 1.0 for f in f2)
    assert f2[0] == 1.0
    assert all(a > b for a, b in zip(f2, f2[1:]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_invalid_synflow_rejected(bad):
    with pytest.raises(ObjectiveError):
        synflow_objective(bad)


def test_synflow_deterministic_and_positive():
    cfg = small_cfg()
    a = evaluate_architecture([0, 1, 2, 3, 4, 3, 2, 1], cfg, DEV)
    b = evaluate_architecture([0, 1, 2, 3, 4, 3, 2, 1], cfg, DEV)
    assert a["synflow_raw"] == b["synflow_raw"] > 0
    assert a["params"] == b["params"] > 0
    assert math.isfinite(a["synflow_log"]) and a["synflow_log"] > 0


def test_synflow_does_not_touch_global_rng():
    cfg = small_cfg()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    evaluate_architecture([4, 4, 4, 4, 4, 4, 4, 4], cfg, DEV)
    assert torch.equal(torch.rand(3), expected)


def test_normalisation_free_equals_official_eval_mode_for_batchnorm_networks():
    """For BN-only networks, bypassing BN equals the official eval-mode procedure (BN = identity at init)."""
    cfg = small_cfg()
    ch = [0, 1, 2, 3, 0, 1, 2, 3]   # no DARTS cell -> BatchNorm only
    torch.manual_seed(7)
    m1 = build_model(ch, cfg)
    s_official = synflow_score(m1, [3, 64, 64], DEV)
    torch.manual_seed(7)
    m2 = build_model(ch, cfg)
    bypass_normalization(m2)
    s_free = synflow_score(m2, [3, 64, 64], DEV)
    assert s_free == pytest.approx(s_official, rel=1e-3)


def test_synflow_restores_weight_signs():
    cfg = small_cfg()
    torch.manual_seed(0)
    m = build_model([0] * 8, cfg).double()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    synflow_score(m, [3, 64, 64], DEV)
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k
