import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from qenas.config import build_config
from qenas.models.network import build_model
from qenas.search.objectives import OBJECTIVE_NAMES, ObjectiveError, evaluate_architecture, getzico, zico_objective
from qenas.training.losses import BCESoftMIoULoss

DEV = torch.device("cpu")


def small_cfg():
    return build_config("BUSI", overrides={"data.image_size": [64, 64], "model.base_channels": 16})


def batches(n=2, b=4, hw=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(b, 3, hw, hw, generator=g)
        y = (torch.rand(b, 1, hw, hw, generator=g) > 0.7).float()
        out.append((x, y))
    return out


# ---------------------------------------------------------------------------------------------- ZiCo
def zico_independent(model: nn.Module, data, loss_fn) -> float:
    """ZiCo written independently from the paper's definition and the official semantics (Conv2d/Linear weight
    gradients over the batches; per layer log of sum over weights with non-zero std of mean|g| / std(g))."""
    model.train()
    layers = [m for m in model.modules() if type(m) in (nn.Conv2d, nn.Linear)]
    grads = {id(m): [] for m in layers}
    for x, y in data:
        model.zero_grad()
        loss_fn(model(x), y).backward()
        for m in layers:
            grads[id(m)].append(m.weight.grad.detach().flatten().clone().numpy())
    total = 0.0
    for m in layers:
        g = np.stack(grads[id(m)])
        sd = g.std(0)
        nz = sd != 0
        s = (np.abs(g).mean(0)[nz] / sd[nz]).sum()
        if s != 0:
            total += float(np.log(s))
    return total


@pytest.mark.parametrize("ch", [[0, 1, 2, 3, 4, 3, 2, 1], [4] * 8, [3] * 8])
def test_zico_matches_independent_implementation(ch):
    cfg, data, lf = small_cfg(), batches(), BCESoftMIoULoss(0.5)
    torch.manual_seed(3)
    z_official = getzico(build_model(ch, cfg), data, lf, DEV)
    torch.manual_seed(3)
    z_indep = zico_independent(build_model(ch, cfg), data, lf)
    assert z_official == pytest.approx(z_indep, rel=1e-5)


def test_zico_excludes_transposed_convolutions_like_the_official_code():
    # isinstance(ConvTranspose2d, Conv2d) is False -> the official getgrad never collects decoder up-convolutions
    assert not issubclass(nn.ConvTranspose2d, nn.Conv2d)


def test_zico_is_non_negative_with_two_batches():
    cfg = small_cfg()
    for ch in ([0] * 8, [1] * 8, [2, 4, 1, 0, 3, 3, 4, 2]):
        torch.manual_seed(1)
        assert getzico(build_model(ch, cfg), batches(), BCESoftMIoULoss(0.5), DEV) >= 0.0


def test_zico_objective_direction():
    vals = [0.0, 1.0, 10.0, 500.0]
    f2 = [zico_objective(v) for v in vals]
    assert f2[0] == 1.0 and all(0 < f <= 1 for f in f2) and all(a > b for a, b in zip(f2, f2[1:]))
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ObjectiveError):
            zico_objective(bad)


# ---------------------------------------------------------------------------------------------- both objectives
def test_evaluation_is_deterministic_positive_and_leaves_global_rng_untouched():
    cfg, data = small_cfg(), batches()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    a = evaluate_architecture([0, 1, 2, 3, 4, 3, 2, 1], cfg, DEV, data)
    assert torch.equal(torch.rand(3), expected)
    b = evaluate_architecture([0, 1, 2, 3, 4, 3, 2, 1], cfg, DEV, data)
    assert OBJECTIVE_NAMES == ("f1_params", "f2_zico")
    for k in ("params", "zico", "f1_params", "f2_zico"):
        assert a[k] == b[k], k
    assert a["params"] > 0 and a["zico"] > 0 and "synflow_raw" not in a
    assert all(math.isfinite(a[k]) and a[k] > 0 for k in OBJECTIVE_NAMES)
