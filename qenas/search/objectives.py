r"""The two QENAS objectives. Both are training-free: no weight is ever updated.

Objective 1 - model complexity
    f1(a) = number of trainable parameters of architecture a            (minimise)

Objective 2 - ZiCo (Li et al., ICLR 2023; official code github.com/SLDGroup/ZiCo,
``ZeroShotProxy/compute_zico.py``)

    The official functions ``getgrad``, ``caculate_zico`` and ``getzico`` are reproduced below without
    any change to their logic:
      * network in train mode; for each of the fixed training batches: zero the gradients, forward,
        compute the *training loss* on the batch, backward, and store the gradient of every
        ``nn.Conv2d`` / ``nn.Linear`` weight;
      * per layer l: over the batches, the standard deviation (np.std) and the mean absolute value of
        every weight gradient; sum mean|g| / std(g) over the weights whose std is non-zero; take log;
      * ZiCo = sum over layers of these logs (a layer whose sum is 0 is skipped).
    Larger is better. The search uses ``ZICO_BATCHES = 2`` training batches (official ``--maxbatch``
    default), fixed once per experiment and identical for every architecture (official: ``trainbatches``
    is built once before the search). The loss is the QENAS training loss with the neutral weighting
    (alpha = beta = 0.5), i.e. the loss used to train the candidates - the same choice as the official
    search (its classification training loss, CrossEntropy) and as ZiCo-BC on Cityscapes (the training
    loss of the segmentation network).

    With two batches every ratio mean|g| / std(g) = (|g1|+|g2|) / |g1-g2| >= 1, so each layer contributes
    log(sum) >= 0 and ZiCo >= 0.

        f2 = 1 / (1 + ZiCo)   in (0, 1]                                 (minimise)

    ZiCo uses the *training* split only (two augmented training batches). Validation and test images are
    never seen by the search.

Both objectives are strictly positive and strictly decreasing in the "goodness" of their criterion; Pareto
dominance is invariant to strictly monotone transforms, so the ranking equals the ranking under
"minimise parameters, maximise ZiCo".

SynFlow, the third objective of QENAS 2.x, was removed in QENAS 3.0 (DESIGN_DECISIONS D10): in evaluation
study v1 it agreed with ZiCo, rewarded larger networks, made the whole final population non-dominated, and
removing it was never significantly worse and significantly better on Kvasir-SEG.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models.network import build_model, count_parameters
from ..training.losses import BCESoftMIoULoss

OBJECTIVE_VERSION = 3
OBJECTIVE_NAMES = ("f1_params", "f2_zico")
ZICO_BATCHES = 2          # official ZiCo --maxbatch default
ZICO_LOSS_ALPHA = 0.5     # neutral BCE / soft-mIoU weighting = the candidate-training loss


class ObjectiveError(RuntimeError):
    """Raised when an objective is NaN, Inf, out of range or otherwise invalid."""


# ============================================================================================== ZiCo
# The three functions below are the official ZiCo implementation (SLDGroup/ZiCo,
# ZeroShotProxy/compute_zico.py). Only device handling differs: the official ``getzico`` calls
# ``network.cuda()`` / ``.cuda()``; here the device is a parameter. Gradients are copied out of the
# parameter (``.copy()``) - on CUDA the official ``.cpu()`` already copies; on CPU ``.numpy()`` would
# otherwise alias the gradient tensor.
def getgrad(model: torch.nn.Module, grad_dict: dict, step_iter=0):
    if step_iter == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                grad_dict[name] = [mod.weight.grad.data.cpu().reshape(-1).numpy().copy()]
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                grad_dict[name].append(mod.weight.grad.data.cpu().reshape(-1).numpy().copy())
    return grad_dict


def caculate_zico(grad_dict):
    allgrad_array = None
    for i, modname in enumerate(grad_dict.keys()):
        grad_dict[modname] = np.array(grad_dict[modname])
    nsr_mean_sum = 0
    nsr_mean_sum_abs = 0
    nsr_mean_avg = 0
    nsr_mean_avg_abs = 0
    for j, modname in enumerate(grad_dict.keys()):
        nsr_std = np.std(grad_dict[modname], axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]
        nsr_mean_abs = np.mean(np.abs(grad_dict[modname]), axis=0)
        tmpsum = np.sum(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx])
        if tmpsum == 0:
            pass
        else:
            nsr_mean_sum_abs += np.log(tmpsum)
            nsr_mean_avg_abs += np.log(np.mean(nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]))
    return nsr_mean_sum_abs


def getzico(network, trainloader, lossfunc, device: torch.device):
    grad_dict = {}
    network.train()

    network.to(device)
    for i, batch in enumerate(trainloader):
        network.zero_grad()
        data, label = batch[0], batch[1]
        data, label = data.to(device), label.to(device)

        logits = network(data)
        loss = lossfunc(logits, label)
        loss.backward()
        grad_dict = getgrad(network, grad_dict, i)

    res = caculate_zico(grad_dict)
    return res


def zico_objective(z: float) -> float:
    """f2 = 1 / (1 + ZiCo). Validates the value (ZiCo >= 0 with two batches, see module docstring)."""
    if z is None or not math.isfinite(z):
        raise ObjectiveError(f"ZiCo is not finite ({z})")
    if z <= -1.0:
        raise ObjectiveError(f"ZiCo = {z} <= -1 cannot be mapped to a positive objective")
    return 1.0 / (1.0 + z)


def zico_batches_digest(batches: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> str:
    """SHA-256 of the exact ZiCo batches (images and masks): identifies the data the score depends on."""
    h = hashlib.sha256()
    for x, y in batches:
        for t in (x, y):
            a = t.detach().cpu().contiguous().numpy()
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
    return h.hexdigest()[:24]


# ============================================================================================== evaluation
def _seeded_model(chromosome: Sequence[int], cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    """Deterministic initialisation (global seed) without touching the global random stream."""
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(int(cfg["seed"]))
        return build_model(chromosome, cfg)


def evaluate_architecture(chromosome: Sequence[int], cfg: Dict[str, Any], device: torch.device,
                          zico_batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, Any]:
    """Training-free evaluation of one architecture: parameters and ZiCo. No weight is updated."""
    t0 = time.time()
    model = _seeded_model(chromosome, cfg, device)
    params = count_parameters(model)
    if params <= 0:
        raise ObjectiveError(f"parameter count {params} is not positive for {list(chromosome)}")
    z = float(getzico(model, zico_batches, BCESoftMIoULoss(alpha=ZICO_LOSS_ALPHA), device))
    f2 = zico_objective(z)
    del model
    return {
        "chromosome": [int(g) for g in chromosome],
        "params": params,
        "zico": z,
        "f1_params": float(params),
        "f2_zico": f2,
        "eval_seconds": round(time.time() - t0, 3),
    }


def objective_signature(cfg: Dict[str, Any], zico_digest: str) -> Dict[str, Any]:
    """Everything besides the architecture that the objective values depend on (cache key context)."""
    return {
        "objective_version": OBJECTIVE_VERSION,
        "input_size": list(cfg["data"]["image_size"]),
        "init_seed": int(cfg["seed"]),
        "zico": f"official getzico, train mode, {ZICO_BATCHES} training batches, BCE/soft-mIoU alpha={ZICO_LOSS_ALPHA}",
        "zico_batches_sha256": zico_digest,
        "torch_version": torch.__version__.split("+")[0],
    }
