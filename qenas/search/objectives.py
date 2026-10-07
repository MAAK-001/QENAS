r"""The two QENAS objectives (training-free).

Objective 1 — model complexity
    f1(a) = number of trainable parameters of architecture a           (minimise)

Objective 2 — SynFlow (Tanaka et al., NeurIPS 2020; used as a zero-cost NAS proxy by
Abdelfattah et al., ICLR 2021)

    Official procedure (ganguli-lab/Synaptic-Flow ``Pruners/pruners.py::SynFlow.score``,
    SamsungLabs zero-cost-nas ``measures/synflow.py``), reproduced exactly:
      1. model in eval mode (Synaptic-Flow ``prune_loop`` default ``train_mode=False``),
         so BatchNorm uses its initial running statistics (mean 0, var 1);
      2. ``linearize``: replace every state-dict tensor by its absolute value;
      3. forward a single all-ones input of the data shape [1, C, H, W] in float64
         (zero-cost-nas runs SynFlow in double precision);
      4. R = sum(output); backpropagate;
      5. per-weight score |theta * dR/dtheta| for the weights of prunable layers
         (Conv/Linear weights; biases and normalisation parameters excluded);
      6. ``nonlinearize``: restore the signs.
    The architecture score is S = sum of all per-weight scores (zero-cost-nas ``sum_arr``).

    Normalisation layers: zero-cost-nas' ``synflow`` measure (``bn=False``) evaluates a copy
    of the network *without* normalisation layers; Synaptic-Flow evaluates BatchNorm in eval
    mode, which at initialisation is the identity (up to 1/sqrt(1+eps)). Both therefore
    compute the synaptic flow of the normalisation-free network. QENAS does the same and
    bypasses *every* normalisation layer (BatchNorm and GroupNorm) during SynFlow. This
    extension matters: the DARTS (NAS-Unet) operations use GroupNorm, which normalises with
    per-sample statistics, cannot be linearised, and was measured to suppress S by up to
    ~30 orders of magnitude, i.e. it would penalise DARTS blocks for a numerical artefact.
    For BatchNorm-only networks the result is identical to the official eval-mode procedure
    (verified in tests/test_objectives.py).

    QENAS-specific modification: ``nn.ConvTranspose2d`` weights are counted as prunable
    layers too (the decoder up-samples with transposed convolutions; excluding them
    would make SynFlow blind to the decoder).

    Meaning / direction:  S >= 0 by construction (sum of absolute values). S measures the
    total synaptic flow through the initialised network; *larger is better* (higher
    zero-cost-NAS correlation with trained accuracy; S = 0 means a disconnected network).

    Transformation to a positive minimisation objective:
        raw SynFlow           S  in [0, inf)          larger is better
        log-SynFlow           L = ln(1 + S) in [0, inf)   (S is a product of layer-wise
                              weight magnitudes, i.e. exponential in depth; the log puts
                              it on an additive scale; log1p keeps L >= 0 and finite at S = 0)
        objective             f2 = 1 / (1 + L)  in (0, 1]   smaller is better
    f2 is strictly decreasing in S, strictly positive, bounded, and equals 1 only for a
    network with no synaptic flow. Because NSGA-II's non-dominated sorting is invariant to
    strictly monotone transformations of an objective, the Pareto ranking equals the
    ranking under "maximise S"; only crowding-distance spacing depends on the chosen
    monotone scale. No sign flip and no negative values are involved.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn

from ..models.network import build_model, count_parameters

OBJECTIVE_VERSION = 1
PRUNABLE_TYPES = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)


class ObjectiveError(RuntimeError):
    """Raised when an objective is NaN, Inf, negative or otherwise invalid."""


def bypass_normalization(model: nn.Module) -> int:
    """Replace all normalisation layers by identities (zero-cost-nas ``bn=False``). Returns the count."""
    n = 0
    for name, child in list(model.named_children()):
        if isinstance(child, NORM_TYPES):
            setattr(model, name, nn.Identity())
            n += 1
        else:
            n += bypass_normalization(child)
    return n


@torch.no_grad()
def _linearize(model: nn.Module) -> Dict[str, torch.Tensor]:
    signs = {}
    for name, param in model.state_dict().items():
        signs[name] = torch.sign(param)
        param.abs_()
    return signs


@torch.no_grad()
def _nonlinearize(model: nn.Module, signs: Dict[str, torch.Tensor]) -> None:
    for name, param in model.state_dict().items():
        param.mul_(signs[name])


def synflow_score(model: nn.Module, input_shape: Sequence[int], device: torch.device,
                  dtype: torch.dtype = torch.float64, batch_size: int = 1) -> float:
    """Raw SynFlow score S of an initialised model (official procedure, see module docstring)."""
    model = model.to(device=device, dtype=dtype)
    model.eval()
    signs = _linearize(model)
    model.zero_grad(set_to_none=True)
    x = torch.ones([batch_size] + list(input_shape), dtype=dtype, device=device)
    out = model(x)
    torch.sum(out).backward()
    total = torch.zeros((), dtype=torch.float64, device=device)
    n_layers = 0
    for m in model.modules():
        if isinstance(m, PRUNABLE_TYPES) and m.weight.grad is not None:
            total += (m.weight.detach() * m.weight.grad).abs().sum().to(torch.float64)
            n_layers += 1
    _nonlinearize(model, signs)
    model.zero_grad(set_to_none=True)
    if n_layers == 0:
        raise ObjectiveError("SynFlow: model has no prunable layers with gradients")
    return float(total.item())


def synflow_objective(raw: float) -> Tuple[float, float]:
    """Map raw SynFlow S to (L = ln(1+S), f2 = 1/(1+L)). Validates the result."""
    if raw is None or not math.isfinite(raw):
        raise ObjectiveError(f"SynFlow is not finite (S={raw}). The float64 synaptic flow overflowed; "
                             f"reduce search.synflow.input_size.")
    if raw < 0:
        raise ObjectiveError(f"SynFlow is negative (S={raw}); impossible for a sum of absolute values")
    log_s = math.log1p(raw)
    f2 = 1.0 / (1.0 + log_s)
    if not (math.isfinite(f2) and 0.0 < f2 <= 1.0):
        raise ObjectiveError(f"Invalid SynFlow objective f2={f2} from S={raw}")
    return log_s, f2


def evaluate_architecture(chromosome: Sequence[int], cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Training-free evaluation: build -> initialise -> count parameters -> SynFlow. No data, no optimiser."""
    sf = cfg["search"]["synflow"]
    init_seed = int(sf["init_seed"] if sf["init_seed"] is not None else cfg["seed"])
    input_hw = sf["input_size"] or cfg["data"]["image_size"]
    input_shape = [int(cfg["data"]["in_channels"]), int(input_hw[0]), int(input_hw[1])]
    dtype = {"float64": torch.float64, "float32": torch.float32}[sf["dtype"]]
    t0 = time.time()
    # Identical, deterministic initialisation for every architecture; the global RNG stream is untouched.
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(init_seed)
        model = build_model(chromosome, cfg)
    params = count_parameters(model)   # counted on the real (normalised) architecture
    if params <= 0:
        raise ObjectiveError(f"parameter count {params} is not positive for {list(chromosome)}")
    bypass_normalization(model)        # throw-away copy used only for SynFlow
    raw = synflow_score(model, input_shape, device, dtype=dtype, batch_size=int(sf["batch_size"]))
    log_s, f2 = synflow_objective(raw)
    del model
    return {
        "chromosome": [int(g) for g in chromosome],
        "params": params,
        "synflow_raw": raw,
        "synflow_log": log_s,
        "f1_params": float(params),
        "f2_synflow": f2,
        "eval_seconds": round(time.time() - t0, 3),
    }


def objective_signature(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sf = cfg["search"]["synflow"]
    return {
        "objective_version": OBJECTIVE_VERSION,
        "synflow_normalization": "bypassed (zero-cost-nas bn=False semantics, BN+GN)",
        "synflow_dtype": sf["dtype"],
        "synflow_input_size": list(sf["input_size"] or cfg["data"]["image_size"]),
        "synflow_init_seed": int(sf["init_seed"] if sf["init_seed"] is not None else cfg["seed"]),
        "synflow_batch_size": int(sf["batch_size"]),
        "torch_version": torch.__version__.split("+")[0],
    }
