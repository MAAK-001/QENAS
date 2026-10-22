"""The QENAS protocol: one fixed configuration, one pipeline.

Resolution order (later wins):
    1. ``DEFAULT_CONFIG`` (this file) - the QENAS method;
    2. ``DATASET_OVERRIDES[dataset]`` - documented dataset *facts* only (resolution, physically
       meaningful augmentation, the batch size that fits the IDRiD resolution); never tuning;
    3. run settings from the command line: data root, device, data-loader workers, seed.

The QENAS method has no alternative modes, options or protocol variants. The comparison conditions of the
evaluation study (baselines, ablations, random search, the CVC leakage study) are defined in ``qenas/methods.py``
and set through the key ``method`` by ``study.py`` only. The fully resolved configuration
is saved as ``config.json`` inside every experiment. A *fingerprint* of all scientific keys guarantees that a run is only
ever resumed with an identical configuration.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

QENAS_ROOT = Path(__file__).resolve().parents[1]          # .../GGNAS/QENAS
GGNAS_ROOT = QENAS_ROOT.parent                             # .../GGNAS

SUPPORTED_DATASETS = ("BUSI", "CVC", "IDRID", "KVASIR")

DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "qenas",                     # qenas/methods.py; main.py always runs "qenas", study.py also the rest
    "seed": 42,                            # search, initialisation, data order and augmentation (NOT the split)
    "device": "auto",
    "paths": {
        "results": str(QENAS_ROOT / "results"),           # results/<DATASET>/<method>/seed_<seed>/
        "cache": str(QENAS_ROOT / "cache"),               # zero-cost evaluations + preprocessed arrays
    },
    "data": {
        "dataset": "BUSI",
        "root": str(GGNAS_ROOT / "Datasets"),
        "image_size": [256, 256],          # [H, W]; divisible by 16 (four 2x down-samplings)
        "in_channels": 3,                  # grayscale images are replicated to 3 channels
        "test_fraction": 0.2,              # used only when the dataset has no official test split
        "val_fraction": 0.2,               # fraction of the development (non-test) data used for validation
        "split_seed": 42,                  # fixed: every method and every seed uses the identical split
        "mask_threshold": 127,             # 8-bit masks: foreground = value > threshold
        "num_workers": 0,
    },
    "augmentation": {                      # training split only
        "hflip": 0.5,
        "vflip": 0.0,
        "affine_p": 0.5,
        "rotate_deg": 15.0,
        "scale": [0.9, 1.1],
        "translate": 0.05,
        "brightness": 0.1,
        "contrast": 0.1,
    },
    "model": {
        "base_channels": 32,               # official Mixed-GGNAS default (base_c=32)
        "scale": 3,                        # fixed convolution scale of the manual blocks (DESIGN_DECISIONS D4)
        "num_classes": 1,                  # binary segmentation, one logit per pixel
    },
    "search": {                            # training-free NSGA-II (objectives: params, ZiCo)
        "population_size": 20,
        "generations": 10,                 # evolutionary generations after the initial population (gen 0)
        "crossover_prob": 0.9,
        "mutation_prob_var": 0.125,        # per-gene random-resetting probability (= 1 / chromosome length)
        "plot_every": 5,                   # intermediate Pareto figure every k generations
    },
    "pareto": {
        "max_candidates": 16,              # candidates of the final population trained by successive halving
    },
    "candidate_training": {                # learning-capability runs (successive halving)
        "epochs": 20,                      # length of a complete candidate run (the last rung)
        "rungs": [5, 10, 20],              # epochs reached at each rung: 16 x 5 -> 8 x 10 -> 4 x 20 epochs
        "keep_fraction": 0.5,              # share of a rung promoted to the next one (halving)
        "batch_size": 8,
        "lr": 1.0e-3,
        "weight_decay": 5.0e-5,
        "scheduler": "none",               # constant LR: learning capability is measured without annealing
        "loss_alpha": 0.5,                 # neutral BCE / soft-mIoU weighting before alpha is optimised
    },
    "selection": {
        "significance": 0.05,              # one-sided Kendall trend tests (gates G3, G4)
        "criterion_window": 5,             # criterion: mean validation Dice over the last k epochs of a rung
    },
    "loss_weights": {                      # runs use the candidate protocol (same epochs, seed, data order)
        "alphas": [0.2, 0.4, 0.5, 0.6, 0.8],
        "criterion_window": 5,             # mean validation Dice over the last k epochs
    },
    "final_training": {
        "epochs": 120,
        "batch_size": 8,
        "lr": 1.0e-3,
        "weight_decay": 5.0e-5,
        "scheduler": "poly",               # official Mixed-GGNAS schedule: linear warm-up + poly(0.9)
        "warmup_epochs": 1,
        "poly_power": 0.9,
        "early_stopping": {                # only on overfitting, never on stagnation
            "patience": 20,
            "min_epochs": 30,
            "significance": 0.05,
        },
    },
}

# Dataset-specific *data facts* (DESIGN_DECISIONS.md D33-D38). No model or optimisation tuning.
DATASET_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "BUSI": {
        "data": {"image_size": [256, 256]},
        # Ultrasound: the depth axis has a physical direction (probe at the top) -> no vertical flip.
        "augmentation": {"vflip": 0.0},
    },
    "CVC": {
        "data": {"image_size": [256, 256]},
        # Colonoscopy frames have no canonical orientation.
        "augmentation": {"vflip": 0.5},
    },
    "IDRID": {
        "data": {"image_size": [320, 512]},   # paper: 512 x 320 (W x H), aspect ~ 4288 x 2848
        "augmentation": {"vflip": 0.0},
        "candidate_training": {"batch_size": 4},
        "final_training": {"batch_size": 4},
    },
    "KVASIR": {
        "data": {"image_size": [256, 256]},   # as CVC-ClinicDB (same modality); native 332x487 ... 1920x1072
        # Colonoscopy frames have no canonical orientation.
        "augmentation": {"vflip": 0.5},
    },
}

# Keys that do not influence any scientific result (excluded from the fingerprint).
NON_SCIENTIFIC_KEYS = {
    ("device",),
    ("paths",),
    ("data", "root"),
    ("data", "num_workers"),
    ("search", "plot_every"),
}


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            raise KeyError(f"Unknown configuration section '{k}' in '{dotted}'")
        cur = cur[k]
    if keys[-1] not in cur:
        raise KeyError(f"Unknown configuration key '{dotted}'")
    cur[keys[-1]] = value


def build_config(dataset: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The configuration for ``dataset``. ``overrides`` ({dotted.key: value}) carries the run settings of the
    command line (root, device, workers, seed) and, for study.py, the study condition (``method``,
    the unit tests also use it to shrink the protocol for fast end-to-end checks."""
    dataset = dataset.upper()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from {SUPPORTED_DATASETS}.")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    deep_update(cfg, DATASET_OVERRIDES[dataset])
    cfg["data"]["dataset"] = dataset
    for dotted, value in (overrides or {}).items():
        if value is not None:
            set_by_path(cfg, dotted, value)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    h, w = cfg["data"]["image_size"]
    if h % 16 or w % 16:
        raise ValueError(f"data.image_size {h}x{w} must be divisible by 16 (four 2x down-samplings).")
    if cfg["model"]["scale"] not in (3, 5, 7):
        raise ValueError("model.scale must be one of 3, 5, 7 (Mixed-GGNAS Table 1).")
    if cfg["model"]["base_channels"] % 16:
        raise ValueError("model.base_channels must be a multiple of 16 (official blocks use groups=16).")
    from .methods import method_spec
    method_spec(cfg["method"])
    if not 0.0 < cfg["data"]["val_fraction"] < 1.0 or not 0.0 < cfg["data"]["test_fraction"] < 1.0:
        raise ValueError("data.val_fraction and data.test_fraction must lie in (0, 1)")
    s = cfg["search"]
    if s["population_size"] < 2 or s["generations"] < 0:
        raise ValueError("search.population_size must be >= 2 and search.generations >= 0")
    for a in cfg["loss_weights"]["alphas"]:
        if not 0.0 <= float(a) <= 1.0:
            raise ValueError("loss_weights.alphas must lie in [0, 1] (beta = 1 - alpha)")
    if float(cfg["candidate_training"]["loss_alpha"]) not in [float(a) for a in cfg["loss_weights"]["alphas"]]:
        raise ValueError("candidate_training.loss_alpha must be one of loss_weights.alphas (its run is reused)")
    ct = cfg["candidate_training"]
    rungs = [int(r) for r in ct["rungs"]]
    if ct["epochs"] < 2:
        raise ValueError("candidate_training.epochs must be >= 2 to measure an improvement")
    if not rungs or rungs != sorted(set(rungs)) or rungs[-1] != int(ct["epochs"]) or rungs[0] < 1:
        raise ValueError("candidate_training.rungs must be strictly increasing and end at candidate_training.epochs")
    if not 0.0 < float(ct["keep_fraction"]) < 1.0:
        raise ValueError("candidate_training.keep_fraction must lie in (0, 1)")


def _strip(cfg: Dict[str, Any], keys: Iterable[tuple]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for path in keys:
        cur = out
        for k in path[:-1]:
            cur = cur.get(k, {})
        if isinstance(cur, dict):
            cur.pop(path[-1], None)
    return out


def config_fingerprint(cfg: Dict[str, Any]) -> str:
    sci = _strip(cfg, NON_SCIENTIFIC_KEYS)
    text = json.dumps(sci, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def diff_configs(a: Dict[str, Any], b: Dict[str, Any], prefix: str = "") -> Dict[str, tuple]:
    """Return {dotted_key: (a_value, b_value)} for all differing scientific keys."""
    out: Dict[str, tuple] = {}
    for k in sorted(set(a) | set(b)):
        p = f"{prefix}{k}"
        if tuple(p.split(".")) in NON_SCIENTIFIC_KEYS:
            continue
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(diff_configs(va, vb, prefix=p + "."))
        elif json.dumps(va, default=str) != json.dumps(vb, default=str):
            out[p] = (va, vb)
    return out
