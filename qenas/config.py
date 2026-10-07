"""Configuration system.

Resolution order (later wins):
    1. ``DEFAULT_CONFIG`` (this file)
    2. ``DATASET_OVERRIDES[dataset]`` (documented, dataset-specific *preprocessing* facts only:
       resolution and physically meaningful augmentation; never hyper-parameter tuning)
    3. an optional YAML file passed with ``--config``
    4. explicit command-line options

The fully resolved configuration is saved as ``config.json`` inside every
experiment directory. A *fingerprint* of all scientifically relevant keys is used
to make sure a run is only ever resumed with an identical configuration.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

QENAS_ROOT = Path(__file__).resolve().parents[1]          # .../GGNAS/QENAS
GGNAS_ROOT = QENAS_ROOT.parent                             # .../GGNAS

SUPPORTED_DATASETS = ("BUSI", "CVC", "IDRID")

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "deterministic": True,
    "smoke_test": False,
    "data": {
        "dataset": "BUSI",
        "root": str(GGNAS_ROOT / "Datasets"),
        "image_size": [256, 256],          # [H, W]; must be divisible by 16 (4 down-samplings)
        "in_channels": 3,                  # grayscale images are replicated to 3 channels
        "test_fraction": 0.2,              # used only when the dataset has no official test split
        "val_fraction": 0.2,               # fraction of the development (non-test) data used for validation
        "mask_threshold": 127,             # 8-bit masks: foreground = value > threshold
        "busi_include_normal": False,      # Mixed-GGNAS uses the 647 benign+malignant images
        "busi_drop_conflicting_duplicates": True,
        "idrid_target": "OD",              # OD | MA | HE | EX | SE | LESIONS (union of MA, HE, EX, SE)
        "normalization": "train_stats",    # per-channel mean/std estimated on the training split only
        "cache_preprocessed": True,        # cache resized arrays under QENAS/cache/preprocessed
        "num_workers": 0,
        "max_samples": None,               # {"train": n, "val": n, "test": n} — smoke tests only
    },
    "augmentation": {
        "enabled": True,
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
        "scale": 3,                        # fixed convolution scale (3 | 5 | 7) of manual blocks — see DESIGN_DECISIONS
        "darts_genotype": "auto",          # auto -> dataset-matched official Mixed-GGNAS genotype
        "num_classes": 1,                  # binary segmentation, single logit
    },
    "search": {
        "population_size": 10,
        "generations": 10,                 # evolutionary generations after the initial population (gen 0)
        "crossover_prob": 0.9,
        "mutation_prob": 1.0,              # probability an offspring is passed to the mutation operator
        "mutation_prob_var": 0.125,        # per-gene random-resetting probability (= 1 / chromosome length)
        "eliminate_duplicates": True,
        "plot_every": 2,                   # save an intermediate Pareto plot every k generations
        "synflow": {
            "dtype": "float64",            # as in the zero-cost-NAS reference implementation
            "input_size": None,            # None -> data.image_size
            "init_seed": None,             # None -> global seed
            "batch_size": 1,               # official SynFlow uses a single all-ones input
        },
        "cache_dir": str(QENAS_ROOT / "cache" / "zero_cost"),
    },
    "pareto": {
        "max_candidates": 8,               # evaluate all Pareto points if the front is smaller
    },
    "candidate_training": {
        "epochs": 20,
        "batch_size": 8,
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "weight_decay": 5.0e-5,
        "momentum": 0.9,
        "scheduler": "none",               # constant LR: learning capability is measured without annealing
        "warmup_epochs": 0,
        "poly_power": 0.9,
        "grad_clip": None,
        "amp": "auto",
        "loss_alpha": 0.5,                 # neutral BCE / soft-mIoU weighting before alpha is optimised
    },
    "selection": {
        "significance": 0.05,              # one-sided Kendall trend tests
        "noise_window": 5,                 # epochs used to estimate validation-Dice noise
    },
    "loss_weights": {
        "enabled": True,
        "alphas": [0.2, 0.4, 0.5, 0.6, 0.8],
        "epochs": 20,
        "criterion_window": 5,             # mean val Dice over the last k epochs
        "reuse_candidate_run": True,       # alpha == candidate alpha reuses the identical candidate run
    },
    "final_training": {
        "epochs": 100,
        "batch_size": 8,
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "weight_decay": 5.0e-5,
        "momentum": 0.9,
        "scheduler": "poly",               # official Mixed-GGNAS schedule: linear warm-up + poly(0.9)
        "warmup_epochs": 1,
        "poly_power": 0.9,
        "grad_clip": None,
        "amp": "auto",
        "early_stopping": {
            "enabled": True,
            "patience": 20,                # epochs without val-Dice improvement (necessary condition)
            "min_epochs": 30,
            "window": 5,                   # trend window for the overfitting test
            "significance": 0.05,
        },
    },
}

# Dataset-specific *data facts* (documented in DESIGN_DECISIONS.md). No model or optimisation tuning.
DATASET_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "BUSI": {
        "data": {"image_size": [256, 256]},
        # Ultrasound: depth axis has a physical direction (probe at top) -> no vertical flip.
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
}

# Keys that do not influence any scientific result (excluded from the fingerprint).
NON_SCIENTIFIC_KEYS = {
    ("device",),
    ("data", "root"),
    ("data", "num_workers"),
    ("data", "cache_preprocessed"),
    ("search", "cache_dir"),
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


def get_by_path(cfg: Dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        cur = cur[k]
    return cur


def build_config(dataset: str, yaml_path: Optional[str] = None,
                 overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dataset = dataset.upper()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Choose from {SUPPORTED_DATASETS}.")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    deep_update(cfg, DATASET_OVERRIDES.get(dataset, {}))
    cfg["data"]["dataset"] = dataset
    if yaml_path:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            ycfg = yaml.safe_load(fh) or {}
        _check_known_keys(cfg, ycfg, prefix="")
        deep_update(cfg, ycfg)
        cfg["data"]["dataset"] = dataset  # the CLI dataset always wins
    for dotted, value in (overrides or {}).items():
        if value is not None:
            set_by_path(cfg, dotted, value)
    validate_config(cfg)
    return cfg


def _check_known_keys(base: Dict[str, Any], upd: Dict[str, Any], prefix: str) -> None:
    for k, v in upd.items():
        path = f"{prefix}{k}"
        if k not in base:
            raise KeyError(f"Unknown configuration key '{path}' in YAML file")
        if isinstance(v, dict) and isinstance(base[k], dict):
            _check_known_keys(base[k], v, prefix=path + ".")


def validate_config(cfg: Dict[str, Any]) -> None:
    h, w = cfg["data"]["image_size"]
    if h % 16 or w % 16:
        raise ValueError(f"data.image_size {h}x{w} must be divisible by 16 (four 2x down-samplings).")
    if cfg["model"]["scale"] not in (3, 5, 7):
        raise ValueError("model.scale must be one of 3, 5, 7 (Mixed-GGNAS Table 1).")
    if cfg["model"]["base_channels"] % 16:
        raise ValueError("model.base_channels must be a multiple of 16 (official blocks use groups=16).")
    if not 0.0 < cfg["data"]["val_fraction"] < 1.0:
        raise ValueError("data.val_fraction must be in (0, 1)")
    if not 0.0 < cfg["data"]["test_fraction"] < 1.0:
        raise ValueError("data.test_fraction must be in (0, 1)")
    s = cfg["search"]
    if s["population_size"] < 2 or s["generations"] < 0:
        raise ValueError("search.population_size must be >= 2 and search.generations >= 0")
    for a in cfg["loss_weights"]["alphas"]:
        if not 0.0 <= float(a) <= 1.0:
            raise ValueError("loss_weights.alphas must lie in [0, 1] (beta = 1 - alpha)")
    if not 0.0 <= cfg["candidate_training"]["loss_alpha"] <= 1.0:
        raise ValueError("candidate_training.loss_alpha must lie in [0, 1]")
    if cfg["candidate_training"]["epochs"] < 2:
        raise ValueError("candidate_training.epochs must be >= 2 to measure an improvement")
    if cfg["data"]["idrid_target"].upper() not in ("OD", "MA", "HE", "EX", "SE", "LESIONS"):
        raise ValueError("data.idrid_target must be one of OD, MA, HE, EX, SE, LESIONS")


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
    keys = set(a) | set(b)
    for k in sorted(keys):
        p = f"{prefix}{k}"
        if tuple(p.split(".")) in NON_SCIENTIFIC_KEYS:
            continue
        va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
        if isinstance(va, dict) and isinstance(vb, dict):
            out.update(diff_configs(va, vb, prefix=p + "."))
        elif json.dumps(va, default=str) != json.dumps(vb, default=str):
            out[p] = (va, vb)
    return out
