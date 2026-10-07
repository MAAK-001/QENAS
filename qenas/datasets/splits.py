"""Deterministic, leakage-free train/validation/test splitting.

* Official test split (IDRID) is used as-is; validation is carved from the official
  training set.
* Otherwise a test split is drawn first, then validation from the remaining
  development data.
* Splitting happens at the level of *leakage groups* (CVC: colonoscopy sequence;
  BUSI: image or exact-duplicate set; IDRID: image) and is stratified by class
  where a class label exists (BUSI benign/malignant).
* Everything is driven by a seeded ``numpy.random.Generator`` -> identical splits
  for identical seeds.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..utils.seed import derive_seed
from .discovery import DatasetError, DiscoveryResult, Sample


def _group_select(samples: Sequence[Sample], fraction: float, rng: np.random.Generator) -> Tuple[List[Sample], List[Sample]]:
    """Select whole groups so that ~``fraction`` of samples (per stratum) are chosen."""
    by_stratum: Dict[str, Dict[str, List[Sample]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_stratum[s.stratum][s.group].append(s)
    chosen_groups = set()
    for stratum in sorted(by_stratum):
        groups = by_stratum[stratum]
        names = sorted(groups)
        order = rng.permutation(len(names))
        n_stratum = sum(len(v) for v in groups.values())
        target = fraction * n_stratum
        taken = 0
        for i in order:
            g = names[i]
            size = len(groups[g])
            # add the group if it brings the selected count closer to the target
            if abs(taken + size - target) < abs(taken - target):
                chosen_groups.add((stratum, g))
                taken += size
        if taken == 0 and n_stratum > 1 and fraction > 0:
            # guarantee a non-empty selection: take the smallest group
            g = min(names, key=lambda n: (len(groups[n]), n))
            chosen_groups.add((stratum, g))
    sel = [s for s in samples if (s.stratum, s.group) in chosen_groups]
    rest = [s for s in samples if (s.stratum, s.group) not in chosen_groups]
    return sel, rest


def make_splits(disc: DiscoveryResult, cfg: Dict[str, Any]) -> Dict[str, List[Sample]]:
    seed = int(cfg["seed"])
    val_frac = float(cfg["data"]["val_fraction"])
    test_frac = float(cfg["data"]["test_fraction"])
    samples = sorted(disc.samples, key=lambda s: s.sample_id)

    if disc.has_official_test:
        test = [s for s in samples if s.official_split == "test"]
        dev = [s for s in samples if s.official_split == "train"]
    else:
        rng_t = np.random.default_rng(derive_seed(seed, disc.dataset, "test-split"))
        test, dev = _group_select(samples, test_frac, rng_t)
    rng_v = np.random.default_rng(derive_seed(seed, disc.dataset, "val-split"))
    val, train = _group_select(dev, val_frac, rng_v)

    splits = {"train": train, "val": val, "test": test}
    for name, lst in splits.items():
        if not lst:
            raise DatasetError(f"{disc.dataset}: '{name}' split is empty — check fractions/data.")
    check_no_leakage(splits)

    ms = cfg["data"].get("max_samples")
    if ms:  # smoke tests only: deterministic subset
        for name in splits:
            n = ms.get(name) if isinstance(ms, dict) else int(ms)
            if n:
                rng_s = np.random.default_rng(derive_seed(seed, disc.dataset, "subset", name))
                idx = sorted(rng_s.permutation(len(splits[name]))[: int(n)])
                splits[name] = [splits[name][i] for i in idx]
    return splits


def check_no_leakage(splits: Dict[str, List[Sample]]) -> None:
    ids, groups, images = {}, {}, {}
    for name, lst in splits.items():
        for s in lst:
            for key, store in ((s.sample_id, ids), (s.group, groups), (s.image, images)):
                if key in store and store[key] != name:
                    raise DatasetError(f"Leakage: '{key}' appears in both '{store[key]}' and '{name}' splits")
                store[key] = name


def split_summary(splits: Dict[str, List[Sample]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, lst in splits.items():
        strata: Dict[str, int] = defaultdict(int)
        for s in lst:
            strata[s.stratum or "all"] += 1
        out[name] = {"n": len(lst), "n_groups": len({s.group for s in lst}), "per_stratum": dict(strata)}
    total = sum(v["n"] for v in out.values())
    for name in out:
        out[name]["fraction"] = round(out[name]["n"] / max(1, total), 4)
    return out
