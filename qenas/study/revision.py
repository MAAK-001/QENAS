"""Journal revision: identity checks across Kaggle accounts (REVISION.md §5).

Every account must run the identical code on the identical data with the identical split. Three fingerprints:

* **code** - SHA-256 over every tracked source file that can influence a result (``qenas/**.py``, ``study.py``,
  ``main.py``, ``splits/*.json``), path and content, in sorted order;
* **splits** - ``splits/<DATASET>.json``: the ordered sample ids of train / validation / test of study v2 (split seed
  42, identical in all 105 v2 experiments). With ``QENAS_REFERENCE_SPLITS`` set (the Kaggle cell sets it), the split
  stage of every experiment refuses to continue if its split differs in membership *or order* (the order fixes the
  data order of training);
* **data** - per dataset, SHA-256 over the bytes of every image and mask file the split lists, in split order.

``python scripts/revision_checksums.py`` prints / writes them (``revision_checksums_<worker>.json``); the merge
script compares the files of all accounts against ``splits/REFERENCE_CHECKSUMS.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..config import QENAS_ROOT, build_config
from ..utils.io import load_json, save_json

SPLITS_DIR = QENAS_ROOT / "splits"
REFERENCE_ENV = "QENAS_REFERENCE_SPLITS"
CODE_GLOBS = ("qenas/**/*.py", "study.py", "main.py", "splits/*.json")
# the software versions of study v2 (environment.json of all 105 experiments)
REFERENCE_ENVIRONMENT = {"torch_version": "2.11.0+cu128", "cuda_version": "12.8", "cudnn_version": 91900,
                         "python_version": "3.13.15", "numpy_version": "2.1.3", "pymoo_version": "0.6.2",
                         "scipy_version": "1.16.3", "gpu_name": "Tesla T4", "nnunetv2": "2.8.1"}


def split_ids(splits: Dict[str, Sequence[Any]]) -> Dict[str, List[str]]:
    """Ordered sample ids per subset (``Sample`` objects or their dicts)."""
    return {k: [s["sample_id"] if isinstance(s, dict) else s.sample_id for s in v] for k, v in splits.items()}


def ids_digest(ids: Dict[str, List[str]]) -> str:
    return hashlib.sha256(json.dumps(ids, sort_keys=True).encode("utf-8")).hexdigest()


def reference_split(dataset: str, directory: Optional[Path] = None) -> Dict[str, Any]:
    return load_json(Path(directory or SPLITS_DIR) / f"{dataset}.json")


def verify_reference_split(dataset: str, splits: Dict[str, Sequence[Any]]) -> Optional[str]:
    """No-op unless ``QENAS_REFERENCE_SPLITS`` is set (to '1' = the repository's ``splits/`` or to a folder).
    Raises if the split differs from the saved split of study v2; returns its digest."""
    flag = os.environ.get(REFERENCE_ENV)
    if not flag:
        return None
    ref = reference_split(dataset, None if flag == "1" else Path(flag))
    ids = split_ids(splits)
    if ids != ref["ids"]:
        diff = {k: (len(ids.get(k, [])), len(ref["ids"].get(k, []))) for k in ref["ids"]}
        raise RuntimeError(f"The {dataset} split differs from the saved study-v2 split (splits/{dataset}.json; sizes "
                           f"now vs saved: {diff}). Check the dataset version (scripts/revision_checksums.py).")
    return ids_digest(ids)


def write_reference_splits(results: Path, out: Path = SPLITS_DIR) -> Dict[str, str]:
    """``splits/<DATASET>.json`` from the study-v2 experiments (all must agree)."""
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    for ds_dir in sorted(p for p in Path(results).iterdir() if p.is_dir()):
        files = sorted(ds_dir.glob("*/seed_*/split.json"))
        if not files:
            continue
        all_ids = {ids_digest(split_ids(load_json(f)["splits"])) for f in files}
        if len(all_ids) != 1:
            raise RuntimeError(f"{ds_dir.name}: the saved splits disagree ({len(all_ids)} variants)")
        d = load_json(files[0])
        ids = split_ids(d["splits"])
        digests[ds_dir.name] = ids_digest(ids)
        save_json(out / f"{ds_dir.name}.json", {
            "dataset": ds_dir.name, "split_seed": d["split_seed"], "sizes": {k: len(v) for k, v in ids.items()},
            "sha256": digests[ds_dir.name], "source": f"{len(files)} study-v2 experiments, all identical",
            "ids": ids})
    return digests


def code_digest(root: Path = QENAS_ROOT) -> Dict[str, Any]:
    files = sorted({p for g in CODE_GLOBS for p in Path(root).glob(g) if p.is_file() and "__pycache__" not in p.parts})
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode() + b"\0")
        h.update(p.read_bytes().replace(b"\r\n", b"\n") + b"\0")      # line endings do not matter
    return {"sha256": h.hexdigest(), "files": len(files)}


def data_digest(dataset: str, data_root: str) -> Dict[str, Any]:
    """SHA-256 of all image and mask bytes of the dataset's split, in split order (no image is decoded)."""
    from ..datasets.discovery import discover, locate_data_root
    cfg = build_config(dataset, overrides={"data.root": locate_data_root(data_root, dataset)})
    by_id = {s.sample_id: s for s in discover(cfg).samples}
    ref = reference_split(dataset)
    h = hashlib.sha256()
    n = 0
    missing = []
    for subset in ("train", "val", "test"):
        for sid in ref["ids"][subset]:
            s = by_id.get(sid)
            if s is None:
                missing.append(sid)
                continue
            for f in [s.image] + list(s.masks):
                h.update(sid.encode() + b"\0" + Path(f).read_bytes())
                n += 1
    return {"sha256": h.hexdigest(), "files": n, "missing_samples": missing}


def environment_digest() -> Dict[str, Any]:
    import platform

    import numpy
    import scipy
    import torch
    info = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "python_version": platform.python_version(), "numpy_version": numpy.__version__,
            "scipy_version": scipy.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for mod, key in (("pymoo", "pymoo_version"), ("nnunetv2", "nnunetv2"), ("timm", "timm_version")):
        try:
            from importlib.metadata import version
            info[key] = version(mod)
        except Exception:
            info[key] = None
    info["differences_from_study_v2"] = {k: (info.get(k), v) for k, v in REFERENCE_ENVIRONMENT.items()
                                         if info.get(k) is not None and str(info.get(k)) != str(v)}
    return info


def checksums(data_root: Optional[str], datasets: Sequence[str] = ("BUSI", "CVC", "IDRID", "KVASIR")) -> Dict[str, Any]:
    out: Dict[str, Any] = {"code": code_digest(), "splits": {}, "data": {}, "environment": environment_digest()}
    for ds in datasets:
        out["splits"][ds] = reference_split(ds)["sha256"]
        if data_root:
            out["data"][ds] = data_digest(ds, data_root)
    return out
