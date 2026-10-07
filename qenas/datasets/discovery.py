"""Dataset-specific discovery, image/mask pairing and integrity validation.

The three datasets have *different* on-disk layouts (inspected recursively
before this module was written; see ``DATASETS.md``):

BUSI   ``BUSI (Breast Ultrasound Image)/{benign,malignant,normal}/``
       ``<cls> (<n>).png`` with one or more masks ``<cls> (<n>)_mask.png``,
       ``<cls> (<n>)_mask_1.png`` ... (multiple lesions -> union of all masks).
       No official split, no patient IDs. One pixel-identical image appears in
       both ``benign`` and ``malignant`` with *different* masks.
CVC    ``CVC-ClinicD (Polyp)/PNG/{Original,Ground Truth}/<frame>.png`` and
       ``metadata.csv`` mapping every frame to one of 29 colonoscopy *sequences*.
       Frames of one sequence are near-duplicates -> splits are made per sequence.
       (The TIF copies are not readable by Pillow and are identical content.)
IDRID  ``IDRID/A.%20Segmentation/A. Segmentation/1. Original Images/{a. Training Set,b. Testing Set}``
       with masks in ``2. All Segmentation Groundtruths/<split>/<k. Lesion>/IDRiD_xx_<CODE>.tif``
       (CODE in MA, HE, EX, SE, OD). Official 54/27 train/test split.

Discovery never decides anything based on image *content* except integrity
(readability, size agreement, binary-mask validity) and exact-duplicate grouping,
both of which are required to build a leakage-free split.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..utils.logging_utils import get_logger

log = get_logger()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DatasetError(RuntimeError):
    """Serious data problem (missing/mismatched/invalid files). Never silently ignored."""


@dataclass
class Sample:
    sample_id: str
    image: str
    masks: List[str]                     # union of these masks is the target; [] -> empty mask
    group: str                           # leakage group (patient / sequence / duplicate set)
    stratum: str = ""                    # stratification label (e.g. BUSI class)
    official_split: Optional[str] = None  # "train" | "test" when the dataset defines it
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Sample":
        return Sample(**d)


@dataclass
class DiscoveryResult:
    dataset: str
    root: str
    samples: List[Sample]
    has_official_test: bool
    report: Dict[str, Any]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _find_dataset_dir(data_root: Path, key: str) -> Path:
    if not data_root.is_dir():
        raise DatasetError(f"Data root does not exist: {data_root}")
    cands = [p for p in data_root.iterdir() if p.is_dir() and p.name.upper().startswith(key.upper())]
    if not cands:
        raise DatasetError(f"No directory starting with '{key}' found in {data_root}. "
                           f"Found: {[p.name for p in data_root.iterdir()]}")
    if len(cands) > 1:
        raise DatasetError(f"Ambiguous dataset directories for '{key}': {[c.name for c in cands]}")
    return cands[0]


def _walk_find(root: Path, predicate) -> List[Path]:
    out = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            p = Path(dirpath) / d
            if predicate(p):
                out.append(p)
    return sorted(out)


def _ext_counts(root: Path) -> Dict[str, int]:
    c: Counter = Counter()
    for dirpath, _, files in os.walk(root):
        for f in files:
            c[Path(f).suffix.lower() or "<none>"] += 1
    return dict(c)


def _tree(root: Path, max_depth: int = 4) -> List[str]:
    lines = []
    base_depth = len(root.parts)
    for dirpath, dirnames, files in os.walk(root):
        dirnames.sort()
        depth = len(Path(dirpath).parts) - base_depth
        if depth > max_depth:
            continue
        exts = Counter(Path(f).suffix.lower() for f in files)
        ext_s = ", ".join(f"{n} {e}" for e, n in sorted(exts.items()))
        lines.append("  " * depth + f"{Path(dirpath).name}/" + (f"  [{ext_s}]" if ext_s else ""))
    return lines


def read_binary_mask(path: str, threshold: int) -> np.ndarray:
    """Read a mask file as a boolean array. Handles 1-bit, palette, 8/16-bit and RGB(A) masks."""
    with Image.open(path) as im:
        mode = im.mode
        if mode == "1":
            return np.array(im, dtype=bool)
        if mode == "P":
            idx = np.array(im)
            if set(np.unique(idx).tolist()) <= {0, 1}:
                return idx == 1
            arr = np.array(im.convert("RGB")).max(axis=2)
            return arr > threshold
        if mode in ("RGB", "RGBA", "LA", "CMYK", "YCbCr"):
            arr = np.array(im.convert("RGB")).max(axis=2)
        else:  # L, I, I;16, F
            arr = np.array(im)
            if arr.ndim == 3:
                arr = arr.max(axis=2)
            if arr.dtype == np.uint16 or arr.max() > 255:
                arr = (arr.astype(np.float64) / max(1.0, float(arr.max())) * 255.0)
        if arr.max() <= 1:  # masks stored as {0,1}
            return arr > 0
        return arr > threshold


def _mask_quality(path: str, threshold: int) -> Tuple[Tuple[int, int], float, float]:
    """Return (size WxH, fraction of 'intermediate' pixels, foreground fraction)."""
    with Image.open(path) as im:
        size = im.size
        mode = im.mode
        if mode in ("1", "P"):
            inter = 0.0
        else:
            a = np.array(im.convert("L"))
            inter = float(((a > 0) & (a < 255)).mean()) if a.max() > 1 else 0.0
    fg = float(read_binary_mask(path, threshold).mean())
    return size, inter, fg


def _validate_pairs(samples: List[Sample], threshold: int, allow_empty: bool,
                    max_intermediate: float = 0.25) -> Dict[str, Any]:
    """Integrity checks: readable files, matching sizes, binary-valid masks."""
    errors: List[str] = []
    warnings: List[str] = []
    modes: Counter = Counter()
    sizes: Counter = Counter()
    inter_fracs, fg_fracs = [], []
    empty = 0
    for s in samples:
        if not os.path.isfile(s.image):
            errors.append(f"missing image: {s.image}")
            continue
        try:
            with Image.open(s.image) as im:
                isz, imode = im.size, im.mode
        except Exception as exc:
            errors.append(f"unreadable image {s.image}: {exc}")
            continue
        modes[imode] += 1
        sizes[isz] += 1
        fg_total = 0.0
        for m in s.masks:
            if not os.path.isfile(m):
                errors.append(f"missing mask: {m}")
                continue
            try:
                msz, inter, fg = _mask_quality(m, threshold)
            except Exception as exc:
                errors.append(f"unreadable mask {m}: {exc}")
                continue
            if msz != isz:
                errors.append(f"size mismatch image {isz} vs mask {msz}: {s.image} | {m}")
            if inter > max_intermediate:
                errors.append(f"invalid (non-binary) mask, {inter:.1%} intermediate pixels: {m}")
            inter_fracs.append(inter)
            fg_total += fg
        fg_fracs.append(fg_total)
        if not s.masks or fg_total == 0.0:
            empty += 1
            if not allow_empty:
                warnings.append(f"empty mask for {s.sample_id}")
    if errors:
        head = "\n  ".join(errors[:20])
        raise DatasetError(f"{len(errors)} data integrity error(s) detected:\n  {head}"
                           + ("\n  ..." if len(errors) > 20 else ""))
    return {
        "n_samples": len(samples),
        "image_modes": dict(modes),
        "n_distinct_image_sizes": len(sizes),
        "most_common_image_sizes": [[list(k), v] for k, v in sizes.most_common(5)],
        "mask_intermediate_pixel_fraction_max": float(max(inter_fracs) if inter_fracs else 0.0),
        "mask_intermediate_pixel_fraction_mean": float(np.mean(inter_fracs) if inter_fracs else 0.0),
        "foreground_fraction_mean": float(np.mean(fg_fracs) if fg_fracs else 0.0),
        "n_empty_masks": empty,
        "warnings": warnings[:50],
    }


# --------------------------------------------------------------------------------------
# BUSI
# --------------------------------------------------------------------------------------

_BUSI_MASK_RE = re.compile(r"^(?P<stem>.+)_mask(?:_(?P<k>\d+))?$")


def discover_busi(data_root: Path, cfg: Dict[str, Any]) -> DiscoveryResult:
    root = _find_dataset_dir(data_root, "BUSI")
    include_normal = bool(cfg["data"]["busi_include_normal"])
    threshold = int(cfg["data"]["mask_threshold"])
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    expected = {"benign", "malignant", "normal"}
    if not expected.issubset(set(classes)):
        raise DatasetError(f"BUSI: expected class folders {sorted(expected)}, found {classes}")

    samples: List[Sample] = []
    per_class: Dict[str, Dict[str, int]] = {}
    for cls in sorted(expected):
        cdir = root / cls
        files = sorted(f for f in os.listdir(cdir) if Path(f).suffix.lower() in IMAGE_EXTS)
        images, masks = {}, defaultdict(list)
        for f in files:
            stem = Path(f).stem
            m = _BUSI_MASK_RE.match(stem)
            if m:
                masks[m.group("stem")].append(str(cdir / f))
            else:
                images[stem] = str(cdir / f)
        orphan = sorted(set(masks) - set(images))
        no_mask = sorted(set(images) - set(masks))
        if orphan:
            raise DatasetError(f"BUSI/{cls}: masks without image: {orphan[:10]}")
        if no_mask:
            raise DatasetError(f"BUSI/{cls}: images without mask: {no_mask[:10]}")
        per_class[cls] = {"images": len(images), "mask_files": sum(len(v) for v in masks.values()),
                          "multi_mask_images": sum(1 for v in masks.values() if len(v) > 1)}
        if cls == "normal" and not include_normal:
            per_class[cls]["excluded"] = len(images)
            continue
        for stem, ip in images.items():
            sid = f"{cls}/{stem}"
            samples.append(Sample(sample_id=sid, image=ip, masks=sorted(masks[stem]), group=sid, stratum=cls))

    # Exact duplicate detection (pixel hash). Duplicates must share a split; duplicates whose
    # masks disagree have ambiguous ground truth and are excluded (configurable).
    hashes: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        with Image.open(s.image) as im:
            hashes[hashlib.md5(np.asarray(im.convert("L")).tobytes()).hexdigest()].append(i)
    drop, dup_report = set(), []
    for h, idxs in hashes.items():
        if len(idxs) < 2:
            continue
        ms = [read_binary_mask_union(samples[i].masks, threshold) for i in idxs]
        identical = all(m.shape == ms[0].shape and np.array_equal(m, ms[0]) for m in ms[1:])
        ids = [samples[i].sample_id for i in idxs]
        if identical:
            for i in idxs[1:]:
                drop.add(i)   # keep a single copy
            action = "kept one copy (identical masks)"
        elif cfg["data"]["busi_drop_conflicting_duplicates"]:
            drop.update(idxs)
            action = "excluded all copies (conflicting masks/labels)"
        else:
            for i in idxs:
                samples[i].group = f"dup:{h[:12]}"
            action = "grouped into one split (conflicting masks kept)"
        dup_report.append({"ids": ids, "masks_identical": identical, "action": action})
        log.warning("BUSI exact duplicate %s -> %s", ids, action)
    samples = [s for i, s in enumerate(samples) if i not in drop]

    report = {
        "layout": "class folders; masks '<stem>_mask[_k].png' (union of all masks per image)",
        "directory": str(root),
        "tree": _tree(root),
        "extensions": _ext_counts(root),
        "per_class": per_class,
        "include_normal": include_normal,
        "exact_duplicates": dup_report,
        "official_split": None,
        "patient_metadata": "none available -> image-level split (documented limitation)",
    }
    report["validation"] = _validate_pairs(samples, threshold, allow_empty=include_normal)
    return DiscoveryResult("BUSI", str(root), samples, False, report)


def read_binary_mask_union(paths: List[str], threshold: int, size_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Union of several binary masks (empty list -> zeros of ``size_hw``)."""
    if not paths:
        if size_hw is None:
            raise ValueError("size_hw required for an empty mask list")
        return np.zeros(size_hw, dtype=bool)
    out = None
    for p in paths:
        m = read_binary_mask(p, threshold)
        if out is None:
            out = m.copy()
        else:
            if m.shape != out.shape:
                raise DatasetError(f"mask size mismatch inside union: {paths}")
            out |= m
    return out


# --------------------------------------------------------------------------------------
# CVC-ClinicDB
# --------------------------------------------------------------------------------------

def discover_cvc(data_root: Path, cfg: Dict[str, Any]) -> DiscoveryResult:
    root = _find_dataset_dir(data_root, "CVC")
    threshold = int(cfg["data"]["mask_threshold"])
    img_dir, gt_dir = root / "PNG" / "Original", root / "PNG" / "Ground Truth"
    if not img_dir.is_dir() or not gt_dir.is_dir():
        raise DatasetError(f"CVC: expected {img_dir} and {gt_dir}")
    meta_path = root / "metadata.csv"
    if not meta_path.is_file():
        raise DatasetError("CVC: metadata.csv (frame -> sequence mapping) is required to prevent "
                           "frame-level leakage between splits, but it was not found.")
    with open(meta_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    imgs = {Path(f).stem: str(img_dir / f) for f in os.listdir(img_dir) if Path(f).suffix.lower() == ".png"}
    gts = {Path(f).stem: str(gt_dir / f) for f in os.listdir(gt_dir) if Path(f).suffix.lower() == ".png"}
    unmatched = sorted(set(imgs) ^ set(gts))
    if unmatched:
        raise DatasetError(f"CVC: unmatched image/mask names: {unmatched[:10]}")
    seq_of = {}
    for r in rows:
        fid = str(r["frame_id"]).strip()
        seq_of[fid] = str(r["sequence_id"]).strip()
        png_rel = r.get("png_image_path", "")
        if png_rel and Path(png_rel).stem != fid:
            raise DatasetError(f"CVC metadata inconsistency for frame {fid}: {png_rel}")
    missing_meta = sorted(set(imgs) - set(seq_of), key=lambda x: int(x) if x.isdigit() else x)
    missing_files = sorted(set(seq_of) - set(imgs))
    if missing_meta or missing_files:
        raise DatasetError(f"CVC: frames without metadata {missing_meta[:10]}; metadata without files {missing_files[:10]}")
    samples = [Sample(sample_id=f"frame_{fid}", image=imgs[fid], masks=[gts[fid]], group=f"seq_{seq_of[fid]}",
                      stratum="", meta={"frame_id": int(fid), "sequence_id": int(seq_of[fid])})
               for fid in sorted(imgs, key=lambda x: int(x))]
    seq_sizes = Counter(s.group for s in samples)
    report = {
        "layout": "PNG/Original/<frame>.png paired with PNG/Ground Truth/<frame>.png; metadata.csv frame->sequence",
        "directory": str(root),
        "tree": _tree(root),
        "extensions": _ext_counts(root),
        "n_sequences": len(seq_sizes),
        "frames_per_sequence": dict(sorted(seq_sizes.items(), key=lambda kv: int(kv[0].split("_")[1]))),
        "tif_copies": "ignored (not decodable by Pillow; PNG copies are lossless and identical in content)",
        "official_split": None,
        "leakage_unit": "sequence (all frames of one colonoscopy sequence stay in one split)",
    }
    report["validation"] = _validate_pairs(samples, threshold, allow_empty=False)
    return DiscoveryResult("CVC", str(root), samples, False, report)


# --------------------------------------------------------------------------------------
# IDRiD (segmentation sub-challenge)
# --------------------------------------------------------------------------------------

IDRID_CODES = {"MA": "Microaneurysms", "HE": "Haemorrhages", "EX": "Hard Exudates", "SE": "Soft Exudates", "OD": "Optic Disc"}


def discover_idrid(data_root: Path, cfg: Dict[str, Any]) -> DiscoveryResult:
    root = _find_dataset_dir(data_root, "IDRID")
    threshold = int(cfg["data"]["mask_threshold"])
    target = cfg["data"]["idrid_target"].upper()
    seg_roots = _walk_find(root, lambda p: p.name.lower().endswith("segmentation") and (p / "1. Original Images").is_dir())
    if len(seg_roots) != 1:
        raise DatasetError(f"IDRID: expected exactly one segmentation root containing '1. Original Images', found {seg_roots}")
    seg = seg_roots[0]
    img_root, gt_root = seg / "1. Original Images", seg / "2. All Segmentation Groundtruths"
    if not gt_root.is_dir():
        raise DatasetError(f"IDRID: ground-truth folder not found: {gt_root}")
    codes = ["MA", "HE", "EX", "SE"] if target == "LESIONS" else [target]
    split_dirs = {"train": "a. Training Set", "test": "b. Testing Set"}
    samples: List[Sample] = []
    per_split: Dict[str, Any] = {}
    for split, sdir in split_dirs.items():
        idir = img_root / sdir
        if not idir.is_dir():
            raise DatasetError(f"IDRID: missing image folder {idir}")
        images = {Path(f).stem: str(idir / f) for f in sorted(os.listdir(idir)) if Path(f).suffix.lower() in IMAGE_EXTS}
        lesion_dirs = {}
        for d in sorted((gt_root / sdir).iterdir()):
            if d.is_dir():
                for code, name in IDRID_CODES.items():
                    if name.lower() in d.name.lower():
                        lesion_dirs[code] = d
        masks_by_code: Dict[str, Dict[str, str]] = {}
        for code, d in lesion_dirs.items():
            mm = {}
            for f in os.listdir(d):
                stem = Path(f).stem
                if not stem.upper().endswith("_" + code):
                    raise DatasetError(f"IDRID: unexpected mask name {f} in {d}")
                mm[stem[: -(len(code) + 1)]] = str(d / f)
            orphan = sorted(set(mm) - set(images))
            if orphan:
                raise DatasetError(f"IDRID/{split}/{code}: masks without image {orphan}")
            masks_by_code[code] = mm
        for code in codes:
            if code not in masks_by_code:
                raise DatasetError(f"IDRID/{split}: no ground-truth folder for target '{code}'")
        per_split[split] = {"images": len(images),
                            "masks_per_lesion": {c: len(m) for c, m in masks_by_code.items()}}
        for stem, ip in images.items():
            mlist = [masks_by_code[c][stem] for c in codes if stem in masks_by_code[c]]
            if target == "OD" and not mlist:
                raise DatasetError(f"IDRID: optic-disc mask missing for {stem}")
            samples.append(Sample(sample_id=stem, image=ip, masks=mlist, group=stem, stratum="",
                                  official_split=split, meta={"target": target}))
    report = {
        "layout": "official train/test folders; one TIF mask per lesion type (absent file = lesion absent)",
        "directory": str(seg),
        "tree": _tree(root, max_depth=6),
        "extensions": _ext_counts(root),
        "per_split": per_split,
        "target": target,
        "target_codes": codes,
        "official_split": "54 train / 27 test (used as-is; validation is carved from the official train set)",
        "patient_metadata": "none available -> image-level validation split (documented limitation)",
    }
    report["validation"] = _validate_pairs(samples, threshold, allow_empty=(target != "OD"))
    return DiscoveryResult("IDRID", str(seg), samples, True, report)


DISCOVERERS = {"BUSI": discover_busi, "CVC": discover_cvc, "IDRID": discover_idrid}


# --------------------------------------------------------------------------------------
# portability: data root detection and path re-basing (experiments can move between machines)
# --------------------------------------------------------------------------------------

def _has_dataset_dir(root: Path, key: str) -> bool:
    try:
        return root.is_dir() and any(p.is_dir() and p.name.upper().startswith(key) for p in root.iterdir())
    except OSError:
        return False


def locate_data_root(root: str, dataset: str, search_roots=("/kaggle/input",), max_depth: int = 4) -> str:
    """Return ``root`` if it contains the dataset folder; otherwise search well-known locations
    (e.g. Kaggle's read-only ``/kaggle/input``) for a folder that does."""
    key = dataset.upper()
    if _has_dataset_dir(Path(root), key):
        return str(root)
    for base in search_roots:
        base = Path(base)
        if not base.is_dir():
            continue
        for dirpath, dirnames, _ in os.walk(base):
            dirnames.sort()
            p = Path(dirpath)
            if len(p.parts) - len(base.parts) > max_depth:
                dirnames[:] = []
                continue
            if _has_dataset_dir(p, key):
                log.info("Data root '%s' has no %s folder; using auto-detected data root %s", root, key, p)
                return str(p)
    return str(root)


def relative_to_root(path: str, data_root: str) -> str:
    try:
        return Path(path).resolve().relative_to(Path(data_root).resolve()).as_posix()
    except ValueError:
        return path


def rebase_path(path: str, data_root: str, dataset: str, old_root: Optional[str] = None) -> str:
    """Map a path stored in split.json onto the current data root.

    Handles relative paths ('<dataset folder>/...'), and absolute paths written on another
    machine/OS (e.g. Windows paths read on Linux). The dataset folder is re-resolved under the
    new root, so it may even have a slightly different name (same rule as discovery: the name
    starts with the dataset key)."""
    parts = [x for x in re.split(r"[\\/]+", path) if x]
    if old_root:
        old = [x for x in re.split(r"[\\/]+", old_root) if x]
        if [x.lower() for x in parts[:len(old)]] == [x.lower() for x in old]:
            parts = parts[len(old):]
    key = dataset.upper()
    for i, part in enumerate(parts):
        if part.upper().startswith(key):
            ds_dir = _find_dataset_dir(Path(data_root), key)
            return str(ds_dir.joinpath(*parts[i + 1:]))
    return path


def discover(cfg: Dict[str, Any]) -> DiscoveryResult:
    name = cfg["data"]["dataset"].upper()
    res = DISCOVERERS[name](Path(cfg["data"]["root"]), cfg)
    if not res.samples:
        raise DatasetError(f"{name}: dataset is empty after discovery")
    return res
