"""Atomic, corruption-resistant file I/O helpers (JSON, CSV, pickle, torch)."""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _to_jsonable(float(obj))
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)  # never write invalid JSON
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, range):
        return list(obj)
    return obj


def retry_on_lock(fn, *args, attempts: int = 30, **kwargs):
    """Run ``fn`` retrying on PermissionError.

    On Windows, cloud-sync clients (OneDrive) and virus scanners briefly lock files that
    were just written, which makes ``os.replace``/``copy`` fail with "Access is denied".
    Waits up to ~20 s in total before giving up.
    """
    for k in range(attempts):
        try:
            return fn(*args, **kwargs)
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(min(0.05 * (k + 1), 1.0))


def atomic_write_bytes(path: os.PathLike, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        retry_on_lock(os.replace, tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def save_json(path: os.PathLike, obj: Any, indent: int = 2) -> None:
    text = json.dumps(_to_jsonable(obj), indent=indent, sort_keys=False)
    atomic_write_bytes(path, text.encode("utf-8"))


def load_json(path: os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_pickle(path: os.PathLike, obj: Any) -> None:
    atomic_write_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def load_pickle(path: os.PathLike) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_csv(path: os.PathLike, rows: Sequence[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    """Write a list of dict rows as CSV (atomic). Field order follows first appearance."""
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r.keys():
                if k not in fieldnames:
                    fieldnames.append(k)
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: _csv_value(r.get(k, "")) for k in fieldnames})
    atomic_write_bytes(path, buf.getvalue().encode("utf-8"))


def _csv_value(v: Any) -> Any:
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(str(int(x)) if isinstance(x, (int, np.integer)) else str(x) for x in v) + "]"
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def save_markdown_table(path: os.PathLike, rows: Sequence[Dict[str, Any]], columns: Sequence[str],
                        title: Optional[str] = None, float_fmt: str = "{:.4f}") -> str:
    """Render rows as a GitHub-flavoured Markdown table, save it and return the text."""

    def fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "yes" if v else ""
        if isinstance(v, (float, np.floating)):
            return float_fmt.format(float(v))
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(str(x) for x in v) + "]"
        return str(v)

    lines = []
    if title:
        lines.append(f"### {title}\n")
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c, "")) for c in columns) + " |")
    text = "\n".join(lines) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))
    return text


def ensure_dir(path: os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
