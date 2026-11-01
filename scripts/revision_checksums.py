"""Identity check of code, splits, data and software across the Kaggle accounts of the revision (REVISION.md §5).

    python scripts/revision_checksums.py --data-root ../Datasets --write-reference    # once, on your PC
    python scripts/revision_checksums.py --data-root <root> --out results/revision_checksums_1.json --check
    python scripts/revision_checksums.py --compare results/revision_checksums_*.json

``--write-reference`` stores the data digests of your local datasets in ``splits/data_checksums.json`` (committed with
the code). ``--check`` exits with code 4 if this machine's split or data digest differs from that reference, so the
Kaggle cell stops before any training. ``--compare`` checks that all accounts ran identical code, data and software.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qenas.study.revision import SPLITS_DIR, checksums, reference_split  # noqa: E402
from qenas.utils.io import load_json, save_json  # noqa: E402

DATASETS = ("BUSI", "CVC", "IDRID", "KVASIR")
REFERENCE = SPLITS_DIR / "data_checksums.json"


def compare(files) -> int:
    recs = {Path(f).name: load_json(f) for f in files}
    bad = 0
    for what, get in (("code", lambda r: r["code"]["sha256"]),
                      *[(f"data {d}", lambda r, d=d: r["data"].get(d, {}).get("sha256")) for d in DATASETS],
                      *[(f"split {d}", lambda r, d=d: r["splits"].get(d)) for d in DATASETS],
                      *[(f"software {k}", lambda r, k=k: r["environment"].get(k))
                        for k in ("torch_version", "cuda_version", "cudnn_version", "numpy_version", "pymoo_version",
                                  "nnunetv2", "timm_version", "gpu_name")]):
        vals = {n: get(r) for n, r in recs.items()}
        distinct = {json.dumps(v) for v in vals.values()}
        ok = len(distinct) == 1
        bad += not ok
        print(f"{'OK  ' if ok else 'DIFF'} {what:24s} " + ("" if ok else json.dumps(vals)))
    for n, r in recs.items():
        if r["environment"].get("differences_from_study_v2"):
            print(f"note {n}: software differs from study v2: {r['environment']['differences_from_study_v2']}")
    print(f"\n{len(recs)} accounts compared: " + ("identical" if not bad else f"{bad} differences"))
    return 0 if not bad else 4


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None)
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--out", default=None)
    p.add_argument("--write-reference", action="store_true")
    p.add_argument("--check", action="store_true", help="exit 4 if split or data differ from the reference")
    p.add_argument("--compare", nargs="+", default=None, help="checksum files of the accounts")
    a = p.parse_args(argv)
    if a.compare:
        return compare(a.compare)
    res = checksums(a.data_root, a.datasets)
    print(json.dumps({"code": res["code"], "splits": {k: v[:16] for k, v in res["splits"].items()},
                      "data": {k: {"sha256": v["sha256"][:16], "files": v["files"], "missing": len(v["missing_samples"])}
                               for k, v in res["data"].items()},
                      "environment": res["environment"]}, indent=1))
    if a.out:
        save_json(Path(a.out), res)
    if a.write_reference:
        save_json(REFERENCE, {d: {"sha256": v["sha256"], "files": v["files"]} for d, v in res["data"].items()})
        print(f"wrote {REFERENCE}")
    if a.check:
        ref = load_json(REFERENCE) if REFERENCE.exists() else {}
        problems = [f"{d}: missing samples {v['missing_samples'][:5]}" for d, v in res["data"].items()
                    if v["missing_samples"]]
        problems += [f"{d}: data sha256 {v['sha256'][:12]} != reference {ref[d]['sha256'][:12]}"
                     for d, v in res["data"].items() if d in ref and v["sha256"] != ref[d]["sha256"]]
        problems += [f"{d}: split file corrupted" for d in a.datasets
                     if reference_split(d)["sha256"] != res["splits"][d]]
        if problems:
            print("CHECK FAILED:\n  " + "\n  ".join(problems))
            return 4
        print("CHECK OK: splits and data identical to the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
