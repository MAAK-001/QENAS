"""Inspect dataset structure, validate image/mask pairing, and print the resulting splits.

Usage (from the QENAS directory):
    python scripts/inspect_datasets.py                 # all three datasets
    python scripts/inspect_datasets.py --dataset CVC
    python scripts/inspect_datasets.py --save          # also writes results/<DS>/dataset_report.json

No model is built and no test image is used for anything except integrity checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qenas.config import QENAS_ROOT, SUPPORTED_DATASETS, build_config  # noqa: E402
from qenas.datasets.discovery import discover  # noqa: E402
from qenas.datasets.splits import make_splits, split_summary  # noqa: E402
from qenas.utils.io import save_json  # noqa: E402
from qenas.utils.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=SUPPORTED_DATASETS, nargs="*", default=list(SUPPORTED_DATASETS))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--tree", action="store_true", help="print the directory tree")
    args = ap.parse_args()
    setup_logging()
    for ds in args.dataset:
        cfg = build_config(ds, overrides={"data.root": args.data_root, "seed": args.seed})
        disc = discover(cfg)
        splits = make_splits(disc, cfg)
        summ = split_summary(splits)
        rep = disc.report
        print("=" * 72)
        print(f"{ds}  ({rep['directory']})")
        print("-" * 72)
        print(f"Layout      : {rep['layout']}")
        print(f"Extensions  : {rep['extensions']}")
        for k in ("per_class", "per_split", "n_sequences", "target", "exact_duplicates", "official_split",
                  "patient_metadata", "leakage_unit"):
            if k in rep and rep[k] not in (None, [], {}):
                print(f"{k:12s}: {rep[k]}")
        v = rep["validation"]
        print(f"Validation  : {v['n_samples']} pairs OK | image modes {v['image_modes']} | "
              f"{v['n_distinct_image_sizes']} distinct sizes, most common {v['most_common_image_sizes'][:2]}")
        print(f"              mask intermediate-pixel fraction max {v['mask_intermediate_pixel_fraction_max']:.4f} "
              f"(binarised at >{cfg['data']['mask_threshold']}) | mean foreground {v['foreground_fraction_mean']:.4f} "
              f"| empty masks {v['n_empty_masks']}")
        if args.tree:
            print("\n".join(rep["tree"]))
        print(f"\n{ds}")
        for name in ("train", "val", "test"):
            s = summ[name]
            label = {"train": "Train", "val": "Validation", "test": "Test"}[name]
            print(f"{label}: {s['n']}  (groups {s['n_groups']}, {s['fraction']:.1%}, {s['per_stratum']})")
        if args.save:
            out = QENAS_ROOT / "results" / ds / "dataset_report.json"
            save_json(out, {"report": rep, "split_summary": summ,
                            "splits": {k: [x.sample_id for x in v] for k, v in splits.items()}})
            print(f"saved {out}")


if __name__ == "__main__":
    main()
