"""Paper analysis of the final QENAS (QENAS 3.0, study v2): every table and figure the paper uses.

    python scripts/paper_analysis.py

Reads the finished study (results/), the preliminary study (Old Results/study_v1) and the datasets (../Datasets, for
the ground-truth masks of the extra metrics), and writes results/paper/:

* tables/*.csv and report.json — one table per metric (all models, all datasets), statistics, ablations, cost;
* figures/*.png — the paper figures (300 dpi);
* per-experiment figures of representative runs are copied from results/<DATASET>/<method>/seed_<s>/figures.

Published results of other methods (other splits; marked as such everywhere) are taken from
  [MG]  Hu et al., Mixed-GGNAS, Expert Systems with Applications 289 (2025) 128338, Tables 4, 5, 7, 8;
  [JHA] Jha et al., Real-time polyp detection, localization and segmentation in colonoscopy using deep learning,
        IEEE Access 9 (2021) 40496-40510, Tables IV, V (Kvasir-SEG, 880 training / 120 evaluation images).
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qenas.config import build_config  # noqa: E402
from qenas.datasets.discovery import Sample, discover, locate_data_root, rebase_path  # noqa: E402
from qenas.datasets.preprocessing import preload_split  # noqa: E402
from qenas.visualization.style import (BLOCK_COLORS, GRID, INK, INK_2, MUTED, SERIES, apply_style,  # noqa: E402
                                       plt, save)

RES = ROOT / "results"
OLD = ROOT / "Old Results" / "study_v1"
OUT = RES / "paper"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = ROOT.parent / "Datasets"
CACHE = ROOT / "cache" / "preprocessed"

DATASETS = ["BUSI", "CVC", "IDRID", "KVASIR"]
DSL = {"BUSI": "BUSI", "CVC": "CVC-ClinicDB", "IDRID": "IDRiD", "KVASIR": "Kvasir-SEG", "EXT": "Kvasir→CVC"}
SEEDS = [45, 46, 47]
MAIN = ["qenas", "nnunet", "attention_unet", "unet"]
LABEL = {"qenas": "QENAS (ours)", "nnunet": "nnU-Net", "attention_unet": "Attention U-Net", "unet": "U-Net",
         "qenas_fixed_alpha": "QENAS, fixed α = 0.5", "qenas_random_pick": "QENAS, random candidate",
         "qenas_zico_pick": "QENAS, zero-shot (max ZiCo)"}
COLOR = {"qenas": SERIES[0], "nnunet": SERIES[1], "attention_unet": SERIES[2], "unet": SERIES[3]}
MG_COLOR = SERIES[6]
BLOCK = {0: "Residual", 1: "Dense", 2: "Inception", 3: "ConvNeXt", 4: "DARTS"}

# ------------------------------------------------------------------------------------------- published results
MG_SRC = "MG"
MG = {  # method: strategy, {dataset: (Dice, mIoU, Params M, FLOPs G per batch)}  - Hu et al. 2025, Tables 4, 5, 7
    "U-Net": ("manual", {"CVC": (88.79, 89.91, 31.04, 437.94), "BUSI": (76.82, 80.23, 31.04, 437.33),
                         "IDRID": (93.08, 94.01, 31.04, 547.42)}),
    "TransUNet": ("manual", {"CVC": (92.04, 93.03, 93.23, 257.92), "BUSI": (79.98, 80.30, 93.23, 257.91),
                             "IDRID": (95.54, 95.80, 93.23, 201.59)}),
    "Attention U-Net": ("manual", {"CVC": (90.05, 90.96, 8.73, 134.56), "BUSI": (78.03, 80.77, 8.73, 134.25),
                                   "IDRID": (93.96, 94.80, 8.73, 168.19)}),
    "NAS-Unet": ("NAS, gradient", {"CVC": (79.88, 83.33, 1.07, 103.44), "BUSI": (77.86, 79.75, 1.33, 118.94),
                                   "IDRID": (93.90, 94.15, 87.32, 481.47)}),
    "HCT-Net": ("NAS, gradient", {"CVC": (90.75, 90.80, 6.79, 126.16), "BUSI": (78.92, 78.65, 6.79, 127.17),
                                  "IDRID": (93.64, 93.88, 7.74, 175.29)}),
    "UNAS-Net": ("NAS, evolution", {"CVC": (89.72, 90.98, 2.49, 53.169), "BUSI": (80.49, 82.49, 6.21, 123.98),
                                    "IDRID": (95.31, 95.60, 1.44, 33.97)}),
    "Genetic U-Net": ("NAS, evolution", {"CVC": (90.17, 90.65, 0.30, 41.99), "BUSI": (79.88, 80.52, 0.31, 61.49),
                                         "IDRID": (96.09, 96.30, 0.22, 45.09)}),
    "MedUNAS": ("NAS, evolution", {"CVC": (91.38, 92.02, 95.41, 703.35), "BUSI": (80.42, 82.58, 15.56, 359.60),
                                   "IDRID": (91.94, 93.19, 130.8, 2425.3)}),
    "BiX-NAS": ("NAS, evolution", {"CVC": (90.26, 92.15, 2.09, 593.07), "BUSI": (77.31, 81.58, 2.09, 648.13),
                                   "IDRID": (93.26, 93.74, 2.09, 1620.71)}),
    "Mixed-GGNAS": ("NAS, evolution + gradient", {"CVC": (93.51, 94.72, 6.01, 88.08),
                                                  "BUSI": (80.64, 82.77, 7.86, 66.10),
                                                  "IDRID": (96.63, 96.78, 29.41, 335.34)}),
}
MG_SEARCH_H = {"NAS-Unet": 3, "HCT-Net": 3, "UNAS-Net": 19, "Genetic U-Net": 4, "MedUNAS": 11, "BiX-NAS": 4,
               "Mixed-GGNAS (w/o ViT)": 15, "Mixed-GGNAS (with ViT)": 26}          # Hu et al. 2025, Table 8 (IDRiD)
JHA_SRC = "JHA"
JHA = {  # method: (DSC, Jaccard = foreground IoU, F2, Precision, Recall, Accuracy, Params)  - Jha et al. 2021
    "U-Net [JHA]": (0.5969, 0.4713, 0.5980, 0.6722, 0.6171, 0.8936, 7858433),
    "ResUNet": (0.6902, 0.5721, 0.6986, 0.7454, 0.7248, 0.9169, 8420077),
    "ResUNet++": (0.7143, 0.6126, 0.7198, 0.7836, 0.7419, 0.9172, 16242785),
    "FCN8 (VGG16)": (0.8310, 0.7365, 0.8248, 0.8817, 0.8346, 0.9524, 134270278),
    "HRNet": (0.8446, 0.7592, 0.8467, 0.8778, 0.8588, 0.9524, 9524036),
    "DoubleUNet (VGG19)": (0.8129, 0.7332, 0.8207, 0.8611, 0.8402, 0.9489, 29303426),
    "PSPNet (ResNet50)": (0.8406, 0.7444, 0.8314, 0.8901, 0.8357, 0.9525, 48631850),
    "DeepLabv3+ (ResNet50)": (0.8572, 0.7759, 0.8545, 0.8907, 0.8616, 0.9614, 39756962),
    "DeepLabv3+ (ResNet101)": (0.8643, 0.7862, 0.8570, 0.9064, 0.8592, 0.9608, 58749090),
    "U-Net (ResNet34)": (0.8757, 0.8100, 0.8622, 0.9435, 0.8597, 0.9681, 33509098),
    "ColonSegNet": (0.8206, 0.7239, 0.8206, 0.8435, 0.8496, 0.9493, 5014049),
}
SOURCES = {MG_SRC: "Hu et al., ESWA 2025 (Mixed-GGNAS), own splits; CVC-ClinicDB with a random frame split",
           JHA_SRC: "Jha et al., IEEE Access 2021, Kvasir-SEG 880 training / 120 evaluation images"}


# ------------------------------------------------------------------------------------------- helpers
def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def exp(ds, m, s):
    return RES / ds / m / f"seed_{s}"


def mean_sd(v, scale=1.0, digits=2):
    v = [x * scale for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return "–"
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{np.mean(v):.{digits}f} ± {np.std(v, ddof=1):.{digits}f}"


def write_csv(name, header, rows):
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def holm(p):
    idx = sorted(range(len(p)), key=lambda i: p[i])
    out, run = [1.0] * len(p), 0.0
    for r, i in enumerate(idx):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


# ------------------------------------------------------------------------------------------- ground truth
def test_ground_truth(ds):
    """Ground-truth masks of the test split at the network resolution (exactly as the test stage loads them)."""
    ref = exp(ds, "qenas", SEEDS[0])
    sp = load(ref / "split.json")
    cfg = load(ref / "config.json")["config"]
    root = locate_data_root(str(DATA), ds)
    samples = []
    for d in sp["splits"]["test"]:
        s = Sample.from_dict(d)
        s.image = rebase_path(s.image, root, ds, sp.get("data_root"))
        s.masks = [rebase_path(m, root, ds, sp.get("data_root")) for m in s.masks]
        samples.append(s)
    _, masks = preload_split(samples, tuple(cfg["data"]["image_size"]), int(cfg["data"]["mask_threshold"]), ds,
                             "test", CACHE)
    return [s.sample_id for s in samples], masks


def external_ground_truth():
    cfg = build_config("CVC", overrides={"data.root": locate_data_root(str(DATA), "CVC"), "paths.cache": str(CACHE.parent)})
    samples = sorted(discover(cfg).samples, key=lambda s: s.sample_id)
    _, masks = preload_split(samples, (256, 256), int(cfg["data"]["mask_threshold"]), "CVC", "all", CACHE)
    return [s.sample_id for s in samples], masks


def confusion_metrics(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = float((p & g).sum())
    fp = float((p & ~g).sum())
    fn = float((~p & g).sum())
    tn = float((~p & ~g).sum())
    prec = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)   # empty prediction of a lesion -> 0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return {"precision": prec, "specificity": tn / (tn + fp) if tn + fp else 1.0,
            "accuracy": (tp + tn) / (tp + tn + fp + fn), "f2": f2, "dice_check": dice}


def extended_metrics():
    """Precision, specificity, accuracy and F2 of every main-method run (test and external), from the saved binary
    predictions; the Dice recomputed from them must equal the stored per-image Dice."""
    out, worst = {}, 0.0
    for ds in DATASETS:
        ids, gts = test_ground_truth(ds)
        for m in MAIN:
            for s in SEEDS:
                d = exp(ds, m, s)
                z = np.load(d / "test_predictions.npz")
                assert list(z["sample_ids"]) == ids, (ds, m, s)
                rows = [confusion_metrics(p, g) for p, g in zip(z["masks"], gts)]
                stored = [float(r["dice"]) for r in read_csv(d / "test_per_image.csv")]
                worst = max(worst, max(abs(a["dice_check"] - b) for a, b in zip(rows, stored)))
                out[(ds, m, s)] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    ids, gts = external_ground_truth()
    for m in MAIN:
        for s in SEEDS:
            d = exp("KVASIR", m, s) / "external_CVC"
            z = np.load(d / "external_predictions.npz")
            assert list(z["sample_ids"]) == ids
            rows = [confusion_metrics(p, g) for p, g in zip(z["masks"], gts)]
            out[("EXT", m, s)] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print(f"[extended metrics] max |Dice recomputed - stored| = {worst:.2e}")
    assert worst < 1e-6, "recomputed Dice differs from the stored test Dice"
    return out


# ------------------------------------------------------------------------------------------- collected results
def finals(ds, m):
    return [load(exp(ds, m, s) / "final_metrics.json") for s in SEEDS]


def externals(m):
    return [load(exp("KVASIR", m, s) / "external_CVC" / "external_metrics.json")["metrics"] for s in SEEDS]


def total_minutes(ds, m):
    return [load(exp(ds, m, s) / "state.json")["total_seconds"] / 60 for s in SEEDS]


def stage_minutes(ds, m, stage):
    return [load(exp(ds, m, s) / "state.json")["stages"].get(stage, {}).get("seconds", 0) / 60 for s in SEEDS]


def per_image(ds, m, key="dice", root=RES, seeds=SEEDS):
    """sample_id -> value averaged over seeds."""
    acc = {}
    for s in seeds:
        for r in read_csv(root / ds / m / f"seed_{s}" / "test_per_image.csv"):
            acc.setdefault(r["sample_id"], []).append(float(r[key]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def per_image_ext(m, key="dice"):
    acc = {}
    for s in SEEDS:
        for r in read_csv(exp("KVASIR", m, s) / "external_CVC" / "external_per_image.csv"):
            acc.setdefault(r["sample_id"], []).append(float(r[key]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired(a, b):
    ids = sorted(set(a) & set(b))
    x, y = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
    d = x - y
    nz = d[d != 0]
    p = 1.0 if len(nz) == 0 else float(wilcoxon(x, y).pvalue)
    r = rankdata(np.abs(nz))
    rb = float((r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()) if len(nz) else 0.0
    boot = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
    return {"n": len(ids), "a": x.mean(), "b": y.mean(), "diff": d.mean(), "lo": np.percentile(boot, 2.5),
            "hi": np.percentile(boot, 97.5), "p": p, "rb": rb, "better": int((d > 0).sum()), "worse": int((d < 0).sum())}


# ------------------------------------------------------------------------------------------- metric tables
METRIC_DEFS = [  # key, title, scale, digits, lower_is_better
    ("dice", "Dice similarity coefficient (DSC, %)", 100, 2, False),
    ("miou", "Mean IoU (class mean of foreground and background, %)", 100, 2, False),
    ("iou", "Foreground IoU (Jaccard, %)", 100, 2, False),
    ("hd95", "95th-percentile Hausdorff distance (HD95, pixels)", 1, 2, True),
    ("sensitivity", "Sensitivity (recall, %)", 100, 2, False),
    ("precision", "Precision (positive predictive value, %)", 100, 2, False),
    ("specificity", "Specificity (%)", 100, 2, False),
    ("accuracy", "Pixel accuracy (%)", 100, 2, False),
    ("f2", "F2-score (%)", 100, 2, False),
]
COLS = DATASETS + ["EXT"]


def metric_tables(ext):
    tables = {}
    for key, title, sc, dg, lower in METRIC_DEFS:
        header = ["Model", "Source / split"] + [DSL[c] for c in COLS]
        rows, best = [], {}
        for m in MAIN:
            row = [LABEL[m], "this study, identical split, 3 seeds"]
            for c in COLS:
                if key in ("precision", "specificity", "accuracy", "f2"):
                    vals = [ext[(c, m, s)][key] for s in SEEDS]
                elif c == "EXT":
                    vals = [e[key] for e in externals(m)]
                else:
                    vals = [f["test"][key] for f in finals(c, m)]
                row.append(mean_sd(vals, sc, dg))
                best.setdefault(c, []).append((np.mean(vals) * sc, m))
            rows.append(row)
        # published rows (other splits)
        if key in ("dice", "miou"):
            for name, (strat, per) in MG.items():
                row = [name + " †", f"[MG] {strat}"]
                for c in COLS:
                    row.append(f"{per[c][0 if key == 'dice' else 1]:.2f}" if c in per else "–")
                rows.append(row)
        jha_idx = {"dice": 0, "iou": 1, "f2": 2, "precision": 3, "sensitivity": 4, "accuracy": 5}
        if key in jha_idx:
            for name, v in JHA.items():
                rows.append([name + " ‡", "[JHA]"] + [f"{100 * v[jha_idx[key]]:.2f}" if c == "KVASIR" else "–"
                                                       for c in COLS])
        winner = {c: (min if lower else max)(best[c])[1] for c in COLS}
        tables[key] = {"title": title, "header": header, "rows": rows, "lower_is_better": lower,
                       "best_own": {DSL[c]: LABEL[winner[c]] for c in COLS}}
        write_csv(f"metric_{key}", header, rows)
    # cost tables
    cost_defs = [("params_m", "Trainable parameters (M)", 2), ("gflops_mixed_ggnas", "FLOPs per training batch "
                 "(G; thop convention of the published tables †: batch 8, IDRiD 4)", 1),
                 ("gflops_per_image", "FLOPs per image (G)", 2),
                 ("inference_ms_per_image", "Inference time per image (ms; Tesla T4, batch 1, median of 50)", 1)]
    for key, title, dg in cost_defs:
        header = ["Model", "Source / split"] + [DSL[c] for c in DATASETS]
        rows = [[LABEL[m], "this study"] + [mean_sd([f[key] for f in finals(c, m)], 1, dg) for c in DATASETS]
                for m in MAIN]
        if key in ("params_m", "gflops_mixed_ggnas", "gflops_per_image"):
            for name, (strat, per) in MG.items():
                vals = []
                for c in DATASETS:
                    if c not in per:
                        vals.append("–")
                    elif key == "params_m":
                        vals.append(f"{per[c][2]:.2f}")
                    elif key == "gflops_mixed_ggnas":
                        vals.append(f"{per[c][3]:.1f}")
                    else:
                        vals.append(f"{per[c][3] / (4 if c == 'IDRID' else 8):.2f}")
                rows.append([name + " †", f"[MG] {strat}"] + vals)
        if key == "inference_ms_per_image":
            for name, fps in JHA_FPS.items():
                rows.append([name + " ‡", "[JHA] RTX 2080 Ti, 512×512"] + [f"{1000 / fps:.1f}" if c == "KVASIR" else "–"
                                                                           for c in DATASETS])
        if key == "params_m":
            for name, v in JHA.items():
                rows.append([name + " ‡", "[JHA]"] + [f"{v[6] / 1e6:.2f}" if c == "KVASIR" else "–" for c in DATASETS])
        tables[key] = {"title": title, "header": header, "rows": rows}
        write_csv(f"cost_{key}", header, rows)
    return tables


# ------------------------------------------------------------------------------------------- statistics
def statistics():
    pair_rows, fr_rows, ranks = [], [], {}
    for c in COLS:
        get = (lambda m: per_image_ext(m)) if c == "EXT" else (lambda m, c=c: per_image(c, m))
        data = {m: get(m) for m in MAIN}
        tests = [(m, paired(data["qenas"], data[m])) for m in MAIN[1:]]
        for (m, r), p_adj in zip(tests, holm([r["p"] for _, r in tests])):
            pair_rows.append([DSL[c], f"QENAS vs {LABEL[m]}", r["n"], f"{100 * r['a']:.2f}", f"{100 * r['b']:.2f}",
                              f"{100 * r['diff']:+.2f}", f"[{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]",
                              fmt_p(r["p"]), fmt_p(p_adj), f"{r['rb']:+.2f}", f"{r['better']} / {r['worse']}",
                              "QENAS better" if p_adj < 0.05 and r["diff"] > 0 else
                              f"{LABEL[m]} better" if p_adj < 0.05 else "no significant difference"])
        ids = sorted(set.intersection(*[set(v) for v in data.values()]))
        M = np.array([[data[m][i] for m in MAIN] for i in ids])
        st, p = friedmanchisquare(*M.T)
        rk = np.mean([rankdata(-row) for row in M], axis=0)
        k, n = len(MAIN), len(ids)
        cd = 2.569 * math.sqrt(k * (k + 1) / (6 * n))          # Nemenyi, alpha = 0.05, k = 4
        ranks[c] = {"ranks": dict(zip(MAIN, rk.tolist())), "cd": cd, "n": n, "p": p}
        fr_rows.append([DSL[c], n, f"{st:.1f}", fmt_p(p)] + [f"{x:.2f}" for x in rk] + [f"{cd:.2f}"])
    h1 = ["Test set", "Comparison", "Images", "Dice QENAS (%)", "Dice other (%)", "Δ Dice (pp)", "95 % CI (pp)",
          "p (Wilcoxon)", "p (Holm)", "Effect size r", "Images better / worse", "Verdict"]
    h2 = ["Test set", "Images", "Friedman χ²", "p"] + [f"Mean rank {LABEL[m]}" for m in MAIN] + ["Nemenyi CD"]
    write_csv("stats_pairwise_dice", h1, pair_rows)
    write_csv("stats_friedman_ranks", h2, fr_rows)
    return {"pairwise": {"header": h1, "rows": pair_rows}, "friedman": {"header": h2, "rows": fr_rows}}, ranks


# ------------------------------------------------------------------------------------------- ablations
def ablations():
    rows = []

    def add(study, ds, name, ref, other, ref_lab="QENAS 3.0"):
        r = paired(ref, other)
        rows.append([study, DSL[ds], name, f"{100 * r['a']:.2f}", f"{100 * r['b']:.2f}", f"{100 * r['diff']:+.2f}",
                     f"[{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]", r["p"], f"{r['rb']:+.2f}"])
        return r

    forest = []
    for ds in DATASETS:
        r = add("v2", ds, "Loss-weight search (α grid) vs fixed α = 0.5", per_image(ds, "qenas"),
                per_image(ds, "qenas_fixed_alpha"))
        forest.append((ds, "α grid vs fixed α = 0.5", r))
    for ds in ("BUSI", "KVASIR"):
        for m, nm in (("qenas_random_pick", "Selection vs random candidate"),
                      ("qenas_zico_pick", "Selection vs zero-shot pick (max ZiCo)")):
            r = add("v2", ds, nm, per_image(ds, "qenas"), per_image(ds, m))
            forest.append((ds, nm.replace("Selection vs ", "vs "), r))
    # Holm within each question and dataset: the loss-weight test alone, the two selection tests together
    # (the families of the study's pre-planned analysis, qenas/study/analysis.py T4 and T6)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r[0], r[1], r[2].startswith("Loss-weight")), []).append(i)
    for idx in groups.values():
        for i, pa in zip(idx, holm([rows[j][7] for j in idx])):
            rows[i] = rows[i][:7] + [fmt_p(rows[i][7]), fmt_p(pa)] + rows[i][8:]
    header = ["Study", "Dataset", "Ablation (QENAS − variant)", "Dice QENAS (%)", "Dice variant (%)", "Δ Dice (pp)",
              "95 % CI (pp)", "p (Wilcoxon)", "p (Holm)", "Effect size r"]
    write_csv("ablations_v2", header, rows)
    # preliminary study v1 (QENAS 2.1): objective ablations and the other answered questions
    v1 = []
    for r in load(OLD / "analysis" / "T4_significance_ablations.json"):
        v1.append([r["Dataset"], r["Comparison"].replace("QENAS vs ", "QENAS 2.1 vs "), f"{r['Dice QENAS (%)']:.2f}",
                   f"{r['Dice other (%)']:.2f}", f"{r['Δ Dice (pp)']:+.2f}", r["95 % CI (pp)"], fmt_p(r["p (Holm)"])])
    write_csv("ablations_v1_preliminary", ["Dataset", "Comparison", "Dice ref (%)", "Dice variant (%)", "Δ (pp)",
                                           "95 % CI (pp)", "p (Holm)"], v1)
    return {"v2": {"header": header, "rows": rows}, "v1": {"header": ["Dataset", "Comparison", "Dice ref (%)",
                                                                       "Dice variant (%)", "Δ (pp)", "95 % CI (pp)",
                                                                       "p (Holm)"], "rows": v1}}, forest


# ------------------------------------------------------------------------------------------- cost
def cost():
    rows = []
    for ds in DATASETS:
        for m in MAIN:
            tot = total_minutes(ds, m)
            if m == "qenas":
                parts = {k: stage_minutes(ds, m, k) for k in ("search", "candidates", "loss_weights", "final")}
                rows.append([DSL[ds], LABEL[m], mean_sd(parts["search"], 1, 1), mean_sd(parts["candidates"], 1, 1),
                             mean_sd(parts["loss_weights"], 1, 1), mean_sd(parts["final"], 1, 1), mean_sd(tot, 1, 1)])
            elif m == "nnunet":
                rows.append([DSL[ds], LABEL[m], "–", "–", "–", mean_sd(stage_minutes(ds, m, "nnunet_train"), 1, 1),
                             mean_sd(tot, 1, 1)])
            else:
                rows.append([DSL[ds], LABEL[m], "–", "–", mean_sd(stage_minutes(ds, m, "loss_weights"), 1, 1),
                             mean_sd(stage_minutes(ds, m, "final"), 1, 1), mean_sd(tot, 1, 1)])
    header = ["Dataset", "Model", "Search (min)", "Successive halving (min)", "α grid (min)", "Final training (min)",
              "Total, start to end (min)"]
    write_csv("cost_time_per_run", header, rows)
    q = np.median(stage_minutes("IDRID", "qenas", "search"))
    qt = np.median(total_minutes("IDRID", "qenas"))
    srows = [[k, "[MG] Table 8", f"{v:.1f}", f"{v * 60:.0f}"] for k, v in MG_SEARCH_H.items()]
    srows += [["QENAS search (ours)", "this study, Tesla T4", f"{q / 60:.2f}", f"{q:.1f}"],
              ["QENAS complete run incl. final training (ours)", "this study, Tesla T4", f"{qt / 60:.2f}", f"{qt:.1f}"]]
    write_csv("cost_search_time_idrid", ["Method", "Source", "Hours", "Minutes"], srows)
    gpu_h = sum(load(p)["total_seconds"] for p in RES.glob("*/*/seed_*/state.json")) / 3600
    return {"per_run": {"header": header, "rows": rows},
            "search": {"header": ["Method", "Source", "Hours", "Minutes"], "rows": srows}}, {"study_gpu_hours": gpu_h,
                                                                                            "q_search_idrid": q,
                                                                                            "q_total_idrid": qt}


# ------------------------------------------------------------------------------------------- time of all models
JHA_FPS = {"U-Net [JHA]": 11.01, "ResUNet": 14.82, "ResUNet++": 7.01, "FCN8 (VGG16)": 24.91, "HRNet": 11.69,
           "DoubleUNet (VGG19)": 7.46, "PSPNet (ResNet50)": 16.80, "DeepLabv3+ (ResNet50)": 27.90,
           "DeepLabv3+ (ResNet101)": 16.75, "U-Net (ResNet34)": 35.00, "ColonSegNet": 182.38}   # Jha et al. 2021, Table V


def runs_all(m, key):
    """Values of all 12 runs (4 datasets x 3 seeds)."""
    out = []
    for ds in DATASETS:
        for s in SEEDS:
            st = load(exp(ds, m, s) / "state.json")
            if key == "total":
                out.append(st["total_seconds"] / 60)
            elif key == "inference":
                out.append(load(exp(ds, m, s) / "final_metrics.json")["inference_ms_per_image"])
            else:
                out.append(st["stages"].get(key, {}).get("seconds", 0.0) / 60)
    return out


MG_TABLE8 = [("NASUnet", "Gradient", "3 h"), ("HCT-net", "Gradient", "3 h"), ("UNAS-Net", "Evolution", "19 h"),
             ("Genetic U-Net", "Evolution", "4 h"), ("MedUNAS", "Evolution", "11 h"), ("BiX-NAS", "Evolution", "4 h"),
             ("Mixed-GGNAS (With ViT)", "Evolution + Gradient", "(21 + 5) h"),
             ("Mixed-GGNAS (Without ViT)", "Evolution + Gradient", "(13 + 2) h")]          # Hu et al. 2025, Table 8


def qenas_search_hours(ds):
    """Total time of the QENAS algorithm that delivers the architecture and its loss weights (mean of 3 seeds):
    (training-free NSGA-II search + successive-halving selection + loss-weight selection), in hours."""
    parts = [np.mean(stage_minutes(ds, "qenas", k)) / 60 for k in ("search", "candidates", "loss_weights")]
    return parts, sum(parts)


def time_tables():
    parts, tot = qenas_search_hours("IDRID")
    t8 = [[n, st, t] for n, st, t in MG_TABLE8]
    t8.append(["QENAS (ours)", "Training-free evolution + successive halving",
               f"({parts[0]:.2f} + {parts[1]:.2f} + {parts[2]:.2f}) h = {tot:.2f} h ({60 * tot:.0f} min)"])
    h8 = ["Approach", "Strategy", "Search Time"]
    write_csv("time_search_idrid_table8_format", h8, t8)
    per = []
    for ds in DATASETS:
        pp, tt = qenas_search_hours(ds)
        per.append(f"({pp[0]:.2f} + {pp[1]:.2f} + {pp[2]:.2f}) h = {tt:.2f} h")
    per_rows = [["QENAS (ours)", "Training-free evolution + successive halving"] + per]
    hp = ["Approach", "Strategy"] + [DSL[d] for d in DATASETS]
    write_csv("time_search_qenas_all_datasets", hp, per_rows)
    tot_rows = []
    for m in MAIN:
        tot_rows.append([LABEL[m], {"qenas": "search + selection + final training", "nnunet": "training (no search)",
                                    "attention_unet": "training (no search)", "unet": "training (no search)"}[m]]
                        + [f"{np.mean(total_minutes(ds, m)) / 60:.2f} h" for ds in DATASETS])
    write_csv("time_total_per_algorithm", hp, tot_rows)
    return {"table8": {"header": h8, "rows": t8}, "search_all": {"header": hp, "rows": per_rows},
            "total": {"header": hp, "rows": tot_rows}, "qenas_search_idrid_h": tot,
            "qenas_search_parts_idrid_h": parts}


def fig_average_time():
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    stages = {"qenas": [("search", "search", SERIES[0]), ("candidates", "successive halving", "#86b6ef"),
                        ("loss_weights", "α grid", "#c9ddf6"), ("final", "final training", "#184f95")],
              "nnunet": [("nnunet_train", "nnU-Net training", COLOR["nnunet"])],
              "attention_unet": [("loss_weights", "α grid", "#9fdcc4"), ("final", "final training", COLOR["attention_unet"])],
              "unet": [("loss_weights", "α grid", "#f6d68a"), ("final", "final training", COLOR["unet"])]}
    for i, m in enumerate(MAIN):
        left = 0.0
        for k, nm, c in stages[m]:
            v = np.mean(runs_all(m, k))
            ax.barh(i, v, left=left, color=c, height=0.6, label=nm if m in ("qenas", "nnunet") else None)
            left += v
        rest = np.mean(runs_all(m, "total")) - left
        ax.barh(i, rest, left=left, color=GRID, height=0.6, label="data, test, report" if m == "qenas" else None)
        ax.text(left + rest + 3, i, f"{left + rest:.0f} min", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(range(4), [LABEL[m] for m in MAIN])
    ax.invert_yaxis()
    ax.set_xlabel("GPU minutes per run (Tesla T4; mean of 12 runs = 4 datasets × 3 seeds)")
    ax.set_title("Average compute time per run, by stage")
    ax.legend(ncol=3, loc="lower right", fontsize=7.5)
    save(fig, FIG / "fig18_average_time_per_model")


# ------------------------------------------------------------------------------------------- architectures
def architectures():
    rows, counts = [], np.zeros((8, 5))
    for ds in DATASETS:
        for s in SEEDS:
            f = load(exp(ds, "qenas", s) / "final_metrics.json")
            ch = f["architecture"]["chromosome"]
            for i, g in enumerate(ch):
                counts[i, g] += 1
            rows.append([DSL[ds], s, " ".join(BLOCK[g][0] if g != 4 else "N" for g in ch[:4]),
                         " ".join(BLOCK[g][0] if g != 4 else "N" for g in ch[4:]), f"{f['params_m']:.2f}",
                         f"{f['gflops_per_image']:.2f}", f"{f['zico']:.1f}", f"{f['alpha']:.1f}", f["best_epoch"],
                         f["epochs_run"], f"{100 * f['test']['dice']:.2f}"])
    header = ["Dataset", "Seed", "Encoder g1–g4", "Decoder g5–g8", "Params (M)", "GFLOPs/img", "ZiCo", "α",
              "Best epoch", "Epochs run", "Test Dice (%)"]
    write_csv("qenas_architectures", header, rows)
    return {"header": header, "rows": rows}, counts


# ------------------------------------------------------------------------------------------- figures
def fig_pipeline():
    apply_style()
    fig, ax = plt.subplots(figsize=(14.5, 3.6))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    steps = [("Dataset\nleakage-free\nsplit", "all"), ("Search space\n8 genes ×\n5 blocks\n5⁸ = 390,625", "free"),
             ("NSGA-II\npop 20 × 10 gen\nf1 = params\nf2 = 1/(1+ZiCo)", "free"),
             ("16 candidates\nfinal front +\nNSGA-II order", "free"),
             ("Successive\nhalving\n16×5 → 8×10\n→ 4×20 epochs", "sup"),
             ("Selection\nmean val Dice\n(last 5 epochs)\n+ gates G1–G5", "sup"),
             ("Loss weights\nα grid on\nvalidation", "sup"), ("Final training\n120 epochs\nbest val ckpt", "sup"),
             ("Single test\nDice, mIoU,\nHD95, Sens.,\nParams, FLOPs", "all")]
    col = {"free": "#e3eefb", "sup": "#fdebe1", "all": "#eeeeea"}
    w, gap = 1.42, 0.17
    for i, (t, kind) in enumerate(steps):
        x = 0.15 + i * (w + gap)
        ax.add_patch(plt.Rectangle((x, 1.0), w, 1.75, facecolor=col[kind], edgecolor=MUTED, linewidth=0.8))
        ax.text(x + w / 2, 1.875, t, ha="center", va="center", fontsize=8, color=INK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.01, 1.875), xytext=(x + w + 0.01, 1.875),
                        arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.0))
    x0, x1 = 0.15 + 1 * (w + gap), 0.15 + 3 * (w + gap) + w
    ax.text((x0 + x1) / 2, 3.05, "training-free search (no weight is updated)", ha="center", fontsize=9,
            color=SERIES[0], weight="semibold")
    ax.plot([x0, x1], [2.88, 2.88], color=SERIES[0], lw=2)
    x0, x1 = 0.15 + 4 * (w + gap), 0.15 + 7 * (w + gap) + w
    ax.text((x0 + x1) / 2, 3.05, "supervised selection and training (train + validation only)", ha="center",
            fontsize=9, color=SERIES[1], weight="semibold")
    ax.plot([x0, x1], [2.88, 2.88], color=SERIES[1], lw=2)
    ax.text(7.25, 0.45, "The test split is loaded once, after every choice is fixed.  Budget of the candidate stage: "
            "16·5 + 8·5 + 4·10 = 160 epochs.", ha="center", fontsize=8.5, color=INK_2)
    save(fig, FIG / "fig01_qenas_pipeline")


def fig_search_space():
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 4.8)
    ax.axis("off")
    enc = [(0.6, 3.9, "Stem\nC = 32"), (1.8, 3.1, "g1\nE1, 64"), (3.0, 2.3, "g2\nE2, 128"), (4.2, 1.5, "g3\nE3, 256"),
           (5.4, 0.7, "g4\nE4, 512\nbottleneck")]
    dec = [(6.6, 1.5, "g5\nD1, 256"), (7.8, 2.3, "g6\nD2, 128"), (9.0, 3.1, "g7\nD3, 64")]
    last = (9.0, 3.9, "g8\nD4, 32")
    boxes = enc + dec + [last]
    for x, y, t in boxes:
        gene = t.startswith("g")
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.33), 1.0, 0.66, facecolor="#e3eefb" if gene else "#eeeeea",
                                   edgecolor=MUTED, lw=0.8))
        ax.text(x, y, t, ha="center", va="center", fontsize=7.5)
    path = [enc[i][:2] for i in range(5)] + [d[:2] for d in dec]
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        ax.annotate("", xy=(x1 - 0.5 if x1 > x0 else x1, y1 + (0.33 if y1 < y0 else -0.33)),
                    xytext=(x0 + 0.5 if x1 > x0 else x0, y0 - (0.33 if y1 < y0 else -0.33)),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.9))
    ax.annotate("", xy=(last[0], last[1] - 0.33), xytext=(dec[-1][0], dec[-1][1] + 0.33),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=0.9))
    for (ex, ey, _), (dx, dy, _) in zip([enc[3], enc[2], enc[1], enc[0]], [dec[0], dec[1], dec[2], last]):
        ax.annotate("", xy=(dx - 0.5, dy), xytext=(ex + 0.5, ey), arrowprops=dict(arrowstyle="-|>", color=SERIES[0],
                                                                                   lw=0.8, linestyle="--"))
    ax.text(5.0, 4.45, "Skip connections with spatial-channel attention (dashed)", ha="center", fontsize=8.5,
            color=SERIES[0])
    for i, (k, nm) in enumerate(BLOCK.items()):
        ax.add_patch(plt.Rectangle((0.2 + i * 1.95, -0.52), 0.25, 0.22, facecolor=BLOCK_COLORS[k], edgecolor="none"))
        ax.text(0.52 + i * 1.95, -0.41, f"{k}: {nm}" + (" (NAS-Unet cell)" if k == 4 else ""), fontsize=8,
                va="center")
    ax.text(5.0, -0.05, "Every gene gᵢ ∈ {0, …, 4} chooses the block of its position (5⁸ = 390,625 architectures)",
            ha="center", fontsize=8.5, color=INK_2)
    save(fig, FIG / "fig02_search_space")


def fig_violin():
    apply_style()
    fig, axs = plt.subplots(1, 5, figsize=(18, 4.2))
    for ax, c in zip(axs, COLS):
        data = [list((per_image_ext(m) if c == "EXT" else per_image(c, m)).values()) for m in MAIN]
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for b, m in zip(parts["bodies"], MAIN):
            b.set_facecolor(COLOR[m])
            b.set_alpha(0.35)
            b.set_edgecolor("none")
        ax.boxplot(data, widths=0.18, showfliers=False, medianprops=dict(color=INK), whiskerprops=dict(color=INK_2),
                   capprops=dict(color=INK_2), boxprops=dict(color=INK_2))
        ax.scatter(range(1, 5), [np.mean(d) for d in data], marker="D", color=INK, s=14, zorder=4, label="mean")
        ax.set_xticks(range(1, 5), [LABEL[m].replace(" (ours)", "") for m in MAIN], rotation=25)
        ax.set_title(DSL[c] + (" (external)" if c == "EXT" else ""))
        ax.set_ylabel("Per-image Dice (mean of 3 seeds)")
        ax.set_ylim(-0.02, 1.02)
    axs[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, FIG / "fig05_per_image_dice_distribution")


def fig_accuracy_cost(kind):
    apply_style()
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, ds in zip(axs, DATASETS):
        for m in MAIN:
            fs = finals(ds, m)
            x = np.mean([f["params_m"] if kind == "params" else f["gflops_per_image"] for f in fs])
            y = [100 * f["test"]["dice"] for f in fs]
            ax.errorbar([x], [np.mean(y)], yerr=[np.std(y, ddof=1)], fmt="o", ms=8, color=COLOR[m], capsize=3,
                        label=LABEL[m], zorder=5)
        for name, (strat, per) in MG.items():
            if ds in per:
                v = per[ds]
                x = v[2] if kind == "params" else v[3] / (4 if ds == "IDRID" else 8)
                ax.scatter([x], [v[0]], marker="o", s=30, facecolor="none", edgecolor=MUTED, zorder=3)
                ax.annotate(name, (x, v[0]), xytext=(3, 3), textcoords="offset points", fontsize=6.3, color=MUTED)
        if ds == "KVASIR" and kind == "params":
            for name, v in JHA.items():
                ax.scatter([v[6] / 1e6], [100 * v[0]], marker="s", s=24, facecolor="none", edgecolor=MUTED, zorder=3)
                ax.annotate(name.replace(" [JHA]", ""), (v[6] / 1e6, 100 * v[0]), xytext=(3, 3),
                            textcoords="offset points", fontsize=6.3, color=MUTED)
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (M, log scale)" if kind == "params" else "GFLOPs per image (log scale)")
        ax.set_ylabel("Test Dice (%)")
        ax.set_title(DSL[ds])
    h, l = axs[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=MUTED))
    l.append("published methods † ‡ (other splits)")
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, FIG / ("fig06_dice_vs_parameters" if kind == "params" else "fig07_dice_vs_flops"))


def fig_heatmaps(ext):
    apply_style()
    keys = [("dice", "Dice", False), ("miou", "mIoU", False), ("iou", "IoU", False), ("hd95", "HD95", True),
            ("sensitivity", "Sens.", False), ("precision", "Prec.", False), ("specificity", "Spec.", False),
            ("params_m", "Params", True), ("gflops_per_image", "GFLOPs", True), ("inference_ms_per_image", "ms", True)]
    fig, axs = plt.subplots(2, 2, figsize=(15, 6.4))
    for ax, ds in zip(axs.flat, DATASETS):
        V = np.zeros((len(MAIN), len(keys)))
        for i, m in enumerate(MAIN):
            fs = finals(ds, m)
            for j, (k, _, _) in enumerate(keys):
                if k in ("precision", "specificity"):
                    V[i, j] = np.mean([ext[(ds, m, s)][k] for s in SEEDS])
                elif k in ("params_m", "gflops_per_image", "inference_ms_per_image"):
                    V[i, j] = np.mean([f[k] for f in fs])
                else:
                    V[i, j] = np.mean([f["test"][k] for f in fs])
        S = np.zeros_like(V)
        for j, (_, _, lower) in enumerate(keys):
            col = V[:, j]
            rng = col.max() - col.min()
            S[:, j] = 0.5 if rng == 0 else ((col.max() - col) if lower else (col - col.min())) / rng
        ax.imshow(S, cmap="Blues", vmin=-0.15, vmax=1.15, aspect="auto")
        for i in range(len(MAIN)):
            for j, (k, _, _) in enumerate(keys):
                v = V[i, j]
                txt = f"{100 * v:.1f}" if k in ("dice", "miou", "iou", "sensitivity", "precision", "specificity") \
                    else f"{v:.1f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="white" if S[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(keys)), [k[1] for k in keys], fontsize=8)
        ax.set_yticks(range(len(MAIN)), [LABEL[m] for m in MAIN], fontsize=8)
        ax.grid(False)
        ax.set_title(DSL[ds])
    fig.suptitle("Per dataset: darker = better within the column (HD95, parameters, FLOPs and time: lower is "
                 "better)", fontsize=10, color=INK_2)
    fig.tight_layout()
    save(fig, FIG / "fig08_performance_cost_heatmap")


def fig_search_time(info):
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _, q_alg = qenas_search_hours("IDRID")
    items = [(k, v) for k, v in MG_SEARCH_H.items()] + [("QENAS (search + selection)", q_alg),
                                                         ("QENAS incl. final training", info["q_total_idrid"] / 60)]
    items.sort(key=lambda kv: kv[1])
    cols = [SERIES[0] if k.startswith("QENAS") else MUTED for k, _ in items]
    ax.barh(range(len(items)), [v for _, v in items], color=cols, height=0.62)
    for i, (k, v) in enumerate(items):
        ax.text(v * 1.08, i, f"{v * 60:.0f} min" if v < 1 else (f"{v:.1f} h" if v < 2 else f"{v:.0f} h"),
                va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(items)), [k for k, _ in items])
    ax.set_xscale("log")
    ax.set_xlabel("Hours (log scale)")
    ax.set_title("Search time on IDRiD (other methods: Hu et al. 2025, Table 8, RTX 4070 Ti Super;\nQENAS: "
                 "mean of 3 runs on a Tesla T4, search + successive halving + loss weights)", fontsize=9.5)
    save(fig, FIG / "fig09_search_time")


def fig_compute_per_run():
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    stages = [("search", "QENAS search", SERIES[0]), ("candidates", "successive halving", "#86b6ef"),
              ("loss_weights", "α grid", "#c9ddf6"), ("final", "final training", "#184f95")]
    xs, w = np.arange(len(DATASETS)), 0.2
    for di, ds in enumerate(DATASETS):
        bottom = 0
        for st, nm, c in stages:
            v = np.mean(stage_minutes(ds, "qenas", st))
            ax.bar(di - 1.5 * w, v, w, bottom=bottom, color=c, label=nm if di == 0 else None)
            bottom += v
        for k, m in enumerate(MAIN[1:], 1):
            ax.bar(di - 1.5 * w + k * w, np.mean(total_minutes(ds, m)), w, color=COLOR[m],
                   label=LABEL[m] if di == 0 else None)
    ax.set_xticks(xs, [DSL[d] for d in DATASETS])
    ax.set_ylabel("GPU minutes per run (Tesla T4, mean of 3)")
    ax.set_title("Compute per run: complete QENAS (stacked stages) vs training the baselines")
    ax.legend(ncol=4, loc="upper left")
    save(fig, FIG / "fig10_compute_per_run")


def fig_forest(rows, name, title):
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.4))
    for i, (lab, r, sig) in enumerate(rows):
        y = len(rows) - 1 - i
        c = SERIES[0] if sig and r["diff"] > 0 else (SERIES[1] if sig else MUTED)
        ax.plot([100 * r["lo"], 100 * r["hi"]], [y, y], color=c, lw=2)
        ax.scatter([100 * r["diff"]], [y], color=c, s=30, zorder=3)
        ax.annotate(f"{100 * r['diff']:+.2f}", (100 * r["hi"], y), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK_2)
    ax.axvline(0, color=INK_2, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8)
    ax.set_xlabel("Δ Dice, QENAS − compared (pp; mean and 95 % bootstrap CI over test images)")
    ax.set_title(title, fontsize=10)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=SERIES[0], lw=2), Line2D([], [], color=SERIES[1], lw=2),
               Line2D([], [], color=MUTED, lw=2)], ["QENAS significantly better", "QENAS significantly worse",
                                                    "not significant (Holm)"], loc="upper center", fontsize=7.5,
              bbox_to_anchor=(0.5, -0.09 - 1.0 / (0.42 * len(rows) + 1.4)), ncol=3)
    save(fig, FIG / name)


def fig_external():
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.38
    for i, m in enumerate(MAIN):
        a = [100 * f["test"]["dice"] for f in finals("KVASIR", m)]
        b = [100 * e["dice"] for e in externals(m)]
        ax.bar(i - w / 2, np.mean(a), w, yerr=np.std(a, ddof=1), color=COLOR[m], capsize=3)
        ax.bar(i + w / 2, np.mean(b), w, yerr=np.std(b, ddof=1), color=COLOR[m], alpha=0.45, capsize=3, hatch="//",
               edgecolor="white")
        ax.text(i - w / 2, np.mean(a) + np.std(a, ddof=1) + 0.8, f"{np.mean(a):.1f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, np.mean(b) + np.std(b, ddof=1) + 0.8, f"{np.mean(b):.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(4), [LABEL[m] for m in MAIN])
    ax.set_ylim(50, 95)
    ax.set_ylabel("Dice (%)")
    from matplotlib.patches import Patch
    ax.legend([Patch(color=MUTED), Patch(facecolor=MUTED, alpha=0.45, hatch="//")],
              ["Kvasir-SEG test (in-domain)", "CVC-ClinicDB, 612 frames (external)"], loc="upper right")
    ax.set_title("Generalisation: trained on Kvasir-SEG, applied unchanged to CVC-ClinicDB")
    save(fig, FIG / "fig12_external_validation")


def fig_ranks(ranks):
    apply_style()
    fig, axs = plt.subplots(len(COLS), 1, figsize=(8, 1.15 * len(COLS) + 0.6), sharex=True)
    for ax, c in zip(axs, COLS):
        r = ranks[c]
        for m in MAIN:
            ax.scatter([r["ranks"][m]], [0], color=COLOR[m], s=60, zorder=3, label=LABEL[m])
        best = min(r["ranks"].values())
        ax.plot([best, best + r["cd"]], [0.35, 0.35], color=INK, lw=2)
        ax.text(best + r["cd"] + 0.03, 0.35, f"CD = {r['cd']:.2f}", va="center", fontsize=7.5)
        ax.set_yticks([])
        ax.set_ylim(-0.6, 0.8)
        ax.set_ylabel(DSL[c], rotation=0, ha="right", va="center", fontsize=8.5)
        ax.grid(axis="y", visible=False)
    axs[-1].set_xlabel("Mean rank of per-image Dice (1 = best; Friedman test, Nemenyi critical difference)")
    axs[-1].set_xlim(0.8, 4.2)
    axs[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 2.0), fontsize=8)
    save(fig, FIG / "fig13_mean_ranks")


def fig_blocks(counts):
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    pos = ["g1 E1", "g2 E2", "g3 E3", "g4 E4", "g5 D1", "g6 D2", "g7 D3", "g8 D4"]
    left = np.zeros(8)
    for k, nm in BLOCK.items():
        ax.barh(range(8), counts[:, k], left=left, color=BLOCK_COLORS[k], label=nm, height=0.65)
        left += counts[:, k]
    ax.set_yticks(range(8), pos)
    ax.invert_yaxis()
    ax.set_xlabel("Number of the 12 final QENAS architectures (4 datasets × 3 seeds)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("Block type chosen at each gene position")
    save(fig, FIG / "fig17_selected_blocks")


def copy_run_figures():
    names = ["pareto_front_final_with_candidates", "search_progress", "successive_halving",
             "learning_capability_ranking", "candidates_val_dice", "loss_weight_selection", "final_training_curves",
             "selected_architecture"]
    copied = []
    for ds in DATASETS:
        src = exp(ds, "qenas", SEEDS[0]) / "figures"
        for n in names:
            if (src / f"{n}.png").exists():
                shutil.copy2(src / f"{n}.png", FIG / f"run_{ds}_seed{SEEDS[0]}_{n}.png")
                copied.append(f"run_{ds}_seed{SEEDS[0]}_{n}.png")
    for n in ("search_vs_random", "proxies_vs_val_dice", "ablation_loss_coefficients", "ablation_selection",
              "qualitative_BUSI", "qualitative_CVC", "qualitative_IDRID", "qualitative_KVASIR"):
        p = RES / "analysis" / "figures" / f"{n}.png"
        if p.exists():
            shutil.copy2(p, FIG / f"analysis_{n}.png")
            copied.append(f"analysis_{n}.png")
    return copied


# ------------------------------------------------------------------------------------------- main
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    ext = extended_metrics()
    report = {"sources": SOURCES, "metric_tables": metric_tables(ext)}
    report["statistics"], ranks = statistics()
    report["ablations"], forest = ablations()
    report["cost"], info = cost()
    report["cost"].update(time_tables())
    report["architectures"], counts = architectures()
    report["info"] = info
    report["ranks"] = ranks
    # figures
    fig_pipeline()
    fig_search_space()
    fig_violin()
    fig_accuracy_cost("params")
    fig_accuracy_cost("flops")
    fig_heatmaps(ext)
    fig_search_time(info)
    fig_compute_per_run()
    ab_rows = []
    for (ds, nm, r), row in zip(forest, report["ablations"]["v2"]["rows"]):
        ab_rows.append((f"{DSL[ds]}: {nm}", r, row[8] == "<1e-4" or float(row[8]) < 0.05))
    fig_forest(ab_rows, "fig11_ablations_forest", "Ablations of QENAS 3.0 (positive = the QENAS component helps)")
    base_rows = []
    for row in report["statistics"]["pairwise"]["rows"]:
        c = [k for k, v in DSL.items() if v == row[0]][0]
        m = [k for k in MAIN if f"QENAS vs {LABEL[k]}" == row[1]][0]
        r = paired(per_image_ext("qenas") if c == "EXT" else per_image(c, "qenas"),
                   per_image_ext(m) if c == "EXT" else per_image(c, m))
        base_rows.append((f"{row[0]}: vs {LABEL[m]}", r, row[8] == "<1e-4" or float(row[8]) < 0.05))
    fig_forest(base_rows, "fig04_paired_differences_vs_baselines",
               "QENAS 3.0 vs the baselines on the same test images")
    fig_external()
    v1_rows, v1v2 = [], []
    for ds in DATASETS:
        r = paired(per_image(ds, "qenas"), per_image(ds, "qenas", root=OLD, seeds=[42, 43, 44]))
        v1v2.append(r)
    for ds, r, pa in zip(DATASETS, v1v2, holm([r["p"] for r in v1v2])):
        v1_rows.append((f"{DSL[ds]}: QENAS 3.0 vs QENAS 2.1", r, pa < 0.05))
    fig_forest(v1_rows, "fig16_v2_vs_v1", "Final QENAS 3.0 (study v2) vs preliminary QENAS 2.1 (study v1), same "
               "test images")
    fig_ranks(ranks)
    fig_blocks(counts)
    report["copied_figures"] = copy_run_figures()
    report["figures"] = sorted(p.name for p in FIG.glob("*.png"))
    with open(OUT / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)
    print(f"[paper] tables -> {TAB} | {len(report['figures'])} figures -> {FIG}")


if __name__ == "__main__":
    main()
