"""Analysis of evaluation study v2: paper tables, statistics and figures (results/analysis/).

Works on partial results (every table states how many seeds it contains). Statistics:

* mean ± standard deviation over seeds (sample SD, n - 1);
* paired comparisons on the *same test images*: per-image Dice averaged over the seeds of each method, two-sided
  Wilcoxon signed-rank test, mean difference with a 95 % bootstrap confidence interval (10 000 resamples of the
  images), Holm-Bonferroni correction over the comparisons of one table and dataset;
* search quality: hypervolume of the non-dominated set of *all* architectures each search evaluated (NSGA-II vs
  random search with the same number of evaluations), both normalised by the union of their archives (reference
  point 1.1); NSGA-II's curve is the best-so-far set after every generation.

With ``old_results`` (the preliminary study v1 of QENAS 2.1, folder ``Old Results/study_v1``) the v1 -> v2 comparison
(T14) is added.
Every output file is listed with the research question it answers in results/analysis/README.md.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr, wilcoxon

from ..methods import METHODS
from ..utils.io import load_json, save_json, save_markdown_table
from .plan import (DATASETS, EXTERNAL_METHODS, LOSS_ABLATION, MAIN_METHODS, SELECTION_DATASETS, SELECTION_METHODS)

OBJ = ("f1_params", "f2_zico")
METRICS = [("dice", "Dice / DSC (%)", 100), ("miou", "mIoU (%)", 100), ("hd95", "HD95 (px)", 1),
           ("sensitivity", "Sensitivity (%)", 100), ("iou", "IoU (%)", 100)]
DATASET_LABEL = {"BUSI": "BUSI", "CVC": "CVC-ClinicDB", "IDRID": "IDRiD (optic disc)", "KVASIR": "Kvasir-SEG"}
# Mixed-GGNAS published results (Hu et al., ESWA 2025, Tables 4, 5, 7, 8) - their own splits, for reference only.
MIXED_GGNAS = {"BUSI": {"dice": 80.64, "miou": 82.77, "params_m": 7.86, "gflops": 66.10},
               "CVC": {"dice": 93.51, "miou": 94.72, "params_m": 6.01, "gflops": 88.08, "note": "random frame split"},
               "IDRID": {"dice": 96.63, "miou": 96.78, "params_m": 29.41, "gflops": 335.34,
                         "search_hours": 26.0, "gpu": "RTX 4070 Ti Super"}}
V1_LABEL = {"qenas": "QENAS 2.1 (v1)", "unet": "U-Net (v1 protocol)", "attention_unet": "Attention U-Net (v1 protocol)"}


# ============================================================================================== loading
def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


class Store:
    """Index of all completed experiments: results/<dataset_dir>/<method>/seed_<s>/."""

    def __init__(self, results: Path):
        self.results = Path(results)
        self.exps: Dict[Tuple[str, str, int], Path] = {}
        for st in self.results.glob("*/*/seed_*/state.json"):
            try:
                s = load_json(st)
            except Exception:
                continue
            if s.get("status") == "completed":
                d = st.parent
                self.exps[(d.parent.parent.name, d.parent.name, int(d.name.split("_")[1]))] = d

    def seeds(self, ds: str, method: str) -> List[int]:
        return sorted(s for (d, m, s) in self.exps if d == ds and m == method)

    def path(self, ds: str, method: str, seed: int) -> Optional[Path]:
        return self.exps.get((ds, method, seed))

    def json(self, ds: str, method: str, name: str) -> List[Dict[str, Any]]:
        return [load_json(self.exps[(ds, method, s)] / name) for s in self.seeds(ds, method)
                if (self.exps[(ds, method, s)] / name).exists()]

    def finals(self, ds: str, method: str) -> List[Dict[str, Any]]:
        return self.json(ds, method, "final_metrics.json")

    def per_image(self, ds: str, method: str, file: str = "test_per_image.csv") -> Dict[str, List[Dict]]:
        """sample_id -> list of per-image rows (one per seed)."""
        out: Dict[str, List[Dict]] = {}
        for s in self.seeds(ds, method):
            p = self.exps[(ds, method, s)] / file
            if p.exists():
                for r in _read_csv(p):
                    out.setdefault(r["sample_id"], []).append(r)
        return out


# ============================================================================================== statistics
def mean_sd(values: Sequence[float]) -> Tuple[float, float, int]:
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0), len(v)


def fmt(values: Sequence[float], scale: float = 1.0, digits: int = 2) -> str:
    m, s, n = mean_sd([x * scale for x in values if x is not None])
    if n == 0:
        return "–"
    return f"{m:.{digits}f} ± {s:.{digits}f}" if n > 1 else f"{m:.{digits}f}"


def holm(pvalues: Sequence[float]) -> List[float]:
    idx = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out = [1.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        out[i] = running
    return out


def paired_test(a: Dict[str, List[Dict]], b: Dict[str, List[Dict]], key: str = "dice") -> Optional[Dict[str, Any]]:
    """Per-image values averaged over seeds; paired over the images both methods evaluated."""
    ids = sorted(set(a) & set(b))
    if len(ids) < 5:
        return None
    x = np.array([np.mean([r[key] for r in a[i]]) for i in ids])
    y = np.array([np.mean([r[key] for r in b[i]]) for i in ids])
    d = x - y
    p = 1.0 if np.allclose(d, 0) else float(wilcoxon(x, y).pvalue)
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return {"n_images": len(ids), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "p_wilcoxon": p,
            "a_better_images": int((d > 0).sum()), "b_better_images": int((d < 0).sum())}


def _hv(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    from pymoo.indicators.hv import HV
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    if len(points) == 0:
        return float("nan")
    z = (points - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    nd = NonDominatedSorting().do(z, only_non_dominated_front=True)
    return float(HV(ref_point=np.full(z.shape[1], 1.1))(z[nd]))


def _label(m: str) -> str:
    return METHODS[m]["label"]


def _sig_row(ds: str, name: str, ref_label: str, r: Dict[str, Any], p_adj: float) -> Dict[str, Any]:
    return {"Dataset": DATASET_LABEL.get(ds, ds), "Comparison": name, "Images": r["n_images"],
            f"Dice {ref_label} (%)": 100 * r["mean_a"], "Dice other (%)": 100 * r["mean_b"],
            "Δ Dice (pp)": 100 * r["mean_diff"], "95 % CI (pp)": f"[{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}]",
            "p (Wilcoxon)": r["p_wilcoxon"], "p (Holm)": p_adj,
            "Images better / worse": f"{r['a_better_images']} / {r['b_better_images']}"}


# ============================================================================================== tables
def table_results(st: Store, datasets: Sequence[str], methods: Sequence[str], with_mixed_ggnas: bool,
                  labels: Optional[Dict[str, str]] = None) -> List[Dict]:
    rows = []
    for ds in datasets:
        for m in methods:
            fs = st.finals(ds, m)
            if not fs:
                continue
            row = {"Dataset": DATASET_LABEL.get(ds, ds), "Method": (labels or {}).get(m, _label(m)), "Seeds": len(fs)}
            for key, label, scale in METRICS:
                row[label] = fmt([f["test"][key] for f in fs], scale)
            row["Params (M)"] = fmt([f["params_m"] for f in fs], 1, 2)
            row["FLOPs (G, batch)"] = fmt([f["gflops_mixed_ggnas"] for f in fs], 1, 1)
            row["GFLOPs / image"] = fmt([f["gflops_per_image"] for f in fs], 1, 2)
            row["ms / image"] = fmt([f["inference_ms_per_image"] for f in fs], 1, 1)
            rows.append(row)
        if with_mixed_ggnas and ds in MIXED_GGNAS:
            g = MIXED_GGNAS[ds]
            rows.append({"Dataset": DATASET_LABEL[ds], "Method": "Mixed-GGNAS (published, own split" +
                         (", " + g["note"] if "note" in g else "") + ")", "Seeds": "–",
                         "Dice / DSC (%)": f"{g['dice']:.2f}", "mIoU (%)": f"{g['miou']:.2f}",
                         "Params (M)": f"{g['params_m']:.2f}", "FLOPs (G, batch)": f"{g['gflops']:.1f}"})
    return rows


def table_significance(st: Store, datasets: Sequence[str], ref: str, others: Sequence[str],
                       file: str = "test_per_image.csv") -> List[Dict]:
    rows = []
    for ds in datasets:
        a = st.per_image(ds, ref, file)
        tests = [(m, r) for m in others for r in [paired_test(a, st.per_image(ds, m, file))] if r is not None]
        for (m, r), p_adj in zip(tests, holm([r["p_wilcoxon"] for _, r in tests])):
            rows.append(_sig_row(ds, f"{_label(ref)} vs {_label(m)}", _label(ref), r, p_adj))
    return rows


def loss_coefficient_table(st: Store) -> Tuple[List[Dict], List[Dict]]:
    """RQ2: alpha grid vs fixed alpha = 0.5 on the same architecture (paired by seed)."""
    rows, per_run = [], []
    for ds in DATASETS:
        seeds = sorted(set(st.seeds(ds, "qenas")) & set(st.seeds(ds, LOSS_ABLATION)))
        if not seeds:
            continue
        chosen, gain_val, d_test, grid_min, share = [], [], [], [], []
        for s in seeds:
            q, f = st.path(ds, "qenas", s), st.path(ds, LOSS_ABLATION, s)
            lw = load_json(q / "loss_weights.json")
            crit = {r["alpha"]: r["criterion"] for r in lw["runs"] if r.get("criterion") is not None}
            tq, tf = (load_json(p / "final_metrics.json")["test"]["dice"] for p in (q, f))
            state = load_json(q / "state.json")
            gm = state["stages"]["loss_weights"]["seconds"] / 60
            tot = state.get("total_seconds", sum(v.get("seconds", 0) for v in state["stages"].values())) / 60
            chosen.append(lw["alpha"])
            gain_val.append(crit[lw["alpha"]] - crit.get(0.5, crit[lw["alpha"]]))
            d_test.append(tq - tf)
            grid_min.append(gm)
            share.append(gm / tot if tot else float("nan"))
            per_run.append({"dataset": ds, "seed": s, "alpha": lw["alpha"], "val_gain": gain_val[-1],
                            "test_dice_grid": tq, "test_dice_fixed": tf, "grid_minutes": gm, "total_minutes": tot})
        rows.append({"Dataset": DATASET_LABEL[ds], "Seeds": len(seeds), "α chosen": ", ".join(f"{a:.1f}" for a in chosen),
                     "Val gain of the grid (pp)": fmt(gain_val, 100), "Test Dice, grid (%)":
                         fmt([r["test_dice_grid"] for r in per_run if r["dataset"] == ds], 100),
                     "Test Dice, α = 0.5 (%)": fmt([r["test_dice_fixed"] for r in per_run if r["dataset"] == ds], 100),
                     "Test Δ grid − fixed (pp)": fmt(d_test, 100), "Grid time (min)": fmt(grid_min, 1, 1),
                     "Grid share of run time (%)": fmt(share, 100, 1)})
    return rows, per_run


def search_efficiency(st: Store) -> Tuple[List[Dict], Dict]:
    rows, curves = [], {}
    for ds in DATASETS:
        hv_n, hv_r = [], []
        n_eval = 0
        for s in sorted(set(st.seeds(ds, "qenas")) & set(st.seeds(ds, "random_search"))):
            qa = load_json(st.path(ds, "qenas", s) / "search_results.json")
            ra = load_json(st.path(ds, "random_search", s) / "search_results.json")
            A = np.array([[r[o] for o in OBJ] for r in qa + ra], float)
            lo, hi = A.min(0), A.max(0)
            gens = sorted({r["first_generation"] for r in qa})
            curve = [_hv(np.array([[r[o] for o in OBJ] for r in qa if r["first_generation"] <= g]), lo, hi)
                     for g in gens]
            hr = _hv(np.array([[r[o] for o in OBJ] for r in ra]), lo, hi)
            hv_n.append(curve[-1])
            hv_r.append(hr)
            n_eval = len(qa)
            curves.setdefault(ds, []).append({"seed": s, "nsga2": curve, "random": hr, "generations": gens})
        if hv_n:
            rows.append({"Dataset": DATASET_LABEL[ds], "Seeds": len(hv_n), "Evaluations": n_eval,
                         "HV NSGA-II": fmt(hv_n, 1, 4), "HV random search": fmt(hv_r, 1, 4),
                         "NSGA-II better in seeds": f"{sum(a > b for a, b in zip(hv_n, hv_r))}/{len(hv_n)}"})
    return rows, curves


def successive_halving_table(st: Store) -> Tuple[List[Dict], List[Dict], Dict]:
    """How consistent are the rungs, and what would the earlier rules have chosen? Plus proxy validity: Spearman
    rho between the objectives and the 5-epoch criterion of all 16 candidates."""
    sh_rows, proxy_rows, pts = [], [], {}
    for ds in DATASETS:
        t12, t23, rank1, lead1, delta_same, n = [], [], [], 0, 0, 0
        pooled = []
        for s in st.seeds(ds, "qenas"):
            sel = load_json(st.path(ds, "qenas", s) / "selection.json")
            sh = load_json(st.path(ds, "qenas", s) / "successive_halving.json")
            an = {a["candidate_id"]: a for a in sel["analysis"]}
            r = [{int(k): v for k, v in x["scores"].items()} for x in sh["rungs"]]
            if len(r) < 2:
                continue
            n += 1
            for i, taus in ((0, t12), (1, t23)):
                if i + 1 < len(r):
                    ids = sorted(r[i + 1])
                    if len(ids) >= 3:
                        taus.append(float(kendalltau([r[i][c] for c in ids], [r[i + 1][c] for c in ids]).statistic))
            order1 = sorted(r[0], key=lambda c: -r[0][c])
            rank1.append(1 + order1.index(sel["selected_id"]))
            lead1 += order1[0] == sel["selected_id"]
            fin = [a for a in an.values() if a.get("finalist")]
            delta_same += max(fin, key=lambda a: a["delta_dice"])["candidate_id"] == sel["selected_id"]
            first = str(sh["rung_epochs"][0])
            pooled += [dict(a, seed=s, score=a["rung_scores"][first]) for a in an.values() if first in a["rung_scores"]]
        if not n:
            continue
        sh_rows.append({"Dataset": DATASET_LABEL[ds], "Runs": n,
                        "Kendall τ rung 1 vs rung 2 (8 candidates)": fmt(t12, 1, 2),
                        "Kendall τ rung 2 vs rung 3 (4 candidates)": fmt(t23, 1, 2),
                        "Rank of the selected candidate after 5 epochs (of 16)": ", ".join(map(str, rank1)),
                        "5-epoch leader selected": f"{lead1}/{n}",
                        "ΔDice rule would choose the same": f"{delta_same}/{n}"})
        pts[ds] = pooled
        row = {"Dataset": DATASET_LABEL[ds], "Candidates": len(pooled), "Runs": n}
        for k, name in (("params", "params"), ("zico", "ZiCo")):
            res = spearmanr([a[k] for a in pooled], [a["score"] for a in pooled])
            per = []
            for s in sorted({a["seed"] for a in pooled}):
                q = [a for a in pooled if a["seed"] == s]
                if len(q) >= 3:
                    per.append(float(spearmanr([a[k] for a in q], [a["score"] for a in q]).statistic))
            row[f"ρ {name} (per run)"] = fmt(per, 1, 2)
            row[f"ρ {name} (pooled), p"] = f"{res.statistic:+.2f}, {res.pvalue:.2g}"
        proxy_rows.append(row)
    return sh_rows, proxy_rows, pts


def training_tables(st: Store) -> Tuple[List[Dict], List[Dict]]:
    train, timing = [], []
    for ds in DATASETS:
        for m in METHODS:
            seeds = st.seeds(ds, m)
            fv = st.json(ds, m, "final_validation.json")
            if fv:
                train.append({"Dataset": DATASET_LABEL[ds], "Method": _label(m), "Runs": len(fv),
                              "Best epoch": fmt([f["best_epoch"] for f in fv], 1, 1),
                              "Epochs run": fmt([f["epochs_run"] for f in fv], 1, 1),
                              "Stopped early (overfitting)": f"{sum(bool(f['stopped_early']) for f in fv)}/{len(fv)}",
                              "Train Dice (%)": fmt([(f["train_metrics_best_checkpoint"] or {}).get("dice")
                                                     for f in fv], 100),
                              "Val Dice (%)": fmt([f["val_metrics_best_checkpoint"]["dice"] for f in fv], 100)})
            states = [load_json(st.path(ds, m, s) / "state.json") for s in seeds]
            if states:
                row = {"Dataset": DATASET_LABEL[ds], "Method": _label(m), "Runs": len(states)}
                for stage in ("search", "candidates", "loss_weights", "final", "nnunet_train"):
                    vals = [x["stages"][stage]["seconds"] / 60 for x in states if stage in x["stages"]]
                    if vals:
                        row[f"{stage} (min)"] = fmt(vals, 1, 1)
                row["total (min)"] = fmt([x.get("total_seconds", sum(v.get("seconds", 0) for v in x["stages"].values()))
                                          / 60 for x in states], 1, 1)
                env = st.path(ds, m, seeds[0]) / "environment.json"
                row["GPU"] = load_json(env).get("gpu_name", "cpu") if env.exists() else "?"
                timing.append(row)
    return train, timing


def external_table(st: Store) -> Tuple[List[Dict], List[Dict]]:
    rows = []
    for m in EXTERNAL_METHODS:
        res = st.json("KVASIR", m, "external_CVC/external_metrics.json")
        if res:
            rows.append({"Train → test": "Kvasir-SEG → CVC-ClinicDB (all 612 frames)", "Method": _label(m),
                         "Seeds": len(res), **{label: fmt([r["metrics"][k] for r in res], sc) for k, label, sc in METRICS}})
    sig = table_significance(st, ["KVASIR"], "qenas", [m for m in EXTERNAL_METHODS if m != "qenas"],
                             file="external_CVC/external_per_image.csv")
    for r in sig:
        r["Dataset"] = "Kvasir-SEG → CVC-ClinicDB"
    return rows, sig


def nnunet_tta_table(st: Store) -> List[Dict]:
    rows = []
    for ds in DATASETS:
        fs = st.finals(ds, "nnunet")
        if fs:
            rows.append({"Dataset": DATASET_LABEL[ds], "Seeds": len(fs),
                         "Dice, no TTA (%) — main tables": fmt([f["test"]["dice"] for f in fs], 100),
                         "Dice, nnU-Net mirroring TTA (%)": fmt([f["test_with_nnunet_mirroring_tta"]["dice"] for f in fs], 100),
                         "HD95, no TTA (px)": fmt([f["test"]["hd95"] for f in fs]),
                         "HD95, mirroring TTA (px)": fmt([f["test_with_nnunet_mirroring_tta"]["hd95"] for f in fs])})
    return rows


def v1_v2_tables(st: Store, v1: Store) -> Tuple[List[Dict], List[Dict]]:
    """The same method family before (study v1, seeds 42-44) and after (study v2, seeds 45-47) the v2 changes,
    on the identical test images (unpaired seeds; per-image Dice averaged over each study's seeds)."""
    rows, sig = [], []
    for ds in DATASETS:
        tests = []
        for m in ("qenas", "unet", "attention_unet"):
            f1, f2 = v1.finals(ds, m), st.finals(ds, m)
            if not (f1 and f2):
                continue
            t1 = [load_json(v1.path(ds, m, s) / "state.json").get("total_seconds", 0) / 60 for s in v1.seeds(ds, m)]
            t2 = [load_json(st.path(ds, m, s) / "state.json").get("total_seconds", 0) / 60 for s in st.seeds(ds, m)]
            rows.append({"Dataset": DATASET_LABEL[ds], "Method": _label(m), "Seeds v1 / v2": f"{len(f1)} / {len(f2)}",
                         "Dice v1 (%)": fmt([f["test"]["dice"] for f in f1], 100),
                         "Dice v2 (%)": fmt([f["test"]["dice"] for f in f2], 100),
                         "HD95 v1 (px)": fmt([f["test"]["hd95"] for f in f1]),
                         "HD95 v2 (px)": fmt([f["test"]["hd95"] for f in f2]),
                         "Params v1 (M)": fmt([f["params_m"] for f in f1]), "Params v2 (M)": fmt([f["params_m"] for f in f2]),
                         "Time v1 (min)": fmt(t1, 1, 1), "Time v2 (min)": fmt(t2, 1, 1)})
            r = paired_test(st.per_image(ds, m), v1.per_image(ds, m))
            if r is not None:
                tests.append((m, r))
        for (m, r), p_adj in zip(tests, holm([r["p_wilcoxon"] for _, r in tests])):
            sig.append(_sig_row(ds, f"{_label(m)}: v2 vs v1", "v2", r, p_adj))
    return rows, sig


# ============================================================================================== figures
def figures(st: Store, out: Path, curves: Dict, proxy_pts: Dict, data_root: Optional[str]) -> List[str]:
    from ..visualization.style import MUTED, SERIES, apply_style, plt, save
    made = []
    apply_style()
    dsl = [ds for ds in DATASETS if any(st.seeds(ds, m) for m in MAIN_METHODS)]
    if dsl:   # per-image Dice boxplots
        fig, axs = plt.subplots(1, len(dsl), figsize=(4.6 * len(dsl), 4.2), squeeze=False)
        for ax, ds in zip(axs[0], dsl):
            data, labels = [], []
            for m in MAIN_METHODS:
                pi = st.per_image(ds, m)
                if pi:
                    data.append([np.mean([r["dice"] for r in v]) for v in pi.values()])
                    labels.append(_label(m))
            ax.boxplot(data, showmeans=True, widths=0.6)
            ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20)
            ax.set_title(DATASET_LABEL[ds])
            ax.set_ylabel("Per-image test Dice (mean over seeds)")
        fig.tight_layout()
        save(fig, out / "per_image_dice_boxplots")
        made.append("per_image_dice_boxplots.png")
        fig, axs = plt.subplots(1, len(dsl), figsize=(4.4 * len(dsl), 3.8), squeeze=False)   # accuracy vs size
        for ax, ds in zip(axs[0], dsl):
            for c, m in enumerate(MAIN_METHODS):
                fs = st.finals(ds, m)
                if fs:
                    ax.errorbar([np.mean([f["params_m"] for f in fs])], [100 * np.mean([f["test"]["dice"] for f in fs])],
                                yerr=[100 * mean_sd([f["test"]["dice"] for f in fs])[1]], fmt="o", color=SERIES[c],
                                label=_label(m), capsize=3)
            if ds in MIXED_GGNAS:
                ax.scatter([MIXED_GGNAS[ds]["params_m"]], [MIXED_GGNAS[ds]["dice"]], marker="x", color=MUTED,
                           label="Mixed-GGNAS (published, own split)")
            ax.set_xscale("log")
            ax.set_xlabel("Parameters (M, log)")
            ax.set_ylabel("Test Dice (%)")
            ax.set_title(DATASET_LABEL[ds])
            ax.legend(loc="lower right", fontsize=7.5)
        fig.tight_layout()
        save(fig, out / "accuracy_vs_parameters")
        made.append("accuracy_vs_parameters.png")
    if curves:   # search progress vs random search
        fig, axs = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.8), squeeze=False)
        for ax, (ds, cs) in zip(axs[0], curves.items()):
            G = np.array([c["nsga2"] for c in cs])
            g = cs[0]["generations"]
            ax.plot(g, G.mean(0), color=SERIES[0], marker="o", markersize=3, label="NSGA-II (mean over seeds)")
            ax.fill_between(g, G.min(0), G.max(0), color=SERIES[0], alpha=0.15, linewidth=0)
            r = np.array([c["random"] for c in cs])
            ax.axhspan(r.min(), r.max(), color=SERIES[1], alpha=0.25, linewidth=0)
            ax.axhline(r.mean(), color=SERIES[1], label="random search, same budget")
            ax.set_title(DATASET_LABEL[ds])
            ax.set_xlabel("Generation")
            ax.set_ylabel("Hypervolume (union-normalised)")
            ax.legend(loc="lower right")
        fig.tight_layout()
        save(fig, out / "search_vs_random")
        made.append("search_vs_random.png")
    if proxy_pts:   # objectives vs 5-epoch validation Dice of all candidates
        fig, axs = plt.subplots(2, len(proxy_pts), figsize=(4.0 * len(proxy_pts), 7.0), squeeze=False)
        for j, (ds, pts) in enumerate(proxy_pts.items()):
            for i, (k, name, sc) in enumerate((("zico", "ZiCo", 1.0), ("params", "Parameters (M)", 1e-6))):
                ax = axs[i, j]
                for c, s in enumerate(sorted({p["seed"] for p in pts})):
                    q = [p for p in pts if p["seed"] == s]
                    ax.scatter([p[k] * sc for p in q], [p["score"] for p in q], s=22, color=SERIES[c], label=f"seed {s}")
                rho = spearmanr([p[k] for p in pts], [p["score"] for p in pts]).statistic
                ax.set_title(f"{DATASET_LABEL[ds]} — ρ = {rho:+.2f}", fontsize=9.5)
                ax.set_xlabel(name)
                ax.set_ylabel("Validation Dice after 5 epochs")
                if i == 0 and j == 0:
                    ax.legend(loc="lower right")
        fig.tight_layout()
        save(fig, out / "proxies_vs_val_dice")
        made.append("proxies_vs_val_dice.png")
    groups = [("ablation_loss_coefficients", DATASETS, ["qenas", LOSS_ABLATION]),
              ("ablation_selection", SELECTION_DATASETS, ["qenas"] + list(SELECTION_METHODS))]
    for name, dss, ms in groups:   # ablation bars
        dss = [ds for ds in dss if st.seeds(ds, "qenas") and any(st.seeds(ds, m) for m in ms[1:])]
        if not dss:
            continue
        fig, axs = plt.subplots(1, len(dss), figsize=(3.4 + 1.1 * len(ms) * len(dss), 3.8), squeeze=False)
        for ax, ds in zip(axs[0], dss):
            vals = [[100 * f["test"]["dice"] for f in st.finals(ds, m)] for m in ms]
            ax.bar(range(len(ms)), [np.mean(v) if v else 0 for v in vals],
                   yerr=[mean_sd(v)[1] if len(v) > 1 else 0 for v in vals], color=[SERIES[0]] + [MUTED] * (len(ms) - 1),
                   capsize=3)
            ax.set_xticks(range(len(ms)), [_label(m).replace("QENAS, ", "") for m in ms], rotation=20, fontsize=7.5)
            lo = min([min(v) for v in vals if v] + [100])
            ax.set_ylim(max(0, lo - 5), 100)
            ax.set_ylabel("Test Dice (%)")
            ax.set_title(DATASET_LABEL[ds])
        fig.tight_layout()
        save(fig, out / name)
        made.append(f"{name}.png")
    if data_root:
        made += _qualitative(st, out, data_root)
    return made


def _qualitative(st: Store, out: Path, data_root: str) -> List[str]:
    from ..datasets.discovery import Sample, locate_data_root, rebase_path
    from ..datasets.preprocessing import load_and_resize
    from ..visualization.style import apply_style, plt, save
    made = []
    for ds in DATASETS:
        seed = min(st.seeds(ds, "qenas") or [0])
        dirs = {m: st.path(ds, m, seed) for m in MAIN_METHODS}
        dirs = {m: d for m, d in dirs.items() if d}
        if "qenas" not in dirs or len(dirs) < 2:
            continue
        preds = {}
        for m, d in dirs.items():
            z = np.load(d / "test_predictions.npz")
            preds[m] = dict(zip(z["sample_ids"].tolist(), z["masks"]))
        q = {r["sample_id"]: r["dice"] for r in _read_csv(dirs["qenas"] / "test_per_image.csv")}
        order = sorted(q, key=q.get)
        pick = [order[min(len(order) - 1, int(f * len(order)))] for f in (0.1, 0.4, 0.7, 0.95)]
        sp = load_json(dirs["qenas"] / "split.json")
        samples = {s["sample_id"]: Sample.from_dict(s) for s in sp["splits"]["test"]}
        cfg = load_json(dirs["qenas"] / "config.json")["config"]
        h, w = cfg["data"]["image_size"]
        per = {m: {r["sample_id"]: r["dice"] for r in _read_csv(d / "test_per_image.csv")} for m, d in dirs.items()}
        try:
            root = locate_data_root(data_root, cfg["data"]["dataset"])
            apply_style()
            fig, axs = plt.subplots(len(pick), 2 + len(dirs), figsize=(2.3 * (2 + len(dirs)), 2.3 * len(pick)))
            for i, sid in enumerate(pick):
                s = samples[sid]
                s.image = rebase_path(s.image, root, cfg["data"]["dataset"], sp.get("data_root"))
                s.masks = [rebase_path(m, root, cfg["data"]["dataset"], sp.get("data_root")) for m in s.masks]
                img, gt = load_and_resize(s, (h, w), int(cfg["data"]["mask_threshold"]))
                panels = [("Image", img, None), ("Ground truth", gt, None)] + \
                         [(_label(m), preds[m][sid], per[m][sid]) for m in dirs]
                for j, (title, arr, dsc) in enumerate(panels):
                    ax = axs[i, j]
                    ax.imshow(arr, cmap=None if arr.ndim == 3 else "gray")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(False)
                    ax.set_title(title if dsc is None else f"{title}\nDice {100 * dsc:.1f}", fontsize=8)
            fig.suptitle(f"{DATASET_LABEL[ds]} — test images at the 10/40/70/95th percentile of QENAS Dice "
                         f"(seed {seed})", fontsize=10)
            fig.tight_layout()
            save(fig, out / f"qualitative_{ds}")
            made.append(f"qualitative_{ds}.png")
        except Exception as exc:   # dataset files not available on this machine
            print(f"[analysis] qualitative figure for {ds} skipped: {exc}")
    return made


# ============================================================================================== driver
def _write(out: Path, name: str, rows: List[Dict], title: str, fmt_: str = "{:.4f}") -> None:
    if not rows:
        return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    save_markdown_table(out / f"{name}.md", rows, cols, title=title, float_fmt=fmt_)
    save_json(out / f"{name}.json", rows)


def analyse(results: Path, data_root: Optional[str] = None, old_results: Optional[Path] = None) -> Dict[str, Any]:
    st = Store(results)
    out = Path(results) / "analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _write(out, "T1_main_results", table_results(st, DATASETS, MAIN_METHODS, True),
           "RQ1 — Test results on identical splits (mean ± SD over seeds)", "{:.2f}")
    _write(out, "T2_significance_main", table_significance(st, DATASETS, "qenas", MAIN_METHODS[1:]),
           "RQ1 — Paired comparison on the same test images (per-image Dice averaged over seeds)", "{:.4g}")
    lc, lc_runs = loss_coefficient_table(st)
    _write(out, "T3_loss_coefficients", lc, "RQ2 — Loss-coefficient search (α grid) vs fixed α = 0.5: validation gain, "
           "test effect and cost (same architecture, paired by seed)", "{:.2f}")
    _write(out, "T4_significance_loss_coefficients", table_significance(st, DATASETS, "qenas", [LOSS_ABLATION]),
           "RQ2 — α grid vs fixed α = 0.5 on the same test images", "{:.4g}")
    _write(out, "T5_selection_ablations", table_results(st, SELECTION_DATASETS, ("qenas",) + SELECTION_METHODS, False),
           "RQ3 — Selection rule: learning capability vs random candidate vs zero-shot (max ZiCo)", "{:.2f}")
    _write(out, "T6_significance_selection", table_significance(st, SELECTION_DATASETS, "qenas", SELECTION_METHODS),
           "RQ3 — QENAS vs each selection ablation on the same test images", "{:.4g}")
    se_rows, curves = search_efficiency(st)
    _write(out, "T7_search_vs_random", se_rows,
           "RQ4 — Hypervolume of NSGA-II vs random search with the same evaluation budget (params, ZiCo; training-free)")
    sh_rows, proxy_rows, pts = successive_halving_table(st)
    _write(out, "T8_successive_halving", sh_rows, "Successive halving: agreement between rungs and the choice of the "
           "earlier rules (D21)")
    _write(out, "T9_proxies_vs_quality", proxy_rows, "Do the objectives predict early trained quality? Spearman ρ "
           "with the 5-epoch criterion of all 16 candidates")
    train, timing = training_tables(st)
    _write(out, "T10_training", train, "Final training (120 epochs): best epoch, early stopping, generalisation — D25, D29")
    _write(out, "T11_time", timing, "Compute time per experiment (minutes) and hardware")
    ext, ext_sig = external_table(st)
    _write(out, "T12_external_validation", ext, "RQ5 — External validation: trained on Kvasir-SEG, tested on all "
           "CVC-ClinicDB frames (mean ± SD over seeds)")
    _write(out, "T13_significance_external", ext_sig, "RQ5 — Paired comparison on the 612 CVC-ClinicDB frames", "{:.4g}")
    _write(out, "T15_nnunet_tta", nnunet_tta_table(st), "nnU-Net with and without its default mirroring TTA "
           "(supplementary; all main tables use no TTA)")
    if old_results is not None and Path(old_results).exists():
        v1 = Store(Path(old_results))
        rows, sig = v1_v2_tables(st, v1)
        _write(out, "T14_v1_vs_v2", rows, "Study v1 (QENAS 2.1, seeds 42–44) vs study v2 (QENAS 3.0 and the v2 "
               "protocol, seeds 45–47) on the identical test images", "{:.2f}")
        _write(out, "T14_significance_v1_vs_v2", sig, "v2 vs v1 on the same test images (per-image Dice averaged over "
               "each study's seeds)", "{:.4g}")
    made = figures(st, out / "figures", curves, pts, data_root)
    save_json(out / "loss_coefficients_per_run.json", lc_runs)
    n = len(st.exps)
    index = ["# QENAS evaluation study v2 — analysis", "",
             f"Completed experiments: {n}. Regenerate with `python study.py analyse` (add `--data-root` for the "
             "qualitative figures; the v1 → v2 comparison reads `Old Results/study_v1`).", "",
             "| file | question |", "|---|---|",
             "| T1_main_results | RQ1: QENAS vs U-Net, Attention U-Net, nnU-Net on identical splits (+ published Mixed-GGNAS) |",
             "| T2_significance_main | RQ1: paired Wilcoxon tests on the same test images, Holm-corrected |",
             "| T3_loss_coefficients, T4 | RQ2: does the α grid help, and what does it cost? |",
             "| T5_selection_ablations, T6 | RQ3: learning-capability selection vs random candidate vs max ZiCo |",
             "| T7_search_vs_random | RQ4: NSGA-II vs random search, same evaluation budget |",
             "| T8_successive_halving | consistency of the rungs; choices of the earlier rules |",
             "| T9_proxies_vs_quality | objectives vs early trained validation Dice |",
             "| T10, T11 | training behaviour, compute time |",
             "| T12, T13 | RQ5: external validation Kvasir-SEG → CVC-ClinicDB |",
             "| T14 | preliminary study v1 vs study v2 (from `Old Results/study_v1`) |",
             "| T15_nnunet_tta | nnU-Net with its default mirroring TTA (supplementary) |",
             "", "Figures: " + ", ".join(f"`figures/{m}`" for m in made), ""]
    (out / "README.md").write_text("\n".join(index), encoding="utf-8")
    print(f"[analysis] {n} completed experiments -> {out}")
    return {"experiments": n, "figures": made}
