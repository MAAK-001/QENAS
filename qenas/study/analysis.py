"""Analysis of the QENAS evaluation study: paper tables, statistics and figures (results/analysis/).

Works on partial results (every table states how many seeds it contains). Statistics:

* mean ± standard deviation over seeds (sample SD, n - 1);
* paired comparisons on the *same test images*: per-image Dice averaged over the seeds of each method, two-sided
  Wilcoxon signed-rank test, mean difference with a 95 % bootstrap confidence interval (10 000 resamples of the
  images), Holm-Bonferroni correction over the comparisons of one table and dataset;
* search quality: hypervolume of the non-dominated set of *all* architectures each search evaluated (NSGA-II vs
  random search with the same number of evaluations), both normalised by the union of their archives
  (3 objectives, reference point 1.1); NSGA-II's curve is the best-so-far set after every generation.

Every output file is listed with the research question it answers in results/analysis/README.md.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr, wilcoxon

from ..methods import METHODS
from ..utils.io import load_json, save_json, save_markdown_table
from .plan import ABLATION_DATASETS, ABLATION_METHODS, DATASETS, EXTERNAL_METHODS, LEAKAGE_METHODS, MAIN_METHODS

OBJ = ("f1_params", "f2_synflow", "f3_zico")
METRICS = [("dice", "Dice / DSC (%)", 100), ("miou", "mIoU (%)", 100), ("hd95", "HD95 (px)", 1),
           ("sensitivity", "Sensitivity (%)", 100), ("iou", "IoU (%)", 100)]
DATASET_LABEL = {"BUSI": "BUSI", "CVC": "CVC-ClinicDB", "IDRID": "IDRiD (optic disc)", "KVASIR": "Kvasir-SEG",
                 "CVC_frame_split": "CVC-ClinicDB, frame-level split"}
# Mixed-GGNAS published results (Hu et al., ESWA 2025, Tables 4, 5, 7, 8) - their own splits, for reference only.
MIXED_GGNAS = {"BUSI": {"dice": 80.64, "miou": 82.77, "params_m": 7.86, "gflops": 66.10},
               "CVC": {"dice": 93.51, "miou": 94.72, "params_m": 6.01, "gflops": 88.08, "note": "random frame split"},
               "IDRID": {"dice": 96.63, "miou": 96.78, "params_m": 29.41, "gflops": 335.34,
                         "search_hours": 26.0, "gpu": "RTX 4070 Ti Super"}}


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

    def finals(self, ds: str, method: str) -> List[Dict[str, Any]]:
        return [load_json(self.exps[(ds, method, s)] / "final_metrics.json") for s in self.seeds(ds, method)
                if (self.exps[(ds, method, s)] / "final_metrics.json").exists()]

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
    m, s, n = mean_sd([x * scale for x in values])
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


# ============================================================================================== tables
def _method_label(m: str) -> str:
    return METHODS[m]["label"]


def table_results(st: Store, datasets: Sequence[str], methods: Sequence[str], with_mixed_ggnas: bool) -> List[Dict]:
    rows = []
    for ds in datasets:
        for m in methods:
            fs = st.finals(ds, m)
            if not fs:
                continue
            row = {"Dataset": DATASET_LABEL.get(ds, ds), "Method": _method_label(m), "Seeds": len(fs)}
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
        tests = []
        for m in others:
            r = paired_test(a, st.per_image(ds, m, file))
            if r is not None:
                tests.append((m, r))
        adj = holm([r["p_wilcoxon"] for _, r in tests])
        for (m, r), p_adj in zip(tests, adj):
            rows.append({"Dataset": DATASET_LABEL.get(ds, ds), "Comparison": f"{_method_label(ref)} vs {_method_label(m)}",
                         "Images": r["n_images"], f"Dice {_method_label(ref)} (%)": 100 * r["mean_a"],
                         "Dice other (%)": 100 * r["mean_b"], "Δ Dice (pp)": 100 * r["mean_diff"],
                         "95 % CI (pp)": f"[{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}]",
                         "p (Wilcoxon)": r["p_wilcoxon"], "p (Holm)": p_adj,
                         "Images better / worse": f"{r['a_better_images']} / {r['b_better_images']}"})
    return rows


def search_efficiency(st: Store) -> Tuple[List[Dict], Dict]:
    rows, curves = [], {}
    for ds in DATASETS:
        hv_n, hv_r, per_seed = [], [], []
        for s in sorted(set(st.seeds(ds, "qenas")) & set(st.seeds(ds, "random_search"))):
            qa = load_json(st.path(ds, "qenas", s) / "search_results.json")
            ra = load_json(st.path(ds, "random_search", s) / "search_results.json")
            A = np.array([[r[o] for o in OBJ] for r in qa + ra], float)
            lo, hi = A.min(0), A.max(0)
            # best-so-far: the non-dominated set of everything NSGA-II has evaluated up to generation g
            gens = sorted({r["first_generation"] for r in qa})
            curve = [_hv(np.array([[r[o] for o in OBJ] for r in qa if r["first_generation"] <= g]), lo, hi)
                     for g in gens]
            hr = _hv(np.array([[r[o] for o in OBJ] for r in ra]), lo, hi)
            hv_n.append(curve[-1])
            hv_r.append(hr)
            per_seed.append({"seed": s, "nsga2": curve[-1], "random": hr})
            curves.setdefault(ds, []).append({"seed": s, "nsga2": curve, "random": hr, "generations": gens})
        if hv_n:
            rows.append({"Dataset": DATASET_LABEL[ds], "Seeds": len(hv_n), "Evaluations": len(qa),
                         "HV NSGA-II": fmt(hv_n, 1, 4), "HV random search": fmt(hv_r, 1, 4),
                         "NSGA-II better in seeds": f"{sum(a > b for a, b in zip(hv_n, hv_r))}/{len(hv_n)}"})
    return rows, curves


def objective_agreement(st: Store) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Unbiased random samples (random-search archives): relations between the objectives (D14), block-type
    effects, and the non-dominated share of random populations (D16)."""
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    tau_rows, block_rows, nd_rows, raw = [], [], [], {}
    for ds in DATASETS:
        taus: Dict[str, List[float]] = {}
        eff: Dict[str, List[np.ndarray]] = {}
        nd: Dict[Tuple[int, int], List[float]] = {}
        for s in st.seeds(ds, "random_search"):
            arch = load_json(st.path(ds, "random_search", s) / "search_results.json")
            F = np.array([[r[o] for o in OBJ] for r in arch], float)
            P, L, Z = (np.array([r[k] for r in arch], float) for k in ("params", "synflow_log", "zico"))
            for name, (x, y) in {"params vs SynFlow (f1, f2)": (F[:, 0], F[:, 1]),
                                 "params vs ZiCo (f1, f3)": (F[:, 0], F[:, 2]),
                                 "SynFlow vs ZiCo (f2, f3)": (F[:, 1], F[:, 2])}.items():
                taus.setdefault(name, []).append(float(kendalltau(x, y).statistic))
            rp = rankdata(P)

            def res(v):
                rv = rankdata(v)
                return rv - np.polyval(np.polyfit(rp, rv, 1), rp)
            taus.setdefault("SynFlow vs ZiCo, size removed", []).append(float(kendalltau(res(L), res(Z)).statistic))
            C = np.array([[r["chromosome"].count(g) for g in range(5)] for r in arch], float)
            X = np.column_stack([np.ones(len(arch)), C[:, 1:]])
            for name, v in (("params (M)", P / 1e6), ("ln(1+SynFlow)", L), ("ZiCo", Z)):
                eff.setdefault(name, []).append(np.linalg.lstsq(X, v, rcond=None)[0])
            rng = np.random.default_rng(s)
            for N in (n for n in (10, 20, 30, 40) if n < len(F)):
                for k in (2, 3):
                    fr = [len(NonDominatedSorting().do(F[idx][:, :k], only_non_dominated_front=True)) / N
                          for idx in (rng.choice(len(F), N, replace=False) for _ in range(1000))]
                    nd.setdefault((N, k), []).append(float(np.mean(fr)))
        if not taus:
            continue
        raw[ds] = {"tau": taus}
        tau_rows.append({"Dataset": DATASET_LABEL[ds], "Seeds": len(taus["SynFlow vs ZiCo (f2, f3)"]),
                         **{k: fmt(v, 1, 2) for k, v in taus.items()}})
        for name, betas in eff.items():
            b = np.mean(betas, 0)
            block_rows.append({"Dataset": DATASET_LABEL[ds], "Criterion": name, "All-Residual network": b[0],
                               "+1 Dense": b[1], "+1 Inception": b[2], "+1 ConvNeXt": b[3], "+1 DARTS": b[4]})
        for N in sorted({n for n, _ in nd}):
            nd_rows.append({"Dataset": DATASET_LABEL[ds], "Population": N,
                            "Non-dominated, 2 objectives (%)": fmt(nd[(N, 2)], 100, 1),
                            "Non-dominated, 3 objectives (%)": fmt(nd[(N, 3)], 100, 1)})
    return tau_rows, block_rows, nd_rows, raw


def proxy_quality(st: Store) -> Tuple[List[Dict], Dict]:
    """Do the zero-cost proxies predict the 20-epoch validation Dice of the trained Pareto candidates?"""
    rows, pts = [], {}
    for ds in DATASETS:
        per_seed: Dict[str, List[float]] = {}
        pooled = []
        for s in st.seeds(ds, "qenas"):
            an = [a for a in load_json(st.path(ds, "qenas", s) / "selection.json")["analysis"]
                  if a.get("final_val_dice") is not None]
            pooled += [dict(a, seed=s) for a in an]
            for k in ("params", "synflow_log", "zico"):
                if len(an) >= 3:
                    per_seed.setdefault(k, []).append(float(spearmanr([a[k] for a in an],
                                                                      [a["final_val_dice"] for a in an]).statistic))
        if not pooled:
            continue
        pts[ds] = pooled
        row = {"Dataset": DATASET_LABEL[ds], "Candidates": len(pooled), "Seeds": len(st.seeds(ds, "qenas"))}
        for k, name in (("params", "params"), ("synflow_log", "SynFlow"), ("zico", "ZiCo")):
            r = spearmanr([a[k] for a in pooled], [a["final_val_dice"] for a in pooled])
            row[f"ρ {name} (per seed)"] = fmt(per_seed.get(k, []), 1, 2)
            row[f"ρ {name} (pooled), p"] = f"{r.statistic:+.2f}, {r.pvalue:.2g}"
        rows.append(row)
    return rows, pts


def selection_quality(st: Store) -> List[Dict]:
    rows = []
    for ds in DATASETS:
        ranks = []
        for s in st.seeds(ds, "qenas"):
            an = [a for a in load_json(st.path(ds, "qenas", s) / "selection.json")["analysis"]
                  if a.get("final_val_dice") is not None]
            order = sorted(an, key=lambda a: -a["final_val_dice"])
            ranks.append(1 + next(i for i, a in enumerate(order) if a["selected"]))
        if ranks:
            rows.append({"Dataset": DATASET_LABEL[ds], "Seeds": len(ranks),
                         "Rank of the selected candidate by 20-epoch val Dice (1 = best)": ", ".join(map(str, ranks)),
                         "Candidates per run": len(an)})
    return rows


def training_tables(st: Store) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    tta, train, timing = [], [], []
    for ds in DATASETS + ("CVC_frame_split",):
        for m in METHODS:
            seeds = st.seeds(ds, m)
            fv = [load_json(st.path(ds, m, s) / "final_validation.json") for s in seeds
                  if (st.path(ds, m, s) / "final_validation.json").exists()]
            if fv:
                tta.append({"Dataset": DATASET_LABEL.get(ds, ds), "Method": _method_label(m), "Runs": len(fv),
                            "TTA chosen": f"{sum(f['tta']['use_tta'] for f in fv)}/{len(fv)}",
                            "Val Dice without TTA (%)": fmt([f["tta"]["val_dice_no_tta"] for f in fv], 100),
                            "Val Dice with TTA (%)": fmt([f["tta"]["val_dice_hflip_tta"] for f in fv], 100)})
                train.append({"Dataset": DATASET_LABEL.get(ds, ds), "Method": _method_label(m), "Runs": len(fv),
                              "Best epoch": fmt([f["best_epoch"] for f in fv], 1, 1),
                              "Epochs run": fmt([f["epochs_run"] for f in fv], 1, 1),
                              "Stopped early (overfitting)": f"{sum(f['stopped_early'] for f in fv)}/{len(fv)}",
                              "Train Dice (%)": fmt([f["train_metrics_best_checkpoint"]["dice"] for f in fv], 100),
                              "Val Dice (%)": fmt([f["val_metrics_best_checkpoint"]["dice"] for f in fv], 100)})
            states = [load_json(st.path(ds, m, s) / "state.json") for s in seeds]
            if states:
                row = {"Dataset": DATASET_LABEL.get(ds, ds), "Method": _method_label(m), "Runs": len(states)}
                for stage in ("search", "candidates", "loss_weights", "final"):
                    vals = [x["stages"][stage]["seconds"] / 60 for x in states if stage in x["stages"]]
                    if vals:
                        row[f"{stage} (min)"] = fmt(vals, 1, 1)
                row["total (min)"] = fmt([x.get("total_seconds", sum(v.get("seconds", 0) for v in x["stages"].values()))
                                          / 60 for x in states], 1, 1)
                env = st.path(ds, m, seeds[0]) / "environment.json"
                row["GPU"] = load_json(env).get("gpu_name", "cpu") if env.exists() else "?"
                timing.append(row)
    return tta, train, timing


def external_table(st: Store) -> Tuple[List[Dict], List[Dict]]:
    rows = []
    for m in EXTERNAL_METHODS:
        res = [load_json(st.path("KVASIR", m, s) / "external_CVC" / "external_metrics.json") for s in st.seeds("KVASIR", m)
               if (st.path("KVASIR", m, s) / "external_CVC" / "external_metrics.json").exists()]
        if res:
            rows.append({"Train → test": "Kvasir-SEG → CVC-ClinicDB (all 612 frames)", "Method": _method_label(m),
                         "Seeds": len(res), **{label: fmt([r["metrics"][k] for r in res], sc) for k, label, sc in METRICS}})
    sig = table_significance(st, ["KVASIR"], "qenas", [m for m in EXTERNAL_METHODS if m != "qenas"],
                             file="external_CVC/external_per_image.csv")
    for r in sig:
        r["Dataset"] = "Kvasir-SEG → CVC-ClinicDB"
    return rows, sig


def leakage_table(st: Store) -> List[Dict]:
    rows = []
    for m in LEAKAGE_METHODS:
        seq, frame = st.finals("CVC", m), st.finals("CVC_frame_split", m)
        if seq or frame:
            rows.append({"Method": _method_label(m),
                         "Seeds (sequence / frame)": f"{len(seq)} / {len(frame)}",
                         "Dice, sequence split (%)": fmt([f["test"]["dice"] for f in seq], 100),
                         "Dice, frame split (%)": fmt([f["test"]["dice"] for f in frame], 100),
                         "Inflation (pp)": (f"{100 * (np.mean([f['test']['dice'] for f in frame]) - np.mean([f['test']['dice'] for f in seq])):+.2f}"
                                            if seq and frame else "–"),
                         "Train Dice, sequence split (%)": fmt([f["train"]["dice"] for f in seq], 100)})
    return rows


# ============================================================================================== figures
def figures(st: Store, out: Path, curves: Dict, proxy_pts: Dict, data_root: Optional[str]) -> List[str]:
    from ..visualization.style import INK_2, MUTED, SERIES, apply_style, plt, save
    made = []
    apply_style()
    main = [m for m in MAIN_METHODS]
    # F1 per-image Dice boxplots
    dsl = [ds for ds in DATASETS if any(st.seeds(ds, m) for m in main)]
    if dsl:
        fig, axs = plt.subplots(1, len(dsl), figsize=(4.2 * len(dsl), 4.2), squeeze=False)
        for ax, ds in zip(axs[0], dsl):
            data, labels = [], []
            for m in main:
                pi = st.per_image(ds, m)
                if pi:
                    data.append([np.mean([r["dice"] for r in v]) for v in pi.values()])
                    labels.append(_method_label(m))
            ax.boxplot(data, showmeans=True, widths=0.6)
            ax.set_xticks(range(1, len(labels) + 1), labels, rotation=20)
            ax.set_title(DATASET_LABEL[ds])
            ax.set_ylabel("Per-image test Dice (mean over seeds)")
        fig.tight_layout()
        save(fig, out / "per_image_dice_boxplots")
        made.append("per_image_dice_boxplots.png")
    # F2 search progress vs random search
    if curves:
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
    # F3 proxies vs 20-epoch validation Dice
    if proxy_pts:
        fig, axs = plt.subplots(2, len(proxy_pts), figsize=(4.0 * len(proxy_pts), 7.0), squeeze=False)
        for j, (ds, pts) in enumerate(proxy_pts.items()):
            for i, (k, name) in enumerate((("zico", "ZiCo"), ("synflow_log", "ln(1 + SynFlow)"))):
                ax = axs[i, j]
                for c, s in enumerate(sorted({p["seed"] for p in pts})):
                    q = [p for p in pts if p["seed"] == s]
                    ax.scatter([p[k] for p in q], [p["final_val_dice"] for p in q], s=22, color=SERIES[c],
                               label=f"seed {s}")
                r = spearmanr([p[k] for p in pts], [p["final_val_dice"] for p in pts]).statistic
                ax.set_title(f"{DATASET_LABEL[ds]} — ρ = {r:+.2f}", fontsize=9.5)
                ax.set_xlabel(name)
                ax.set_ylabel("20-epoch validation Dice")
                if i == 0 and j == 0:
                    ax.legend(loc="lower right")
        fig.tight_layout()
        save(fig, out / "proxies_vs_val_dice")
        made.append("proxies_vs_val_dice.png")
    # F4 accuracy vs size
    fig, axs = plt.subplots(1, max(1, len(dsl)), figsize=(4.2 * max(1, len(dsl)), 3.8), squeeze=False)
    for ax, ds in zip(axs[0], dsl if dsl else []):
        for c, m in enumerate(main):
            fs = st.finals(ds, m)
            if fs:
                ax.errorbar([np.mean([f["params_m"] for f in fs])], [100 * np.mean([f["test"]["dice"] for f in fs])],
                            yerr=[100 * mean_sd([f["test"]["dice"] for f in fs])[1]], fmt="o", color=SERIES[c],
                            label=_method_label(m), capsize=3)
        if ds in MIXED_GGNAS:
            ax.scatter([MIXED_GGNAS[ds]["params_m"]], [MIXED_GGNAS[ds]["dice"]], marker="x", color=MUTED,
                       label="Mixed-GGNAS (published, own split)")
        ax.set_xscale("log")
        ax.set_xlabel("Parameters (M, log)")
        ax.set_ylabel("Test Dice (%)")
        ax.set_title(DATASET_LABEL[ds])
        ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    if dsl:
        save(fig, out / "accuracy_vs_parameters")
        made.append("accuracy_vs_parameters.png")
    else:
        plt.close(fig)
    # F5 ablations
    abl = [ds for ds in ABLATION_DATASETS if st.seeds(ds, "qenas")]
    if abl:
        ms = ["qenas"] + list(ABLATION_METHODS)
        fig, axs = plt.subplots(1, len(abl), figsize=(5.2 * len(abl), 3.8), squeeze=False)
        for ax, ds in zip(axs[0], abl):
            vals = [[100 * f["test"]["dice"] for f in st.finals(ds, m)] for m in ms]
            ax.bar(range(len(ms)), [np.mean(v) if v else 0 for v in vals],
                   yerr=[mean_sd(v)[1] if len(v) > 1 else 0 for v in vals], color=[SERIES[0]] + [MUTED] * 4, capsize=3)
            ax.set_xticks(range(len(ms)), [_method_label(m).replace("QENAS, ", "").replace("QENAS ", "") for m in ms],
                          rotation=25, fontsize=7.5)
            lo = min([min(v) for v in vals if v] + [100])
            ax.set_ylim(max(0, lo - 5), 100)
            ax.set_ylabel("Test Dice (%)")
            ax.set_title(f"Ablations — {DATASET_LABEL[ds]}")
        fig.tight_layout()
        save(fig, out / "ablations")
        made.append("ablations.png")
    # F6 qualitative comparison (seed 42 predictions on the same test images)
    if data_root:
        made += _qualitative(st, out, data_root)
    return made


def _qualitative(st: Store, out: Path, data_root: str) -> List[str]:
    from ..datasets.discovery import Sample, locate_data_root, rebase_path
    from ..datasets.preprocessing import load_and_resize
    from ..visualization.style import apply_style, plt, save
    made = []
    for ds in DATASETS:
        seed = 42
        dirs = {m: st.path(ds, m, seed) for m in MAIN_METHODS}
        if not all(dirs.values()):
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
        root = locate_data_root(data_root, cfg["data"]["dataset"])
        h, w = cfg["data"]["image_size"]
        per = {m: {r["sample_id"]: r["dice"] for r in _read_csv(d / "test_per_image.csv")} for m, d in dirs.items()}
        try:
            apply_style()
            fig, axs = plt.subplots(len(pick), 2 + len(MAIN_METHODS), figsize=(2.3 * (2 + len(MAIN_METHODS)), 2.3 * len(pick)))
            for i, sid in enumerate(pick):
                s = samples[sid]
                s.image = rebase_path(s.image, root, cfg["data"]["dataset"], sp.get("data_root"))
                s.masks = [rebase_path(m, root, cfg["data"]["dataset"], sp.get("data_root")) for m in s.masks]
                img, gt = load_and_resize(s, (h, w), int(cfg["data"]["mask_threshold"]))
                panels = [("Image", img, None), ("Ground truth", gt, None)] + \
                         [(_method_label(m), preds[m][sid], per[m][sid]) for m in MAIN_METHODS]
                for j, (title, arr, dsc) in enumerate(panels):
                    ax = axs[i, j]
                    ax.imshow(arr, cmap=None if arr.ndim == 3 else "gray")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(False)
                    ax.set_title(title if dsc is None else f"{title}\nDice {100 * dsc:.1f}", fontsize=8)
            fig.suptitle(f"{DATASET_LABEL[ds]} — test images at the 10/40/70/95th percentile of QENAS Dice (seed 42)",
                         fontsize=10)
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


def analyse(results: Path, data_root: Optional[str] = None) -> Dict[str, Any]:
    st = Store(results)
    out = Path(results) / "analysis"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _write(out, "T1_main_results", table_results(st, DATASETS, MAIN_METHODS, True),
           "RQ1 — Test results on identical splits (mean ± SD over seeds)", "{:.2f}")
    _write(out, "T2_significance_main", table_significance(st, DATASETS, "qenas", [m for m in MAIN_METHODS if m != "qenas"]),
           "RQ1 — Paired comparison on the same test images (per-image Dice averaged over seeds)", "{:.4g}")
    _write(out, "T3_ablations", table_results(st, ABLATION_DATASETS, ("qenas",) + ABLATION_METHODS, False),
           "RQ2 — Ablations (mean ± SD over seeds)", "{:.2f}")
    _write(out, "T4_significance_ablations", table_significance(st, ABLATION_DATASETS, "qenas", ABLATION_METHODS),
           "RQ2 — QENAS vs each ablation on the same test images", "{:.4g}")
    se_rows, curves = search_efficiency(st)
    _write(out, "T5_search_vs_random", se_rows,
           "RQ3 — Hypervolume of NSGA-II vs random search with the same evaluation budget (training-free)")
    tau, blocks, nd, raw = objective_agreement(st)
    _write(out, "T6_objective_agreement", tau, "Objective relations on random architectures (Kendall τ of the "
           "minimised objectives; > 0 = improve together, < 0 = trade-off) — DESIGN_DECISIONS D14")
    _write(out, "T7_block_effects", blocks, "Effect of one gene of each block type (vs Residual), least squares on "
           "random architectures — DESIGN_DECISIONS D14", "{:.3f}")
    _write(out, "T8_nondominated_share", nd, "Non-dominated share of random populations — DESIGN_DECISIONS D16")
    pq_rows, pts = proxy_quality(st)
    _write(out, "T9_proxies_vs_quality", pq_rows, "Do the zero-cost proxies predict trained quality? Spearman ρ with "
           "the 20-epoch validation Dice of the Pareto candidates")
    _write(out, "T10_selection_quality", selection_quality(st), "Learning-capability selection: rank of the chosen "
           "candidate among the trained candidates")
    tta, train, timing = training_tables(st)
    _write(out, "T11_tta", tta, "Flip TTA decided on validation — DESIGN_DECISIONS D27")
    _write(out, "T12_training", train, "Final training: best epoch, early stopping, generalisation — D25, D29")
    _write(out, "T13_time", timing, "Compute time per experiment (minutes) and hardware")
    ext, ext_sig = external_table(st)
    _write(out, "T14_external_validation", ext, "RQ4 — External validation: trained on Kvasir-SEG, tested on all "
           "CVC-ClinicDB frames (mean ± SD over seeds)")
    _write(out, "T15_significance_external", ext_sig, "RQ4 — Paired comparison on the 612 CVC-ClinicDB frames", "{:.4g}")
    _write(out, "T16_cvc_leakage", leakage_table(st), "RQ5 — CVC-ClinicDB: sequence-level vs random frame split")
    made = figures(st, out / "figures", curves, pts, data_root)
    n = len(st.exps)
    index = ["# QENAS evaluation study — analysis", "",
             f"Completed experiments: {n}. Regenerate with `python study.py analyse` (add `--data-root` for the "
             "qualitative figures).", "",
             "| file | question |", "|---|---|",
             "| T1_main_results | RQ1: QENAS vs U-Net vs Attention U-Net on identical splits (+ published Mixed-GGNAS) |",
             "| T2_significance_main | RQ1: paired Wilcoxon tests on the same test images, Holm-corrected |",
             "| T3_ablations, T4_significance_ablations | RQ2: contribution of ZiCo, SynFlow and learning-capability selection |",
             "| T5_search_vs_random | RQ3: NSGA-II vs random search, same evaluation budget |",
             "| T6–T8 | objective relations, block effects, non-dominated share (D14, D16) |",
             "| T9_proxies_vs_quality | zero-cost proxies vs trained validation Dice |",
             "| T10_selection_quality | rank of the learning-capability choice |",
             "| T11–T13 | TTA decisions, training behaviour, compute time |",
             "| T14, T15 | RQ4: external validation Kvasir-SEG → CVC-ClinicDB |",
             "| T16_cvc_leakage | RQ5: score inflation of a random frame split on CVC-ClinicDB |",
             "", "Figures: " + ", ".join(f"`figures/{m}`" for m in made), ""]
    (out / "README.md").write_text("\n".join(index), encoding="utf-8")
    save_json(out / "objective_agreement_raw.json", raw)
    print(f"[analysis] {n} completed experiments -> {out}")
    return {"experiments": n, "figures": made}
