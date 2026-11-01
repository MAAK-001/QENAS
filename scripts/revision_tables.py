"""Journal revision: merge the results of the five Kaggle accounts and regenerate the affected tables.

    python scripts/revision_tables.py --fetch           # fetch + merge the 5 result branches, then the tables
    python scripts/revision_tables.py                   # tables only (results already merged)
    python scripts/revision_tables.py --margin 2.0      # TOST equivalence margin in Dice points (pre-specified: 2)

--fetch extracts ``results/`` of the GitHub branches ``kaggle-revision-1`` ... ``-5`` into ``results/`` (only new
experiment folders, logs and status files; the study-v2 results are not touched), compares the checksum files of the
accounts (code, data, splits, software) and lists missing jobs.

Writes ``results/revision/`` (CSV + Markdown per table, ``report.json``):

* ``R1_accuracy_<metric>``    - accuracy table (Dice, mIoU, HD95, sensitivity, IoU): QENAS, nnU-Net (100 and 1000
                                epochs), Attention U-Net, U-Net, UNeXt, Supervised NSGA-II; four test sets + external;
* ``R2_paired_dice``          - QENAS vs nnU-Net, Attention U-Net, U-Net, UNeXt: per-image paired Wilcoxon, Holm over
                                these four per test set, bootstrap CI, rank-biserial r;
  ``R2_friedman``             - Friedman test and mean ranks of the five networks, Nemenyi critical difference;
* ``R3_qenas_vs_nnunet_1000`` - paired statistics + TOST (margin ±``--margin`` Dice points) of QENAS vs nnU-Net with
                                1000 epochs, without TTA (main) and with nnU-Net's mirroring TTA (secondary);
* ``R4_size_and_gpu_time``    - parameters, FLOPs (thop convention), inference time and GPU hours per run;
* ``R5_qenas_vs_supervised``  - the new table "QENAS vs Supervised NSGA-II" (Dice, HD95, parameters, GPU time).

Statistics: the study's own code (scripts/paper_analysis.py: ``paired`` = per-image values averaged over seeds,
two-sided Wilcoxon signed-rank, 10 000-resample bootstrap CI, rank-biserial r; ``holm``). TOST (two one-sided tests)
on the same per-image differences d = QENAS - other: H0a mean d <= -margin, H0b mean d >= +margin, each tested with a
one-sided Wilcoxon signed-rank test (and, as a check, a one-sided paired t-test); equivalence if both are rejected at
0.05, i.e. p_TOST = max(p_a, p_b) < 0.05; the 90 % bootstrap CI of the mean difference is reported with it.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, ttest_1samp, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import paper_analysis as pa  # noqa: E402  (the study's statistics: paired, holm, fmt_p, mean_sd)

from qenas.study.plan import REVISION_WORKERS, revision_plan  # noqa: E402
from qenas.study.scheduler import job_done  # noqa: E402

DATASETS = ["BUSI", "CVC", "IDRID", "KVASIR"]
COLS = DATASETS + ["EXT"]
DSL = {"BUSI": "BUSI", "CVC": "CVC-ClinicDB", "IDRID": "IDRiD", "KVASIR": "Kvasir-SEG", "EXT": "Kvasir-SEG → CVC-ClinicDB"}
SEEDS = [45, 46, 47]
LABEL = {"qenas": "QENAS (ours)", "nnunet": "nnU-Net (100 epochs)", "nnunet_full": "nnU-Net (1000 epochs)",
         "attention_unet": "Attention U-Net", "unet": "U-Net", "unext": "UNeXt",
         "supervised_nsga2": "Supervised NSGA-II"}
ACCURACY_MODELS = ["qenas", "nnunet", "nnunet_full", "attention_unet", "unet", "unext", "supervised_nsga2"]
HOLM_FAMILY = ["nnunet", "attention_unet", "unet", "unext"]          # QENAS vs these four, Holm per test set
FRIEDMAN = ["qenas", "nnunet", "attention_unet", "unet", "unext"]
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}   # Demsar 2006, Table 5
METRICS = [("dice", "Dice / DSC (%)", 100, False), ("miou", "mIoU (%)", 100, False), ("hd95", "HD95 (px)", 1, True),
           ("sensitivity", "Sensitivity (%)", 100, False), ("iou", "Foreground IoU (%)", 100, False)]
BRANCHES = [f"kaggle-revision-{w}" for w in sorted(REVISION_WORKERS)]


# ------------------------------------------------------------------------------------------- loading
def exp(res, ds, m, s):
    return res / ds / m / f"seed_{s}"


def done_seeds(res, ds, m):
    out = []
    for s in SEEDS:
        st = exp(res, ds, m, s) / "state.json"
        if st.exists() and pa.load(st).get("status") == "completed":
            out.append(s)
    return out


def finals(res, ds, m):
    return [pa.load(exp(res, ds, m, s) / "final_metrics.json") for s in done_seeds(res, ds, m)]


def per_image(res, ds, m, key="dice", file="test_per_image.csv"):
    """sample_id -> value averaged over the finished seeds (EXT: the external CVC-ClinicDB test of Kvasir-SEG)."""
    acc = {}
    for s in done_seeds(res, "KVASIR" if ds == "EXT" else ds, m):
        p = (exp(res, "KVASIR", m, s) / "external_CVC" / "external_per_image.csv") if ds == "EXT" else \
            exp(res, ds, m, s) / file
        if p.exists():
            for r in pa.read_csv(p):
                acc.setdefault(r["sample_id"], []).append(float(r[key]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def metric_values(res, c, m, key):
    if c == "EXT":
        vals = []
        for s in done_seeds(res, "KVASIR", m):
            p = exp(res, "KVASIR", m, s) / "external_CVC" / "external_metrics.json"
            if p.exists():
                vals.append(pa.load(p)["metrics"][key])
        return vals
    return [f["test"][key] for f in finals(res, c, m)]


def stage_hours(res, ds, m, stage=None):
    out = []
    for s in done_seeds(res, ds, m):
        st = pa.load(exp(res, ds, m, s) / "state.json")
        out.append((st["total_seconds"] if stage is None else st["stages"].get(stage, {}).get("seconds", 0.0)) / 3600)
    return out


def write(out, name, header, rows, title):
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md = [f"# {title}", "", "| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    (out / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"title": title, "header": header, "rows": rows}


# ------------------------------------------------------------------------------------------- statistics
def tost(a, b, margin):
    """Equivalence of the per-image means of a and b (dicts sample_id -> value in [0, 1]) within ±margin."""
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    if len(d) < 5:
        return None
    lo_w = float(wilcoxon(d + margin, alternative="greater").pvalue) if np.any(d + margin != 0) else 0.0
    hi_w = float(wilcoxon(d - margin, alternative="less").pvalue) if np.any(d - margin != 0) else 0.0
    lo_t = float(ttest_1samp(d, -margin, alternative="greater").pvalue)
    hi_t = float(ttest_1samp(d, margin, alternative="less").pvalue)
    boot = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
    ci90 = (float(np.percentile(boot, 5)), float(np.percentile(boot, 95)))
    return {"n": len(d), "diff": float(d.mean()), "ci90": ci90, "p_tost_wilcoxon": max(lo_w, hi_w),
            "p_tost_t": max(lo_t, hi_t), "equivalent": max(lo_w, hi_w) < 0.05,
            "ci90_inside_margin": -margin < ci90[0] and ci90[1] < margin}


def paired_row(c, name, r, p_adj, lower=False, unit="pp", scale=100):
    verdict = "no significant difference"
    if p_adj < 0.05:
        qenas_better = (r["diff"] < 0) if lower else (r["diff"] > 0)
        verdict = "QENAS better" if qenas_better else f"{name.split(' vs ')[1]} better"
    return [DSL[c], name, r["n"], f"{scale * r['a']:.2f}", f"{scale * r['b']:.2f}", f"{scale * r['diff']:+.2f}",
            f"[{scale * r['lo']:+.2f}, {scale * r['hi']:+.2f}]", pa.fmt_p(r["p"]), pa.fmt_p(p_adj), f"{r['rb']:+.2f}",
            f"{r['better']} / {r['worse']}", verdict]


PAIRED_HEADER = ["Test set", "Comparison", "Images", "QENAS", "Other", "Δ (QENAS − other)", "95 % CI",
                 "p (Wilcoxon)", "p (Holm)", "Effect size r", "Images higher / lower", "Verdict"]


# ------------------------------------------------------------------------------------------- tables
def r1_accuracy(res, out):
    tables = {}
    for key, title, sc, lower in METRICS:
        rows = []
        for m in ACCURACY_MODELS:
            vals = [metric_values(res, c, m, key) for c in COLS]
            if not any(vals):
                continue
            rows.append([LABEL[m], max(len(v) for v in vals)] + [pa.mean_sd(v, sc, 2) for v in vals])
        tables[key] = write(out, f"R1_accuracy_{key}", ["Model", "Runs"] + [DSL[c] for c in COLS], rows,
                            f"{title} - mean ± SD over seeds 45-47 ({'lower' if lower else 'higher'} is better)")
    return tables


def r2_paired(res, out):
    rows, fr_rows, ranks = [], [], {}
    for c in COLS:
        q = per_image(res, c, "qenas")
        tests = [(m, pa.paired(q, o)) for m in HOLM_FAMILY for o in [per_image(res, c, m)] if len(set(q) & set(o)) >= 5]
        for (m, r), p_adj in zip(tests, pa.holm([r["p"] for _, r in tests])):
            rows.append(paired_row(c, f"QENAS vs {LABEL[m]}", r, p_adj))
        data = {m: per_image(res, c, m) for m in FRIEDMAN}
        if all(data.values()):
            ids = sorted(set.intersection(*[set(v) for v in data.values()]))
            M = np.array([[data[m][i] for m in FRIEDMAN] for i in ids])
            st, p = friedmanchisquare(*M.T)
            rk = np.mean([rankdata(-row) for row in M], axis=0)
            k, n = len(FRIEDMAN), len(ids)
            cd = NEMENYI_Q05[k] * math.sqrt(k * (k + 1) / (6 * n))
            ranks[c] = {"ranks": dict(zip(FRIEDMAN, rk.tolist())), "cd": cd, "n": n, "p": float(p)}
            fr_rows.append([DSL[c], n, f"{st:.1f}", pa.fmt_p(p)] + [f"{x:.2f}" for x in rk] + [f"{cd:.2f}"])
    t1 = write(out, "R2_paired_dice", PAIRED_HEADER, rows,
               "QENAS vs nnU-Net (100 epochs), Attention U-Net, U-Net and UNeXt - per-image Dice (%), Holm over the "
               "four comparisons of each test set")
    t2 = write(out, "R2_friedman", ["Test set", "Images", "Friedman χ²", "p"] +
               [f"Mean rank {LABEL[m]}" for m in FRIEDMAN] + ["Nemenyi CD (α = 0.05)"], fr_rows,
               "Friedman test over the five networks (per-image Dice; rank 1 = best) and Nemenyi critical difference")
    return {"pairwise": t1, "friedman": t2, "ranks": ranks}


def r3_nnunet_full(res, out, margin):
    rows = []
    for ds in DATASETS:
        if not done_seeds(res, ds, "nnunet_full"):
            continue
        q = per_image(res, ds, "qenas")
        for other, file, lab in (("nnunet_full", "test_per_image.csv", "nnU-Net 1000 epochs, no TTA (main)"),
                                 ("nnunet_full", "test_per_image_mirroring_tta.csv",
                                  "nnU-Net 1000 epochs, mirroring TTA (secondary)"),
                                 ("nnunet", "test_per_image.csv", "nnU-Net 100 epochs, no TTA (reference)")):
            o = per_image(res, ds, other, file=file)
            if len(set(q) & set(o)) < 5:
                continue
            r, t = pa.paired(q, o), tost(q, o, margin / 100)
            rows.append([DSL[ds], f"QENAS vs {lab}", r["n"], len(done_seeds(res, ds, other)), f"{100 * r['a']:.2f}",
                         f"{100 * r['b']:.2f}", f"{100 * r['diff']:+.2f}", f"[{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]",
                         pa.fmt_p(r["p"]), f"{r['rb']:+.2f}", f"[{100 * t['ci90'][0]:+.2f}, {100 * t['ci90'][1]:+.2f}]",
                         pa.fmt_p(t["p_tost_wilcoxon"]), pa.fmt_p(t["p_tost_t"]),
                         f"equivalent within ±{margin:g}" if t["equivalent"] else f"not shown equivalent (±{margin:g})"])
    return write(out, "R3_qenas_vs_nnunet_1000", [
        "Dataset", "Comparison", "Images", "nnU-Net runs", "Dice QENAS (%)", "Dice nnU-Net (%)", "Δ Dice (pp)",
        "95 % CI (pp)", "p (Wilcoxon)", "Effect size r", "90 % CI (pp)", "p TOST (Wilcoxon)", "p TOST (t)", "TOST verdict"],
        rows, f"QENAS vs nnU-Net with the default 1000-epoch trainer: paired per-image Dice and TOST equivalence "
              f"(pre-specified margin ±{margin:g} Dice points)")


def r4_cost(res, out):
    rows = []
    for ds in DATASETS:
        for m in ACCURACY_MODELS:
            fs = finals(res, ds, m)
            if not fs:
                continue
            rows.append([DSL[ds], LABEL[m], len(fs), pa.mean_sd([f["params_m"] for f in fs], 1, 3),
                         pa.mean_sd([f["gflops_mixed_ggnas"] for f in fs], 1, 1),
                         pa.mean_sd([f["gflops_per_image"] for f in fs], 1, 2),
                         pa.mean_sd([f["inference_ms_per_image"] for f in fs], 1, 1),
                         pa.mean_sd(stage_hours(res, ds, m), 1, 2), f"{sum(stage_hours(res, ds, m)):.2f}"])
    return write(out, "R4_size_and_gpu_time", [
        "Dataset", "Model", "Runs", "Params (M)", "FLOPs (G, training batch: 8, IDRiD 4)", "GFLOPs / image",
        "ms / image (T4, batch 1)", "GPU hours per run (start to end)", "GPU hours, all runs"], rows,
        "Model size, FLOPs (thop convention), inference time and GPU time (Tesla T4)")


def rung1_identity(res, ds, s):
    """How many of the 16 candidates have a successive-halving rung-1 score equal to their search fitness S1."""
    d = exp(res, ds, "supervised_nsga2", s)
    if not (d / "candidate_results.json").exists():
        return None
    s1 = {tuple(r["chromosome"]): r["s1_dice"] for r in pa.load(d / "search_results.json")}
    eq = [abs(c["rung_scores"].get("5", float("nan")) - s1[tuple(c["chromosome"])]) == 0.0
          for c in pa.load(d / "candidate_results.json") if c.get("rung_scores")]
    diff = [abs(c["rung_scores"]["5"] - s1[tuple(c["chromosome"])]) for c in pa.load(d / "candidate_results.json")
            if c.get("rung_scores")]
    return sum(eq), len(eq), max(diff) if diff else float("nan")


def r5_supervised(res, out):
    rows = []
    for ds in DATASETS:
        if not done_seeds(res, ds, "supervised_nsga2"):
            continue
        fq, fs = finals(res, ds, "qenas"), finals(res, ds, "supervised_nsga2")
        tests = []
        for key, lab, lower in (("dice", "Test Dice (%)", False), ("hd95", "Test HD95 (px)", True)):
            r = pa.paired(per_image(res, ds, "qenas", key), per_image(res, ds, "supervised_nsga2", key))
            tests.append((key, lab, lower, r))
        padj = pa.holm([t[3]["p"] for t in tests])
        for (key, lab, lower, r), pj in zip(tests, padj):
            sc = 100 if key == "dice" else 1
            rows.append([DSL[ds], lab, pa.mean_sd([f["test"][key] for f in fq], sc), pa.mean_sd([f["test"][key] for f in fs], sc),
                         f"{sc * r['diff']:+.2f} [{sc * r['lo']:+.2f}, {sc * r['hi']:+.2f}]", pa.fmt_p(r["p"]),
                         pa.fmt_p(pj), f"{r['rb']:+.2f}", "lower is better" if lower else "higher is better"])
        pq, ps = [f["params_m"] for f in fq], [f["params_m"] for f in fs]
        rows.append([DSL[ds], "Params (M)", pa.mean_sd(pq, 1, 3), pa.mean_sd(ps, 1, 3),
                     f"ratio {np.mean(ps) / np.mean(pq):.2f}×", "–", "–", "–", "per seed: " +
                     ", ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(pq, ps))])
        for stage, lab in (("search", "Search (GPU h)"), ("candidates", "Successive halving (GPU h)"),
                           ("loss_weights", "α grid (GPU h)"), ("final", "Final training (GPU h)"), (None, "Total, start to end (GPU h)")):
            hq, hs = stage_hours(res, ds, "qenas", stage), stage_hours(res, ds, "supervised_nsga2", stage)
            rows.append([DSL[ds], lab, pa.mean_sd(hq, 1, 2), pa.mean_sd(hs, 1, 2),
                         f"ratio {np.mean(hs) / np.mean(hq):.1f}×" if np.mean(hq) > 0 else "–", "–", "–", "–", ""])
        ep = [pa.load(exp(res, ds, "supervised_nsga2", s) / "search_summary.json")
              for s in done_seeds(res, ds, "supervised_nsga2")]
        rows.append([DSL[ds], "Training epochs in the search", "0 (training-free)",
                     pa.mean_sd([e["fitness_epochs_total"] for e in ep], 1, 0), "–", "–", "–", "–",
                     f"distinct architectures trained: {pa.mean_sd([e['fitness_trainings'] for e in ep], 1, 0)}"])
        ident = [rung1_identity(res, ds, s) for s in done_seeds(res, ds, "supervised_nsga2")]
        ident = [x for x in ident if x]
        if ident:
            rows.append([DSL[ds], "Rung-1 score = search S1 (bit-identical)", "–",
                         f"{sum(x[0] for x in ident)}/{sum(x[1] for x in ident)} candidates", "–", "–", "–", "–",
                         f"max |difference| {max(x[2] for x in ident):.2e} (retrained, GPU kernels)"])
        sel_q = [pa.load(exp(res, ds, "qenas", s) / "final_metrics.json")["architecture_label"] for s in done_seeds(res, ds, "qenas")]
        sel_s = [f["architecture_label"] for f in fs]
        rows.append([DSL[ds], "Selected chromosome per seed", ", ".join(sel_q), ", ".join(sel_s), "–", "–", "–", "–", ""])
    return write(out, "R5_qenas_vs_supervised_nsga2", [
        "Dataset", "Quantity", "QENAS (training-free search)", "Supervised NSGA-II", "Δ QENAS − Supervised [95 % CI] / ratio",
        "p (Wilcoxon, per image)", "p (Holm)", "Effect size r", "Note"], rows,
        "QENAS vs Supervised NSGA-II (identical except the search objective f2): accuracy, size and GPU time; "
        "mean ± SD over seeds 45-47")


# ------------------------------------------------------------------------------------------- merge
def fetch(res):
    """Extract results/ of every revision branch into res (git archive; no checkout of main is touched)."""
    for b in BRANCHES:
        r = subprocess.run(["git", "fetch", "-q", "origin", b], cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[fetch] {b}: not available ({r.stderr.strip()[-200:]})")
            continue
        data = subprocess.run(["git", "archive", f"origin/{b}", "results"], cwd=ROOT, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            members = [m for m in tar.getmembers() if m.name.startswith("results/")]
            for m in members:
                m.name = m.name[len("results/"):]
            tar.extractall(res, members=[m for m in members if m.name], filter="data")
        print(f"[fetch] {b}: {len(members)} files -> {res}")


def check(res):
    files = sorted(res.glob("revision_checksums_*.json"))
    rc = 0
    if files:
        import revision_checksums
        print(f"[check] identity of {len(files)} accounts:")
        rc = revision_checksums.compare(files)
    jobs = revision_plan()
    missing = [j.id for j in jobs if not job_done(res, j)]
    print(f"[check] {len(jobs) - len(missing)}/{len(jobs)} required revision jobs done" +
          (": missing " + ", ".join(missing) if missing else ""))
    return rc, missing


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--fetch", action="store_true", help="merge the kaggle-revision-1..5 branches first")
    p.add_argument("--margin", type=float, default=2.0, help="TOST equivalence margin in Dice points")
    a = p.parse_args(argv)
    res = Path(a.results)
    if a.fetch:
        fetch(res)
    rc, missing = check(res)
    out = res / "revision"
    report = {"margin_dice_points": a.margin, "missing_jobs": missing, "identity_check_ok": rc == 0,
              "R1": r1_accuracy(res, out), "R2": r2_paired(res, out), "R3": r3_nnunet_full(res, out, a.margin),
              "R4": r4_cost(res, out), "R5": r5_supervised(res, out)}
    with open(out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)
    print(f"[revision] tables -> {out}" + (f" | {len(missing)} jobs still missing" if missing else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
