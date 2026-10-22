"""Publication tables of QENAS 3.0 (study v2) -> one Word file.

    python scripts/paper_tables.py
    NODE_PATH=<folder containing the npm package docx> node scripts/render_tables_docx.js

Writes results/tables/tables_spec.json (every number computed here from the saved results); the renderer turns it
into results/tables/QENAS Tables.docx. Conventions (stated in the document too):
* every model of this study, QENAS included: mean ± SD of its 3 runs (seeds 45–47), identical split (the
  single-run figures show the QENAS run with the highest test Dice; the tables do not);
* bold: best value of a column among the models of this study; published results (other splits) are not ranked.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import paper_figures as pf  # noqa: E402

pa = pf.pa
OUT = pf.RES / "tables"
DATASETS, TESTS, SEEDS, MAIN = pf.DATASETS, pf.TESTS, pf.SEEDS, pf.MAIN
BASE = MAIN[1:]
DSL = {"BUSI": "BUSI", "CVC": "CVC-ClinicDB", "IDRID": "IDRiD", "KVASIR": "Kvasir-SEG", "EXT": "Kvasir-SEG → CVC-ClinicDB"}
NAME = pf.NAME
DAG, DDAG = "†", "‡"


# ------------------------------------------------------------------------------------------- helpers
def c(t, b=False, i=False, span=1, al=None):
    t = re.sub(r"(?<![\w.])-(?=\d)", "−", str(t)).replace("<1e-4", "< 0.0001")   # typographic minus
    d = {"t": t}
    if b:
        d["b"] = True
    if i:
        d["i"] = True
    if span > 1:
        d["span"] = span
    if al:
        d["al"] = al
    return d


def sub(text, n):
    return [c(text, i=True, span=n, al="l")]


def fmt(v, dg=2):
    v = [x for x in v if x is not None]
    if not v:
        return "–"
    if len(v) == 1 or np.std(v) < 10 ** (-dg - 1):
        return f"{np.mean(v):.{dg}f}"
    return f"{np.mean(v):.{dg}f} ± {np.std(v, ddof=1):.{dg}f}"


def ranked_rows(labels, values, lower=False, dg=2):
    """labels: row labels; values[r][k] = list of run values (or None) -> rows with the best cell of each column
    (by the rounded mean) in bold. lower: one flag for all columns or one per column."""
    ncol = len(values[0])
    low = lower if isinstance(lower, (list, tuple)) else [lower] * ncol
    best = []
    for k in range(ncol):
        ms = [round(float(np.mean(v[k])), dg) for v in values if v[k]]
        best.append((min if low[k] else max)(ms) if ms else None)
    rows = []
    for lab, vals in zip(labels, values):
        row = [c(lab, al="l")]
        for k, v in enumerate(vals):
            row.append(c(fmt(v, dg), b=bool(v) and round(float(np.mean(v)), dg) == best[k]))
        rows.append(row)
    return rows


def mvals(t, m, key, seeds):
    sc = 1 if key == "hd95" else 100
    out = []
    for s in seeds:
        if key in ("precision", "specificity", "accuracy", "f2"):
            out.append(pf.EXTENDED[(t, m, s)][key] * sc)
        elif t == "EXT":
            out.append(pf.ext_metrics(m, s)[key] * sc)
        else:
            out.append(pf.fm(t, m, s)["test"][key] * sc)
    return out


def own_rows(cols, getter, lower=False, dg=2):
    """The four rows of this study: QENAS, nnU-Net, Attention U-Net, U-Net (mean ± SD of 3 runs each)."""
    labels = [NAME[m] for m in MAIN]
    vals = [[getter(t, m, SEEDS) for t in cols] for m in MAIN]
    return ranked_rows(labels, vals, lower, dg)


TABLES = []


def table(title, desc, header, rows, notes=(), widths=None, landscape=False):
    TABLES.append({"type": "table", "num": len(TABLES) + 1, "title": title, "desc": desc, "header": header,
                   "rows": rows, "notes": list(notes), "widths": widths, "landscape": landscape})


SPLIT_NOTE = (f"{DAG} Hu et al., Expert Syst. Appl. 289 (2025) 128338; {DDAG} Jha et al., IEEE Access 9 (2021) 40496 "
              "(Kvasir-SEG, 880 training / 120 test images). Published results use their own data splits and are "
              "therefore indicative only and not ranked.")
CVC_NOTE = ("Published CVC-ClinicDB results are omitted: they were obtained with a random frame-level split, which "
            "places frames of the same colonoscopy sequence in training and test; this study splits by sequence.")
OWN_NOTE = ("Models of this study: mean ± SD of three runs (seeds 45–47) on the identical split. Bold: best value of the "
            "column among the models of this study.")


# ------------------------------------------------------------------------------------------- tables
def t_datasets():
    rows = []
    info = [("BUSI", "Breast ultrasound", "Breast lesion (benign and malignant images)", "256 × 256",
             "image-level random split (no patient metadata); duplicate images with conflicting masks removed"),
            ("CVC", "Colonoscopy", "Polyp", "256 × 256",
             "sequence-level split: all frames of a colonoscopy sequence in one subset (29 sequences)"),
            ("IDRID", "Retinal fundus", "Optic disc", "320 × 512", "official training / test split; validation "
                                                                    "taken from the official training set"),
            ("KVASIR", "Colonoscopy", "Polyp", "256 × 256", "official 880 / 120 split; validation taken from the 880 "
                                                            "training images")]
    for ds, mod, tgt, res, how in info:
        n = {k: len(v) for k, v in pa.load(pa.exp(ds, "qenas", SEEDS[0]) / "split.json")["splits"].items()}
        rows.append([c(DSL[ds], al="l"), c(mod, al="l"), c(tgt, al="l"), c(n["train"]), c(n["val"]), c(n["test"]),
                     c(res), c(how, al="l")])
    rows.append([c("CVC-ClinicDB (external)", al="l"), c("Colonoscopy", al="l"), c("Polyp", al="l"), c("–"), c("–"),
                 c(612), c("256 × 256"), c("all 612 frames; the Kvasir-SEG models are applied unchanged", al="l")])
    table("Datasets, data splits and input resolution",
          "The split of every dataset is fixed (split seed 42) and identical for all models and all runs; the test "
          "images are used only once, for the final evaluation.",
          [c("Dataset", al="l"), c("Modality", al="l"), c("Target", al="l"), c("Train"), c("Validation"), c("Test"),
           c("Input (H × W)"), c("Split protocol", al="l")], rows, widths=[13, 11, 15, 6, 8, 6, 9, 32])


def t_models():
    p = {m: pf.fm("BUSI", m, SEEDS[0])["params_m"] for m in MAIN}
    qp = [pf.fm(ds, "qenas", s)["params_m"] for ds in DATASETS for s in SEEDS]
    rows = [[c("QENAS (ours)", al="l"), c("Training-free evolutionary NAS + successive halving", al="l"),
             c(f"{min(qp):.2f}–{max(qp):.2f}"),
             c("searched architecture; α selected on validation; 120 epochs, AdamW, poly schedule; "
               "best-validation checkpoint", al="l")],
            [c("nnU-Net", al="l"), c("Self-configuring U-Net (rule-based, no architecture search)", al="l"),
             c(f"{p['nnunet']:.2f}"), c("nnU-Net v2.8.1, 2d configuration, default plans, 100 epochs, fold = the same "
                                        "training / validation split, final checkpoint", al="l")],
            [c("Attention U-Net", al="l"), c("Hand-designed (Oktay et al., 2018)", al="l"),
             c(f"{p['attention_unet']:.2f}"), c("same protocol as the QENAS architecture: α selected on validation, "
                                                "120 epochs, AdamW, poly schedule", al="l")],
            [c("U-Net", al="l"), c("Hand-designed (Ronneberger et al., 2015)", al="l"), c(f"{p['unet']:.2f}"),
             c("same protocol as the QENAS architecture", al="l")]]
    table("Models compared in this study",
          "All models were trained and tested three times (seeds 45, 46 and 47) on the identical splits and images on "
          "the same GPU type (NVIDIA Tesla T4).",
          [c("Model", al="l"), c("Design", al="l"), c("Parameters (M)"), c("Training protocol", al="l")], rows,
          notes=["Parameters of QENAS: range over the architectures selected in the 12 runs (4 datasets × 3 seeds)."],
          widths=[13, 28, 11, 48])


def t_settings():
    cfg = pa.load(pa.exp("BUSI", "qenas", SEEDS[0]) / "config.json")["config"]
    s, ct, ft = cfg["search"], cfg["candidate_training"], cfg["final_training"]
    items = [
        ("Search space", "U-shaped network with 8 searchable positions (4 encoder, 4 decoder), 5 candidate blocks "
                         "(Residual, Dense, Inception, ConvNeXt, DARTS cell): 5⁸ = 390,625 architectures; 32 base "
                         "channels; skip connections with spatial–channel attention"),
        ("Search objectives", "f₁ = number of parameters; f₂ = 1 / (1 + ZiCo), ZiCo from 2 training batches; both "
                              "minimised; no network weight is trained"),
        ("Evolutionary search", f"NSGA-II, population {s['population_size']}, {s['generations']} generations after "
                                f"the initial population; binary tournament, uniform crossover "
                                f"(p = {s['crossover_prob']}), random-reset mutation (p = 1/8 per gene)"),
        ("Candidates", f"{cfg['pareto']['max_candidates']} architectures: final Pareto front, filled up in NSGA-II "
                       "survival order"),
        ("Successive halving", "16 × 5 → 8 × 10 → 4 × 20 epochs (160 candidate epochs), best half kept per rung; "
                               f"AdamW, learning rate {ct['lr']:g}, weight decay {ct['weight_decay']:g}, constant "
                               f"learning rate, batch size {ct['batch_size']} (IDRiD 4)"),
        ("Selection", "highest mean validation Dice over the last 5 epochs among the finalists that pass the gates "
                      "G1–G5 (finite loss, decreasing validation loss, no over-fitting, significant learning trend, "
                      "better than a trivial predictor)"),
        ("Loss function", "L = α·BCE + (1 − α)·soft-mIoU loss; α ∈ {0.2, 0.4, 0.5, 0.6, 0.8} chosen by the mean "
                          "validation Dice of 20-epoch runs"),
        ("Final training", f"{ft['epochs']} epochs, AdamW, learning rate {ft['lr']:g}, weight decay "
                           f"{ft['weight_decay']:g}, poly(0.9) schedule with {ft['warmup_epochs']} warm-up epoch, "
                           f"batch size {ft['batch_size']} (IDRiD 4); early stopping only on significant "
                           f"over-fitting; best-validation-Dice weights"),
        ("Augmentation", "horizontal flip (p = 0.5), vertical flip on colonoscopy (p = 0.5), affine (p = 0.5: "
                         "rotation ± 15°, scale 0.9–1.1, translation 5 %), brightness and contrast ± 10 %"),
        ("Evaluation", "single test pass, no test-time augmentation; Dice, mIoU, IoU, HD95, sensitivity, precision, "
                       "specificity, accuracy, F2; parameters, FLOPs, inference time (batch 1)"),
    ]
    table("Settings of QENAS", "One fixed configuration is used for all datasets; only the input resolution and the "
                               "physically meaningful flips depend on the dataset.",
          [c("Component", al="l"), c("Setting", al="l")], [[c(a, al="l"), c(b, al="l")] for a, b in items],
          widths=[18, 82])


def t_summary():
    keys = [("dice", False, "Dice (%)"), ("miou", False, "mIoU (%)"), ("hd95", True, "HD95 (px)"),
            ("sensitivity", False, "Sensitivity (%)"), ("precision", False, "Precision (%)"),
            ("params_m", True, "Params (M)"), ("gflops_per_image", True, "GFLOPs / image"),
            ("inference_ms_per_image", True, "Inference (ms)"), ("hours", True, "GPU hours")]

    def avg(m, seeds_of, key):
        per_ds = []
        for ds in DATASETS:
            seeds = seeds_of(ds)
            if key in ("params_m", "gflops_per_image", "inference_ms_per_image"):
                per_ds.append(np.mean([pf.fm(ds, m, s)[key] for s in seeds]))
            elif key == "hours":
                per_ds.append(np.mean([pf.state(ds, m, s)["total_seconds"] / 3600 for s in seeds]))
            else:
                per_ds.append(np.mean(mvals(ds, m, key, seeds)))
        return [float(np.mean(per_ds))]

    labels = [NAME[m] for m in MAIN]
    vals = [[avg(m, lambda ds: SEEDS, k) for k, _, _ in keys] for m in MAIN]
    rows = ranked_rows(labels, vals, lower=[lw for _, lw, _ in keys])
    table("Overall comparison: mean over the four datasets",
          "Each value is the mean over BUSI, CVC-ClinicDB, IDRiD and Kvasir-SEG of the per-dataset result. GPU hours: "
          "complete time to a trained model (QENAS: search, selection, loss-weight selection and final training).",
          [c("Model", al="l")] + [c(h) for _, _, h in keys], rows,
          notes=["Per dataset: mean of three runs. Bold: best value of the column (HD95, parameters, GFLOPs, "
                 "inference time and GPU hours: lowest)."],
          widths=[19] + [9] * 9, landscape=False)


METRICS = [
    ("dice", "Dice similarity coefficient (DSC, %)", False, 0, {"dice"}),
    ("miou", "Mean intersection over union (mIoU, %; mean of foreground and background IoU)", False, 1, set()),
    ("iou", "Foreground intersection over union (IoU / Jaccard, %)", False, None, set()),
    ("hd95", "95th-percentile Hausdorff distance (HD95, pixels; lower is better)", True, None, set()),
    ("sensitivity", "Sensitivity (recall, %)", False, None, set()),
    ("precision", "Precision (%)", False, None, set()),
    ("specificity", "Specificity (%)", False, None, set()),
    ("accuracy", "Pixel accuracy (%)", False, None, set()),
    ("f2", "F2-score (%)", False, None, set()),
]
JHA_IDX = {"dice": 0, "iou": 1, "f2": 2, "precision": 3, "sensitivity": 4, "accuracy": 5}


def t_metrics():
    header = [c("Model", al="l")] + [c(DSL[t] if t != "EXT" else "Kvasir-SEG → CVC-ClinicDB (external)") for t in TESTS]
    for key, title, lower, mg_idx, _ in METRICS:
        rows = own_rows(TESTS, lambda t, m, s, key=key: mvals(t, m, key, s), lower)
        pub = []
        if mg_idx is not None:
            for name, (_, per) in pa.MG.items():
                pub.append([c(name + " " + DAG, al="l")] + [c(f"{per[t][mg_idx]:.2f}") if t in ("BUSI", "IDRID") else
                                                             c("–") for t in TESTS])
        if key in JHA_IDX:
            for name, v in pa.JHA.items():
                pub.append([c(name.replace(" [JHA]", "") + " " + DDAG, al="l")] +
                           [c(f"{100 * v[JHA_IDX[key]]:.2f}") if t == "KVASIR" else c("–") for t in TESTS])
        notes = [OWN_NOTE]
        if pub:
            rows += [sub("Published results (other data splits)", len(TESTS) + 1)] + pub
            notes.append(SPLIT_NOTE)
            if mg_idx is not None:
                notes.append(CVC_NOTE)
        desc = ("Test-set results of all models; the last column applies the Kvasir-SEG models unchanged to all 612 "
                "CVC-ClinicDB images (external validation).")
        table(title, desc, header, rows, notes=notes, widths=[24, 14, 14, 14, 14, 20])


def t_cost():
    header = [c("Model", al="l")] + [c(DSL[d]) for d in DATASETS]
    defs = [("params_m", "Number of trainable parameters (M)", 2),
            ("gflops_per_image", "Computational cost (GFLOPs per image)", 2),
            ("inference_ms_per_image", "Inference time per image (ms; Tesla T4, batch size 1, median of 50 passes)", 1)]
    for key, title, dg in defs:
        rows = own_rows(DATASETS, lambda t, m, s, key=key: [pf.fm(t, m, x)[key] for x in s], lower=True, dg=dg)
        pub, notes = [], [OWN_NOTE.replace("Bold: best", "Bold: lowest")]
        if key != "inference_ms_per_image":
            for name, (_, per) in pa.MG.items():
                v = [(per[d][2] if key == "params_m" else per[d][3] / (4 if d == "IDRID" else 8)) if d in per else None
                     for d in DATASETS]
                pub.append([c(name + " " + DAG, al="l")] + [c("–") if x is None else c(f"{x:.{dg}f}") for x in v])
            if key == "params_m":
                for name, v in pa.JHA.items():
                    pub.append([c(name.replace(" [JHA]", "") + " " + DDAG, al="l")] +
                               [c(f"{v[6] / 1e6:.2f}") if d == "KVASIR" else c("–") for d in DATASETS])
            if key == "gflops_per_image":
                notes.append(f"FLOPs counted with thop; published values {DAG} are per training batch in the source "
                             "(batch 8, IDRiD 4) and were divided by the batch size.")
        else:
            for name, fps in pa.JHA_FPS.items():
                pub.append([c(name.replace(" [JHA]", "") + " " + DDAG, al="l")] +
                           [c(f"{1000 / fps:.1f}") if d == "KVASIR" else c("–") for d in DATASETS])
            notes.append(f"Published times {DDAG}: 1000 / reported frames per second, NVIDIA RTX 2080 Ti, 512 × 512 "
                         "input; not directly comparable with the Tesla T4 measurements of this study.")
        if pub:
            rows += [sub("Published results", len(DATASETS) + 1)] + pub
            notes.append(SPLIT_NOTE.split(" Published results use")[0])
        desc = ("Lower is better. Measured for the trained network of every run; parameters and FLOPs depend only on "
                "the architecture and the input size." if key != "inference_ms_per_image" else
                "Lower is better. Forward pass of one image on an NVIDIA Tesla T4 (batch size 1, median of 50 timed "
                "passes after warm-up).")
        table(title, desc, header, rows, notes=notes, widths=[30, 17, 17, 17, 19])


def t_search_time():
    ds = "IDRID"
    rows = [[c(n.replace("NASUnet", "NAS-Unet").replace("HCT-net", "HCT-Net"), al="l"), c(st, al="l"), c(t),
             c("RTX 4070 Ti Super")] for n, st, t in pa.MG_TABLE8]
    rows = [r for r in rows]
    pm = [np.mean([pf.qenas_stage_hours(ds, s)[k] for s in SEEDS]) for k in range(4)]

    def q(p):
        return f"({p[0]:.2f} + {p[1]:.2f} + {p[2]:.2f}) h = {sum(p[:3]):.2f} h ({60 * sum(p[:3]):.0f} min)"
    rows.append([c("QENAS (ours), mean of 3 runs", al="l", b=True),
                 c("Training-free evolution + successive halving", al="l"), c(q(pm), b=True), c("Tesla T4")])
    table("Search time of NAS methods on IDRiD",
          "Time of the architecture search algorithm, in the format of Hu et al. (2025, Table 8); training the final "
          "network is not included for any method.",
          [c("Approach", al="l"), c("Search strategy", al="l"), c("Search time"), c("GPU")], rows,
          notes=["QENAS: (training-free NSGA-II search + successive-halving selection + loss-weight selection). "
                 "Published values from Hu et al. (2025), Table 8. The Tesla T4 is a slower GPU than the RTX 4070 Ti "
                 "Super. Bold: shortest search time."], widths=[24, 27, 31, 18])


def t_total_time():
    header = [c("Model", al="l")] + [c(DSL[d]) for d in DATASETS]
    rows = own_rows(DATASETS, lambda t, m, s: [pf.state(t, m, x)["total_seconds"] / 3600 for x in s], lower=True)
    srch_m = [[sum(pf.qenas_stage_hours(d, s)[:3]) for s in SEEDS] for d in DATASETS]
    rows += [sub("QENAS architecture search and selection only (without final training)", len(DATASETS) + 1),
             [c("QENAS", al="l")] + [c(fmt(v)) for v in srch_m]]
    table("Total computation time to a trained model (GPU hours, Tesla T4)",
          "Complete wall-clock GPU time of one run, from the data split to the tested model: for QENAS the search, "
          "successive halving, loss-weight selection and final training; for the other models their training "
          "(including nnU-Net's self-configuration and preprocessing). Lower is better.",
          header, rows, notes=[OWN_NOTE.replace("Bold: best", "Bold: lowest")], widths=[30, 17, 17, 17, 19])


def t_stats():
    rows_s, ranks = pf.comparison_stats(qenas_mean=True)
    rows, cur = [], None
    for t, m, r, pad in rows_s:
        if t != cur:
            rows.append(sub(DSL[t] + (" (external)" if t == "EXT" else "") + f" — {r['n']} test images", 9))
            cur = t
        a, b = 100 * r["a"], 100 * r["b"]
        sig = pad < 0.05
        verdict = ("QENAS better" if r["diff"] > 0 else f"{NAME[m]} better") if sig else "no significant difference"
        rows.append([c(f"QENAS vs {NAME[m]}", al="l"), c(f"{a:.2f}", b=a >= b), c(f"{b:.2f}", b=b > a),
                     c(f"{100 * r['diff']:+.2f}"), c(f"[{100 * r['lo']:+.2f}, {100 * r['hi']:+.2f}]"),
                     c(pa.fmt_p(r["p"])), c(pa.fmt_p(pad), b=sig), c(f"{r['better']} / {r['worse']}"),
                     c(verdict, al="l")])
    table("Paired statistical comparison of the per-image test Dice",
          "Two-sided Wilcoxon signed-rank test on the same test images, Holm-corrected over the three comparisons of a "
          "test set; Δ = QENAS − compared model with a 95 % bootstrap confidence interval (10,000 resamples).",
          [c("Comparison", al="l"), c("Dice QENAS (%)"), c("Dice model (%)"), c("Δ Dice (pp)"), c("95 % CI (pp)"),
           c("p"), c("p (Holm)"), c("Images better / worse"), c("Result", al="l")], rows,
          notes=["Per-image Dice of every model averaged over its three runs. Bold: higher Dice of the pair, and "
                 "p (Holm) < 0.05."], widths=[20, 9, 9, 8, 14, 7, 8, 10, 15], landscape=False)
    rows = []
    for t in TESTS:
        r = ranks[t]
        rk = [r["ranks"][m] for m in MAIN]
        rows.append([c(DSL[t] if t != "EXT" else "Kvasir-SEG → CVC-ClinicDB", al="l"), c(r["n"]),
                     c(pa.fmt_p(r["p"]))] + [c(f"{x:.2f}", b=x == min(rk)) for x in rk] + [c(f"{r['cd']:.2f}")])
    table("Mean ranks of all models (Friedman test with Nemenyi critical difference)",
          "Models ranked per test image by Dice (1 = best); two models differ significantly (α = 0.05) when their mean "
          "ranks differ by more than the critical difference (CD).",
          [c("Test set", al="l"), c("Images"), c("Friedman p")] + [c(f"Rank {pf.NAME[m]}") for m in MAIN] + [c("CD")],
          rows, notes=["Per-image Dice of every model averaged over its three runs. Bold: best (lowest) mean rank."], widths=[24, 8, 10, 11, 11, 13, 11, 8])


def t_ablation_matrix():
    """Configurations with / without each component (as in Hu et al. 2025): test Dice, mean ± SD of three runs."""
    yes, no = "✓", "✗"
    confs = [("qenas", yes, yes, "QENAS (full)"),
             ("qenas_fixed_alpha", yes, no, "without α search (fixed α = 0.5)"),
             ("qenas_random_pick", no + " (random candidate)", yes, "without Dice-based selection"),
             ("qenas_zico_pick", no + " (max ZiCo)", yes, "without Dice-based selection")]

    vals = [[mvals(t, m, "dice", SEEDS) if pf.pa.exp(t, m, SEEDS[0]).joinpath("final_metrics.json").exists()
             else None for t in DATASETS] for m, *_ in confs]
    best = [max(round(float(np.mean(v[k])), 2) for v in vals if v[k]) for k in range(len(DATASETS))]
    rows = []
    for (m, sel, al, lab), v in zip(confs, vals):
        rows.append([c(lab, al="l"), c(sel), c(al)] +
                    [c(fmt(x or []), b=bool(x) and round(float(np.mean(x)), 2) == best[k]) for k, x in enumerate(v)])
    table("Ablation study: test Dice of QENAS with and without each component (%)",
          "Each row removes one component of QENAS while the training-free NSGA-II search, candidates and seeds stay "
          "identical. Values: test Dice, mean ± SD of three runs (seeds 45–47).",
          [c("Configuration", al="l"), c("Dice-based selection (successive halving)"), c("α search")] +
          [c(DSL[t]) for t in DATASETS], rows,
          notes=["–: not run (selection ablations were run on BUSI and Kvasir-SEG). The training-free NSGA-II search "
                 "is compared with random search by hypervolume (preceding table), as random search does not train a "
                 "network. Bold: best value of the column. Paired per-image tests: ablation table above."],
          widths=[24, 18, 9, 12, 12, 12, 13])


def t_ablation():
    ab, _ = pa.ablations()
    rows = []
    lab = {"Loss-weight search (α grid) vs fixed α = 0.5": "Loss-weight search vs fixed α = 0.5",
           "Selection vs random candidate": "Learning-capability selection vs random candidate",
           "Selection vs zero-shot pick (max ZiCo)": "Learning-capability selection vs zero-shot pick (max ZiCo)"}
    order = sorted(ab["v2"]["rows"], key=lambda r: (list(lab).index(r[2]), list(DSL.values()).index(r[1])))
    cur = None
    for r in order:
        if r[2] != cur:
            rows.append(sub(lab[r[2]], 7))
            cur = r[2]
        a, b = float(r[3]), float(r[4])
        sig = r[8] == "<1e-4" or float(r[8]) < 0.05
        rows.append([c(r[1], al="l"), c(r[3], b=a >= b), c(r[4], b=b > a), c(r[5]), c(r[6]), c(r[8], b=sig),
                     c(("QENAS component helps" if a > b else "variant better") if sig else "no significant difference",
                       al="l")])
    table("Ablation study of the QENAS components (test Dice, %)",
          "Each component of QENAS is replaced by a simpler alternative while everything else stays identical (same "
          "search, candidates and seeds). Values: per-image test Dice averaged over three runs; paired Wilcoxon test, "
          "Holm-corrected within each question and dataset.",
          [c("Dataset", al="l"), c("QENAS"), c("Variant"), c("Δ (pp)"), c("95 % CI (pp)"), c("p (Holm)"),
           c("Result", al="l")], rows,
          notes=["Selection ablations were run on BUSI and Kvasir-SEG. Bold: higher Dice of the pair, and p (Holm) < "
                 "0.05."], widths=[16, 10, 10, 9, 18, 10, 27])
    t7 = pa.load(pf.RES / "analysis" / "T7_search_vs_random.json")
    rows = []
    for r in t7:
        hn, hr = float(r["HV NSGA-II"].split(" ")[0]), float(r["HV random search"].split(" ")[0])
        rows.append([c(r["Dataset"].replace(" (optic disc)", ""), al="l"), c(r["Evaluations"]),
                     c(r["HV NSGA-II"], b=hn >= hr), c(r["HV random search"], b=hr > hn), c(r["NSGA-II better in seeds"])])
    table("Search quality: NSGA-II versus random search with the same budget",
          "Hypervolume (higher is better) of the non-dominated set of all architectures evaluated by each search, both "
          "normalised by the union of the two archives (objectives: parameters and ZiCo; training-free).",
          [c("Dataset", al="l"), c("Evaluated architectures"), c("Hypervolume NSGA-II"), c("Hypervolume random search"),
           c("NSGA-II better (runs)")], rows, notes=["Mean ± SD of three runs. Bold: higher hypervolume."],
          widths=[20, 18, 22, 22, 18])


# ------------------------------------------------------------------------------------------- main
def main():
    pf.EXTENDED = pa.extended_metrics()
    t_datasets()
    t_models()
    t_settings()
    t_summary()
    t_metrics()
    t_cost()
    t_search_time()
    t_total_time()
    t_stats()
    t_ablation()
    t_ablation_matrix()
    intro = [
        "Tables of the final QENAS (study v2). Every value was computed from the saved results of the study; "
        "run-dependent results (search progress, candidate training, selected architectures) are shown as figures.",
        f"Conventions. Every model of this study (QENAS, nnU-Net, Attention U-Net, U-Net): mean ± SD of three runs "
        f"(seeds 45–47) on the identical split. Bold: best value of a column among the models of this "
        f"study. Published results ({DAG} Hu et al. 2025; {DDAG} Jha et al. 2021) use other data splits and are listed "
        f"for reference only.",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    spec = {"title": "QENAS 3.0 — Tables", "intro": intro, "tables": TABLES}
    with open(OUT / "tables_spec.json", "w", encoding="utf-8") as fh:
        json.dump(spec, fh, ensure_ascii=False, indent=1)
    print(f"[tables] {len(TABLES)} tables -> {OUT / 'tables_spec.json'}")


if __name__ == "__main__":
    main()
