"""Publication figures of QENAS 3.0 (study v2).

    python scripts/paper_figures.py

Writes results/figures/ (600 dpi PNG, Times New Roman, one font size, no titles; the file name is the title):

1 Comparison with other models/  run-independent comparisons. QENAS: its best run per dataset (highest test Dice of
                                 the 3 seeds, the user's choice); every other model: mean (± SD) of its 3 runs.
2 QENAS stages/                  one figure per stage, from the best QENAS run (BUSI, seed 47).
3 Architecture/                  framework, search space and blocks, the architecture found in the best run.
4 Qualitative results/           masks of all models on test images, per dataset (QENAS best run).
review/                          further figures for the paper, to be checked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from scipy.stats import friedmanchisquare, kendalltau, rankdata, spearmanr, wilcoxon  # noqa: E402

import paper_analysis as pa  # noqa: E402

RES = pa.RES
OUT = RES / "figures"
D_CMP = OUT / "1 Comparison with other models"
D_STG = OUT / "2 QENAS stages"
D_ARC = OUT / "3 Architecture"
D_QUA = OUT / "4 Qualitative results"
D_REV = OUT / "review"

DATASETS, SEEDS, MAIN = pa.DATASETS, pa.SEEDS, pa.MAIN
DSL = {"BUSI": "BUSI", "CVC": "CVC-ClinicDB", "IDRID": "IDRiD", "KVASIR": "Kvasir-SEG",
       "EXT": "Kvasir-SEG → CVC-ClinicDB"}
SHORT = {"BUSI": "BUSI", "CVC": "CVC", "IDRID": "IDRiD", "KVASIR": "Kvasir", "EXT": "Kvasir→CVC"}
TESTS = DATASETS + ["EXT"]

# ------------------------------------------------------------------------------------------- style
FS = 9                                   # the one font size of every figure (pt, at print size)
W1, W2 = 3.5, 7.16                       # IEEE single / double column width (in)
INK, INK2, MUTED, GRID, AXIS = "#000000", "#3a3a3a", "#8c8c8c", "#e3e3e3", "#4d4d4d"
COLOR = {"qenas": "#2a78d6", "nnunet": "#eb6834", "attention_unet": "#1baf7a", "unet": "#eda100"}
MARK = {"qenas": "o", "nnunet": "s", "attention_unet": "^", "unet": "D"}
NAME = {"qenas": "QENAS", "nnunet": "nnU-Net", "attention_unet": "Attention U-Net", "unet": "U-Net"}
LEG = {"qenas": "QENAS", "nnunet": "nnU-Net", "attention_unet": "Attention U-Net", "unet": "U-Net"}
BLUES = ["#0d366b", "#2a78d6", "#7fb1ee", "#c4dbf7"]          # QENAS stages, dark -> light
BLOCK_NAME = {0: "Residual", 1: "Dense", 2: "Inception", 3: "ConvNeXt", 4: "DARTS cell"}
BLOCK_SHORT = {0: "RES", 1: "DEN", 2: "INC", 3: "CNX", 4: "DRT"}
BLOCK_COLOR = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#eda100", 4: "#7b5cc4"}
BLOCK_FILL = {0: "#dbe8f9", 1: "#fce2d6", 2: "#d4f1e6", 3: "#fcefcc", 4: "#e6dff5"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "legend.title_fontsize": FS, "figure.titlesize": FS,
    "figure.dpi": 150, "savefig.dpi": 600, "pdf.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.columnspacing": 1.2,
    "lines.linewidth": 1.2, "errorbar.capsize": 2,
})


def save(fig, folder: Path, title: str):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{title}.png", dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print("  ", (folder / f"{title}.png").relative_to(OUT))


def caption(i, text):
    return f"({'abcdefghij'[i]}) {text}"


# ------------------------------------------------------------------------------------------- data
def load(p):
    return pa.load(p)


def fm(ds, m, s):
    return load(pa.exp(ds, m, s) / "final_metrics.json")


def best_seed(ds):
    return max(SEEDS, key=lambda s: fm(ds, "qenas", s)["test"]["dice"])


BEST = {ds: best_seed(ds) for ds in DATASETS}
BEST["EXT"] = BEST["KVASIR"]            # the external test applies the best Kvasir-SEG model unchanged
STAGE_RUN = ("BUSI", BEST["BUSI"])      # the run shown in the stage figures


def representative_seed(ds, m):
    """Run of a baseline whose test Dice is closest to its 3-run mean (used for its masks)."""
    v = {s: fm(ds, m, s)["test"]["dice"] for s in SEEDS}
    mu = np.mean(list(v.values()))
    return min(SEEDS, key=lambda s: abs(v[s] - mu))


def ext_metrics(m, s):
    return load(pa.exp("KVASIR", m, s) / "external_CVC" / "external_metrics.json")["metrics"]


EXTENDED = None


def metric_values(test, m, key):
    """QENAS: [best-run value]; others: the 3 run values. Scaled to % where the metric is a ratio."""
    sc = 1 if key == "hd95" else 100
    seeds = [BEST[test]] if m == "qenas" else SEEDS
    out = []
    for s in seeds:
        if key in ("precision", "specificity", "accuracy", "f2"):
            out.append(EXTENDED[(test, m, s)][key] * sc)
        elif test == "EXT":
            out.append(ext_metrics(m, s)[key] * sc)
        else:
            out.append(fm(test, m, s)["test"][key] * sc)
    return out


def cost_values(ds, m, key):
    seeds = [BEST[ds]] if m == "qenas" else SEEDS
    return [fm(ds, m, s)[key] for s in seeds]


def per_image(test, m, qenas_mean=False):
    """sample_id -> Dice: QENAS best run (qenas_mean: mean over its 3 runs); others mean over their 3 runs."""
    if test == "EXT":
        if m == "qenas" and not qenas_mean:
            return {r["sample_id"]: float(r["dice"]) for r in
                    pa.read_csv(pa.exp("KVASIR", m, BEST["EXT"]) / "external_CVC" / "external_per_image.csv")}
        return pa.per_image_ext(m)
    if m == "qenas" and not qenas_mean:
        return {r["sample_id"]: float(r["dice"]) for r in pa.read_csv(pa.exp(test, m, BEST[test]) / "test_per_image.csv")}
    return pa.per_image(test, m)


def state(ds, m, s):
    return load(pa.exp(ds, m, s) / "state.json")


def qenas_stage_hours(ds, s):
    st = state(ds, "qenas", s)
    g = lambda k: st["stages"].get(k, {}).get("seconds", 0.0) / 3600   # noqa: E731
    parts = [g("search"), g("pareto") + g("candidates") + g("selection"), g("loss_weights")]
    parts.append(st["total_seconds"] / 3600 - sum(parts))
    return parts


def total_hours(ds, m):
    seeds = [BEST[ds]] if m == "qenas" else SEEDS
    return [state(ds, m, s)["total_seconds"] / 3600 for s in seeds]


def holm(p):
    return pa.holm(p)


def paired(a, b):
    return pa.paired(a, b)


# ------------------------------------------------------------------------------------------- helpers
def model_legend(ax_or_fig, ms=("qenas", "nnunet", "attention_unet", "unet"), kind="bar", **kw):
    if kind == "bar":
        h = [Patch(facecolor=COLOR[m], edgecolor="none") for m in ms]
    else:
        h = [Line2D([], [], marker=MARK[m], color=COLOR[m], ls="", ms=5) for m in ms]
    return ax_or_fig.legend(h, [LEG[m] for m in ms], **kw)


def errbar_note():
    return Line2D([], [], color=INK2, marker="|", ls="", ms=7, mew=0.8)


def label_points(ax, pts, own):
    """Greedy, deterministic labelling of the points pts = [(x, y, name)]: every label takes the first candidate
    position (nearest first) that overlaps no marker, no error bar, no other label and stays inside the axes.
    own = [(x, y, sd)] markers of this study's models (obstacles)."""
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox
    fig = ax.figure
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    s = 5.0 * fig.dpi / 72
    occupied = []
    for x, y, sd in own:
        px, py = ax.transData.transform((x, y))
        lo = ax.transData.transform((x, y - sd))[1]
        hi = ax.transData.transform((x, y + sd))[1]
        occupied.append(Bbox([[px - s, min(py - s, lo - 3)], [px + s, max(py + s, hi + 3)]]))
    for x, y, _ in pts:
        px, py = ax.transData.transform((x, y))
        occupied.append(Bbox([[px - s * 0.8, py - s * 0.8], [px + s * 0.8, py + s * 0.8]]))
    cands = []
    for d in (4, 9, 15, 22, 30, 40):
        cands += [(d, 0.4 * d, "left", "bottom"), (d, -0.4 * d, "left", "top"), (-d, 0.4 * d, "right", "bottom"),
                  (-d, -0.4 * d, "right", "top"), (0, d + 2, "center", "bottom"), (0, -d - 2, "center", "top"),
                  (d, 0, "left", "center"), (-d, 0, "right", "center")]
    for x, y, name in sorted(pts, key=lambda p: -p[1]):
        for dx, dy, ha, va in cands:
            lead = abs(dx) + abs(dy) > 10
            t = ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=INK2,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=1, shrinkB=4) if lead
                            else None)
            t.update_positions(r)
            bb = Text.get_window_extent(t, r)
            ok = (axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1
                  and not any(bb.overlaps(o) for o in occupied))
            if ok:
                occupied.append(bb)
                break
            t.remove()
        else:
            print("   [label] no free position for", name)
            ax.annotate(name, (x, y), xytext=(4, 2), textcoords="offset points", color=INK2)


def plain_log(ax, axis="x", subs=(1, 2, 5)):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = getattr(ax, f"{axis}axis")
    a.set_major_locator(LogLocator(base=10, subs=subs))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


EXT_TICK = "Kvasir-SEG →\nCVC-ClinicDB\n(external test)"


# =========================================================================================== 1 comparison figures
def fig_dice_bars():
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    w = 0.2
    for j, m in enumerate(MAIN):
        for i, t in enumerate(TESTS):
            v = metric_values(t, m, "dice")
            mu = np.mean(v)
            sd = np.std(v, ddof=1) if len(v) > 1 else None
            x = i + (j - 1.5) * w
            ax.bar(x, mu, w * 0.92, color=COLOR[m], edgecolor="none", zorder=2)
            if sd is not None:
                ax.errorbar(x, mu, yerr=sd, color=INK2, lw=0.7, capsize=1.6, zorder=3)
            ax.text(x, mu + (sd or 0) + 2.6, f"{mu:.1f}", ha="center", va="bottom", rotation=90, fontsize=FS)
    ax.set_xticks(range(len(TESTS)), [DSL[t] if t != "EXT" else EXT_TICK for t in TESTS])
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Test Dice (%)")
    ax.set_xlabel("Test set")
    h = [Patch(facecolor=COLOR[m]) for m in MAIN] + [errbar_note()]
    ax.legend(h, [LEG[m] for m in MAIN] + ["± SD over 3 runs"], ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), handlelength=1.2)
    save(fig, D_CMP, "Test Dice of QENAS and the compared models on every test set")


METRICS8 = [("miou", "Mean IoU (%)"), ("iou", "Foreground IoU (%)"), ("hd95", "HD95 (pixels, lower is better)"),
            ("sensitivity", "Sensitivity (%)"), ("precision", "Precision (%)"), ("specificity", "Specificity (%)"),
            ("accuracy", "Pixel accuracy (%)"), ("f2", "F2-score (%)")]


def fig_all_metrics():
    fig, axs = plt.subplots(4, 2, figsize=(W2, 8.2), sharex=True, constrained_layout=True)
    for ax, (key, lab) in zip(axs.T.flat, METRICS8):
        for j, m in enumerate(MAIN):
            xs, mus, sds = [], [], []
            for i, t in enumerate(TESTS):
                v = metric_values(t, m, key)
                xs.append(i + (j - 1.5) * 0.17)
                mus.append(np.mean(v))
                sds.append(np.std(v, ddof=1) if len(v) > 1 else 0)
            ax.errorbar(xs, mus, yerr=sds if m != "qenas" else None, fmt=MARK[m], color=COLOR[m], ms=4.2,
                        mew=0, elinewidth=0.8, capsize=1.6, zorder=3)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
        for i in range(len(TESTS) - 1):
            ax.axvline(i + 0.5, color=GRID, lw=0.5, zorder=0)
    for ax in axs[-1]:
        ax.set_xticks(range(len(TESTS)), [SHORT[t] for t in TESTS])
        ax.set_xlabel("Test set")
    model_legend(fig, kind="dot", ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, D_CMP, "Segmentation metrics of QENAS and the compared models")


def published_points(ds, kind):
    """[(name, x, Dice)] of the published methods (other data splits) for a dataset."""
    pts = []
    if ds in ("BUSI", "IDRID"):                          # CVC-ClinicDB: published values use a random frame split
        for name, (_, per) in pa.MG.items():
            if ds in per:
                v = per[ds]
                pts.append((name, v[2] if kind == "params" else v[3] / (4 if ds == "IDRID" else 8), v[0]))
    if ds == "KVASIR" and kind == "params":
        short = {"U-Net [JHA]": "U-Net", "DoubleUNet (VGG19)": "DoubleUNet", "PSPNet (ResNet50)": "PSPNet",
                 "DeepLabv3+ (ResNet50)": "DeepLabv3+ (R50)", "DeepLabv3+ (ResNet101)": "DeepLabv3+ (R101)",
                 "U-Net (ResNet34)": "U-Net (R34)", "FCN8 (VGG16)": "FCN8"}
        for name, v in pa.JHA.items():
            pts.append((short.get(name, name), v[6] / 1e6, 100 * v[0]))
    return pts


def fig_dice_vs_cost(kind):
    key = "params_m" if kind == "params" else "gflops_per_image"
    xlab = "Parameters (M, log scale)" if kind == "params" else "GFLOPs per image (log scale)"
    fig, axs = plt.subplots(2, 2, figsize=(W2, 6.6), constrained_layout=True)
    for i, (ax, ds) in enumerate(zip(axs.flat, DATASETS)):
        pts = published_points(ds, kind)
        if pts:
            ax.scatter([p[1] for p in pts], [p[2] for p in pts], marker="o", s=22, facecolor="white",
                       edgecolor=MUTED, lw=0.9, zorder=3)
        own = []
        for m in MAIN:
            x = np.mean(cost_values(ds, m, key))
            y = metric_values(ds, m, "dice")
            sd = np.std(y, ddof=1) if len(y) > 1 else 0.0
            ax.errorbar([x], [np.mean(y)], yerr=[sd] if len(y) > 1 else None, fmt=MARK[m],
                        color=COLOR[m], ms=6.5 if m == "qenas" else 5.5, mew=0.6, mec="white", elinewidth=0.8,
                        capsize=2, zorder=5)
            own.append((x, np.mean(y), sd))
        ax.set_xscale("log")
        plain_log(ax, "x", subs=(1, 3) if kind == "params" else (1, 2, 5))
        ax.grid(axis="x", which="major", visible=True)
        ax.set_ylabel("Test Dice (%)")
        ax.set_xlabel(f"{xlab}\n{caption(i, DSL[ds])}", linespacing=1.7)
        ax.margins(x=0.15, y=0.14)
    fig.canvas.draw()                                  # final layout first, then place labels in display space
    fig.set_layout_engine("none")
    for ax, ds in zip(axs.flat, DATASETS):
        pts = published_points(ds, kind)
        if pts:
            own = [(np.mean(cost_values(ds, m, key)), np.mean(metric_values(ds, m, "dice")),
                    np.std(metric_values(ds, m, "dice"), ddof=1) if m != "qenas" else 0.0) for m in MAIN]
            label_points(ax, [(p[1], p[2], p[0]) for p in pts], own)
    h = [Line2D([], [], marker=MARK[m], color=COLOR[m], ls="", ms=5.5) for m in MAIN]
    h.append(Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=5))
    lab = [LEG[m] for m in MAIN] + ["published results (other data splits)"]
    fig.legend(h, lab, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, D_CMP, "Test Dice versus number of parameters of all models" if kind == "params"
         else "Test Dice versus computational cost (GFLOPs) of all models")


def fig_search_time():
    ds = "IDRID"
    parts = qenas_stage_hours(ds, BEST[ds])
    q_search = sum(parts[:3])
    q_total = sum(parts)
    items = [("NAS-Unet", 3), ("HCT-Net", 3), ("UNAS-Net", 19), ("Genetic U-Net", 4), ("MedUNAS", 11),
             ("BiX-NAS", 4), ("Mixed-GGNAS (without ViT)", 15), ("Mixed-GGNAS (with ViT)", 26)]
    items = sorted(items, key=lambda kv: kv[1])
    items = [("QENAS: search and selection", q_search), ("QENAS: including final training", q_total)] + items
    fig, ax = plt.subplots(figsize=(W1, 3.1))
    ys = np.arange(len(items))[::-1]
    for y, (k, v) in zip(ys, items):
        ax.barh(y, v, 0.66, color=COLOR["qenas"] if k.startswith("QENAS") else "#b5b5b5", edgecolor="none")
        ax.text(v * 1.1, y, f"{v * 60:.0f} min" if v < 1 else f"{v:.0f} h" if v >= 2 else f"{v:.1f} h",
                va="center", fontsize=FS)
    ax.set_yticks(ys, [k for k, _ in items])
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    plain_log(ax, "x", subs=(1,))
    ax.set_xlabel("Search time on IDRiD (hours, log scale)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.legend([Patch(color=COLOR["qenas"]), Patch(color="#b5b5b5")],
              ["QENAS (Tesla T4)", "published NAS methods (RTX 4070 Ti Super)"], loc="upper center",
              bbox_to_anchor=(0.32, 1.17), ncol=1)
    save(fig, D_CMP, "Search time of QENAS and published NAS methods on IDRiD")


def fig_total_time():
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    w = 0.2
    stage_lab = ["QENAS: training-free search", "QENAS: successive halving and selection",
                 "QENAS: loss-weight (α) selection", "QENAS: final training and testing"]
    for i, ds in enumerate(DATASETS):
        bottom = 0.0
        for k, v in enumerate(qenas_stage_hours(ds, BEST[ds])):
            ax.bar(i - 1.5 * w, v, w * 0.92, bottom=bottom, color=BLUES[k], edgecolor="white", lw=0.5,
                   label=stage_lab[k] if i == 0 else None, zorder=2)
            bottom += v
        ax.text(i - 1.5 * w, bottom + 0.1, f"{bottom:.2f}", ha="center", va="bottom", rotation=90)
        for j, m in enumerate(MAIN[1:], 1):
            v = total_hours(ds, m)
            mu, sd = np.mean(v), np.std(v, ddof=1)
            ax.bar(i + (j - 1.5) * w, mu, w * 0.92, color=COLOR[m], edgecolor="none",
                   label=LEG[m] if i == 0 else None, zorder=2)
            ax.errorbar(i + (j - 1.5) * w, mu, yerr=sd, color=INK2, lw=0.7, capsize=1.6, zorder=3)
            ax.text(i + (j - 1.5) * w, mu + sd + 0.14, f"{mu:.2f}", ha="center", va="bottom", rotation=90)
    ax.set_xticks(range(4), [DSL[d] for d in DATASETS])
    ax.set_xlabel("Dataset")
    ax.set_ylabel("GPU hours to a trained model")
    ax.set_ylim(0, 6.2)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h + [errbar_note()], l + ["± SD over 3 runs"], ncol=2, loc="upper left", handlelength=1.2)
    save(fig, D_CMP, "Total computation time of QENAS and the compared models")


def fig_cost_bars():
    keys = [("params_m", "Parameters (M)", "{:.1f}"), ("gflops_per_image", "GFLOPs per image", "{:.1f}"),
            ("inference_ms_per_image", "Inference time per image (ms)", "{:.1f}")]
    fig, axs = plt.subplots(3, 1, figsize=(W2, 6.6), sharex=True, constrained_layout=True)
    w = 0.2
    for ax, (key, lab, f) in zip(axs, keys):
        top = 0
        for i, ds in enumerate(DATASETS):
            for j, m in enumerate(MAIN):
                v = cost_values(ds, m, key)
                mu = np.mean(v)
                sd = np.std(v, ddof=1) if len(v) > 1 else 0
                x = i + (j - 1.5) * w
                ax.bar(x, mu, w * 0.92, color=COLOR[m], edgecolor="none", zorder=2)
                if len(v) > 1 and sd > 0:
                    ax.errorbar(x, mu, yerr=sd, color=INK2, lw=0.7, capsize=1.6, zorder=3)
                ax.text(x, mu + sd, " " + f.format(mu), ha="center", va="bottom", rotation=90)
                top = max(top, mu + sd)
        ax.set_ylim(0, top * 1.38)
        ax.set_ylabel(lab)
    axs[-1].set_xticks(range(4), [DSL[d] for d in DATASETS])
    axs[-1].set_xlabel("Dataset")
    model_legend(fig, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, D_CMP, "Model size, computational cost and inference time of all models")


def fig_dice_vs_time():
    fig, ax = plt.subplots(figsize=(W2, 3.2))
    dmark = {"BUSI": "o", "CVC": "s", "IDRID": "^", "KVASIR": "D"}
    for ds in DATASETS:
        for m in MAIN:
            x = np.mean(total_hours(ds, m))
            y = np.mean(metric_values(ds, m, "dice"))
            ax.scatter([x], [y], marker=dmark[ds], s=38, color=COLOR[m], edgecolor="white", lw=0.5, zorder=3)
    ax.set_xscale("log")
    plain_log(ax, "x", subs=(1, 2, 5))
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("GPU hours to a trained model (log scale)")
    ax.set_ylabel("Test Dice (%)")
    h = [Line2D([], [], marker="o", color=COLOR[m], ls="", ms=5.5) for m in MAIN]
    h += [Line2D([], [], marker=dmark[d], color=INK2, ls="", ms=5.5) for d in DATASETS]
    ax.legend(h, [LEG[m] for m in MAIN] + [DSL[d] for d in DATASETS], ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), columnspacing=1.0)
    save(fig, D_CMP, "Test Dice versus total computation time of all models")


def fig_distribution():
    fig, axs = plt.subplots(1, 5, figsize=(W2, 3.0), sharey=True, constrained_layout=True)
    for i, (ax, t) in enumerate(zip(axs, TESTS)):
        data = [np.array(list(per_image(t, m).values())) * 100 for m in MAIN]
        vp = ax.violinplot(data, showextrema=False, widths=0.85)
        for b, m in zip(vp["bodies"], MAIN):
            b.set_facecolor(COLOR[m])
            b.set_alpha(0.35)
            b.set_edgecolor("none")
        ax.boxplot(data, widths=0.22, showfliers=False, medianprops=dict(color=INK, lw=0.9),
                   whiskerprops=dict(color=INK2, lw=0.7), capprops=dict(color=INK2, lw=0.7),
                   boxprops=dict(color=INK2, lw=0.7))
        ax.scatter(range(1, 5), [d.mean() for d in data], marker="D", s=10, color="white", edgecolor=INK, lw=0.6,
                   zorder=4)
        ax.set_xticks(range(1, 5), [])
        ax.tick_params(axis="x", length=0)
        ax.set_xlabel(caption(i, DSL[t] if t != "EXT" else "Kvasir-SEG →\nCVC-ClinicDB"))
        ax.set_ylim(-3, 103)
    axs[0].set_ylabel("Per-image test Dice (%)")
    h = [Patch(facecolor=COLOR[m], alpha=0.5) for m in MAIN] + [
        Line2D([], [], color=INK, lw=0.9), Line2D([], [], marker="D", color="white", mec=INK, ls="", ms=4)]
    fig.legend(h, [LEG[m] for m in MAIN] + ["median", "mean"], ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.2, columnspacing=0.9)
    save(fig, D_CMP, "Distribution of per-image test Dice of all models")


def comparison_stats(qenas_mean=False):
    """Paired Wilcoxon (Holm within a test set) and Friedman ranks: QENAS best run (qenas_mean: mean of its 3 runs)
    vs the others (mean of 3 runs)."""
    rows, ranks = [], {}
    for t in TESTS:
        data = {m: per_image(t, m, qenas_mean) for m in MAIN}
        tests = [(m, paired(data["qenas"], data[m])) for m in MAIN[1:]]
        for (m, r), pad in zip(tests, holm([r["p"] for _, r in tests])):
            rows.append((t, m, r, pad))
        ids = sorted(set.intersection(*[set(v) for v in data.values()]))
        M = np.array([[data[m][i] for m in MAIN] for i in ids])
        rk = np.mean([rankdata(-row) for row in M], axis=0)
        k, n = len(MAIN), len(ids)
        ranks[t] = {"ranks": dict(zip(MAIN, rk)), "cd": 2.569 * np.sqrt(k * (k + 1) / (6 * n)), "n": n,
                    "p": friedmanchisquare(*M.T).pvalue}
    return rows, ranks


def fig_forest(rows):
    fig, ax = plt.subplots(figsize=(W1, 4.6))
    n = len(rows)
    labels = []
    for i, (t, m, r, pad) in enumerate(rows):
        y = n - 1 - i - (list(TESTS).index(t)) * 0.5
        sig = pad < 0.05
        ax.plot([100 * r["lo"], 100 * r["hi"]], [y, y], color=COLOR[m], lw=1.6, solid_capstyle="butt")
        ax.scatter([100 * r["diff"]], [y], marker=MARK[m], s=26, color=COLOR[m] if sig else "white",
                   edgecolor=COLOR[m], lw=1.0, zorder=3)
        labels.append((y, f"{SHORT[t]}: vs {NAME[m]}"))
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks([y for y, _ in labels], [s for _, s in labels])
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Δ Dice: QENAS − compared model (pp)")
    h = [Line2D([], [], marker="o", color=INK2, ls="", ms=5), Line2D([], [], marker="o", mfc="white", mec=INK2, ls="",
                                                                     ms=5)]
    ax.legend(h, ["significant (Wilcoxon–Holm, p < 0.05)", "not significant"], loc="upper center",
              bbox_to_anchor=(0.3, -0.09), ncol=1)
    save(fig, D_CMP, "Paired per-image Dice difference between QENAS and each compared model")


def cliques(order, rk, cd):
    """Maximal groups of consecutive models (sorted by rank) whose rank range is below the critical difference."""
    out = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and rk[order[j + 1]] - rk[order[i]] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in out):
            out.append((i, j))
    return out


def fig_cd(ranks):
    fig, axs = plt.subplots(len(TESTS), 1, figsize=(W2, 5.6), sharex=True)
    for i, (ax, t) in enumerate(zip(axs, TESTS)):
        r = ranks[t]
        rk = r["ranks"]
        order = sorted(MAIN, key=lambda m: rk[m])
        ax.set_ylim(-1.15, 1.0)
        ax.axhline(0, color=AXIS, lw=0.6)
        for m in MAIN[::-1]:
            ax.scatter([rk[m]], [0], marker=MARK[m], s=34, color=COLOR[m], zorder=3, edgecolor="white", lw=0.5)
        for q, (a, b) in enumerate(cliques(order, rk, r["cd"])):
            ax.plot([rk[order[a]] - 0.03, rk[order[b]] + 0.03], [-0.42 - 0.24 * q] * 2, color=INK,
                    lw=2.0, solid_capstyle="round")
        if i < len(TESTS) - 1:
            ax.spines["bottom"].set_visible(False)
        lo = 1.0
        ax.plot([lo, lo + r["cd"]], [0.55, 0.55], color=INK2, lw=1.0)
        for xx in (lo, lo + r["cd"]):
            ax.plot([xx, xx], [0.47, 0.63], color=INK2, lw=1.0)
        ax.text(lo + r["cd"] + 0.04, 0.55, f"CD = {r['cd']:.2f}", va="center")
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.text(-0.01, 0.0, f"{caption(i, DSL[t])}\n(n = {r['n']}, Friedman p {'< 0.001' if r['p'] < 1e-3 else '= %.3f' % r['p']})",
                transform=ax.get_yaxis_transform(), ha="right", va="center")
        if i < len(TESTS) - 1:
            ax.tick_params(axis="x", length=0)
    axs[-1].set_xlim(0.9, 4.1)
    axs[-1].set_xticks([1, 1.5, 2, 2.5, 3, 3.5, 4])
    axs[-1].set_xlabel("Mean rank of per-image test Dice (1 = best)")
    h = [Line2D([], [], marker=MARK[m], color=COLOR[m], ls="", ms=5.5) for m in MAIN]
    h.append(Line2D([], [], color=INK, lw=2.0))
    fig.legend(h, [LEG[m] for m in MAIN] + ["no significant difference (Nemenyi, α = 0.05)"], ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.89))
    fig.subplots_adjust(hspace=0.15)
    save(fig, D_CMP, "Critical difference diagram of the mean ranks of all models")


# =========================================================================================== review: fair comparison
def fig_dice_bars_fair():
    fig, ax = plt.subplots(figsize=(W2, 2.9))
    w = 0.2
    for j, m in enumerate(MAIN):
        for i, t in enumerate(TESTS):
            v = [(ext_metrics(m, s) if t == "EXT" else fm(t, m, s)["test"])["dice"] * 100 for s in SEEDS]
            mu, sd = np.mean(v), np.std(v, ddof=1)
            x = i + (j - 1.5) * w
            ax.bar(x, mu, w * 0.92, color=COLOR[m], edgecolor="none", zorder=2)
            ax.errorbar(x, mu, yerr=sd, color=INK2, lw=0.7, capsize=1.6, zorder=3)
            ax.text(x, mu + sd + 2.6, f"{mu:.1f}", ha="center", va="bottom", rotation=90)
    ax.set_xticks(range(len(TESTS)), [DSL[t] if t != "EXT" else EXT_TICK for t in TESTS])
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Test Dice (%)")
    ax.set_xlabel("Test set")
    h = [Patch(facecolor=COLOR[m]) for m in MAIN] + [errbar_note()]
    ax.legend(h, [NAME[m] for m in MAIN] + ["mean ± SD over 3 runs"], ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), handlelength=1.2)
    save(fig, D_REV, "Test Dice of all models, mean of three runs for every model (fair version)")


# =========================================================================================== 2 QENAS stages
def run_dir(ds=None, s=None):
    ds, s = (ds, s) if ds else STAGE_RUN
    return pa.exp(ds, "qenas", s)


def candidate_status(d):
    """candidate id -> 'r1' (eliminated at rung 1), 'r2', 'final' (finalist), 'sel' (selected)."""
    sel = load(d / "selection.json")["selected_id"]
    sh = load(d / "successive_halving.json")
    out = {}
    for c in sh["rungs"][0]["trained"]:
        out[c] = "r1"
    for c in sh["rungs"][1]["trained"]:
        out[c] = "r2"
    for c in sh["rungs"][2]["trained"]:
        out[c] = "final"
    out[sel] = "sel"
    return out, sel, sh


ST_STYLE = {"r1": ("#bdbdbd", 0.8, "eliminated after rung 1 (5 epochs)"),
            "r2": ("#7fb1ee", 0.9, "eliminated after rung 2 (10 epochs)"),
            "final": ("#0d366b", 1.0, "finalist, not selected (20 epochs)"),
            "sel": ("#eb6834", 1.8, "selected architecture")}
STAR = "#eb6834"


def fig_objective_space():
    d = run_dir()
    arch = load(d / "search_results.json")
    cands = pa.read_csv(d / "pareto_candidates.csv")
    sel = load(d / "selection.json")["selected_id"]
    hist = pa.read_csv(d / "search_population_history.csv")
    last = max(int(r["generation"]) for r in hist)
    front = sorted([(float(r["params"]) / 1e6, float(r["zico"])) for r in hist
                    if int(r["generation"]) == last and r["nondominated"] == "True"])
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    g = np.array([r["first_generation"] for r in arch])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#d5e5f9", "#0d366b"])
    sc = ax.scatter([r["params"] / 1e6 for r in arch], [r["zico"] for r in arch], c=g, cmap=cmap, s=11, lw=0,
                    vmin=0, vmax=last, zorder=2)
    fx, fy = zip(*front)
    ax.plot(fx, fy, color=INK, lw=0.6, zorder=3)
    sc_ = [c for c in cands if int(c["candidate_id"]) == sel][0]
    others = [c for c in cands if int(c["candidate_id"]) != sel]
    ax.scatter([float(c["params"]) / 1e6 for c in others], [float(c["zico"]) for c in others], marker="s", s=24,
               facecolor="none", edgecolor=INK, lw=0.8, zorder=4)
    ax.scatter([float(sc_["params"]) / 1e6], [float(sc_["zico"])], marker="s", s=24, color=STAR, edgecolor=STAR,
               lw=0.8, zorder=5)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, aspect=25)
    cb.set_label("Generation of first evaluation")
    cb.set_ticks(range(0, last + 1, 2))
    cb.outline.set_linewidth(0.5)
    ax.set_xlabel("Parameters (M), objective 1 (minimised)")
    ax.set_ylabel("ZiCo score, objective 2 (maximised)")
    ax.grid(axis="x", visible=True)
    h = [Line2D([], [], marker="o", color="#5c8fd0", ls="", ms=3.5), Line2D([], [], color=INK, lw=0.6),
         Line2D([], [], marker="s", mfc="none", mec=INK, ls="", ms=5),
         Line2D([], [], marker="s", color=STAR, ls="", ms=5)]
    ax.legend(h, [f"evaluated architectures ({len(arch)})", "final Pareto front", "16 candidates",
                  "selected architecture"], ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              columnspacing=0.8)
    save(fig, D_STG, "Objective space explored by the training-free NSGA-II search")


def fig_front_evolution():
    d = run_dir()
    hist = pa.read_csv(d / "search_population_history.csv")
    last = max(int(r["generation"]) for r in hist)
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    gens = [0, last // 2, last]
    cols = ["#9cc2f2", "#2a78d6", "#0d366b"]
    mk = ["o", "s", "D"]
    for gg, c, m in zip(gens, cols, mk):
        pop = [(float(r["params"]) / 1e6, float(r["zico"]), r["nondominated"] == "True") for r in hist
               if int(r["generation"]) == gg]
        f = sorted((x, y) for x, y, nd in pop if nd)
        ax.scatter([p[0] for p in pop if not p[2]], [p[1] for p in pop if not p[2]], marker=m, s=10,
                   facecolor="none", edgecolor=c, lw=0.6, zorder=2)
        ax.plot(*zip(*f), color=c, lw=0.6, zorder=3)
        ax.scatter(*zip(*f), marker=m, s=16, color=c, zorder=4, edgecolor="white", lw=0.4,
                   label=f"generation {gg}" + (" (initial)" if gg == 0 else " (final)" if gg == last else ""))
    ax.set_xlabel("Parameters (M), minimised")
    ax.set_ylabel("ZiCo score, maximised")
    ax.grid(axis="x", visible=True)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", mfc="none", mec=MUTED, ls="", ms=4))
    l.append("dominated members")
    ax.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.4, columnspacing=0.8)
    save(fig, D_STG, "Pareto front of the population at the initial, middle and final generation")


def search_hv_curves(ds, s):
    from qenas.study.analysis import OBJ, _hv
    qa = load(pa.exp(ds, "qenas", s) / "search_results.json")
    ra = load(pa.exp(ds, "random_search", s) / "search_results.json")
    A = np.array([[r[o] for o in OBJ] for r in qa + ra], float)
    lo, hi = A.min(0), A.max(0)
    gens = sorted({r["first_generation"] for r in qa})
    curve = [_hv(np.array([[r[o] for o in OBJ] for r in qa if r["first_generation"] <= g]), lo, hi) for g in gens]
    hr = _hv(np.array([[r[o] for o in OBJ] for r in ra]), lo, hi)
    return gens, curve, hr, len(ra)


def fig_hypervolume():
    gens, curve, hr, nr = search_hv_curves(*STAGE_RUN)
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    ax.plot(gens, curve, marker="o", ms=3.5, color=COLOR["qenas"], label="NSGA-II search of QENAS")
    ax.axhline(hr, color=INK2, ls="--", lw=1.0, label=f"random search, same budget ({nr} architectures)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalised hypervolume")
    ax.set_xticks(gens)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, D_STG, "Hypervolume of the NSGA-II search across generations")


def fig_successive_halving():
    d = run_dir()
    status, sel, sh = candidate_status(d)
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    order = sorted(status, key=lambda c: ["r1", "r2", "final", "sel"].index(status[c]))
    for c in order:
        h = pa.read_csv(d / "candidate_histories" / f"candidate_{c:02d}.csv")
        col, lw, _ = ST_STYLE[status[c]]
        ax.plot([int(r["epoch"]) for r in h], [100 * float(r["val_dice"]) for r in h], color=col, lw=lw,
                zorder=3 if status[c] == "sel" else 2)
        ax.scatter([int(h[-1]["epoch"])], [100 * float(h[-1]["val_dice"])], s=8, color=col, zorder=3)
    for e in sh["rung_epochs"][:-1]:
        ax.axvline(e + 0.5, color=AXIS, lw=0.6, ls=(0, (3, 2)))
    lo, hi = ax.get_ylim()
    for k, ((a, b), n, e) in enumerate(zip([(0.5, 5.5), (5.5, 10.5), (10.5, 20.5)], sh["sizes"], sh["rung_epochs"])):
        ax.text((a + b) / 2, hi, f"rung {k + 1}\n{n} × {e}", ha="center", va="bottom")
    ax.set_ylim(lo, hi)
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation Dice (%)")
    h = [Line2D([], [], color=ST_STYLE[k][0], lw=max(ST_STYLE[k][1], 1.2)) for k in ST_STYLE]
    ax.legend(h, [ST_STYLE[k][2] for k in ST_STYLE], loc="upper center", bbox_to_anchor=(0.45, -0.2),
              handlelength=1.4)
    save(fig, D_STG, "Successive halving of the 16 candidate architectures")


def fig_rung_selection():
    d = run_dir()
    status, sel, sh = candidate_status(d)
    fig, ax = plt.subplots(figsize=(W1, 3.4))
    xs = [1, 2, 3]
    finals_ = []
    for c, stt in sorted(status.items(), key=lambda kv: ["r1", "r2", "final", "sel"].index(kv[1])):
        pts = [(xs[i], 100 * rg["scores"][str(c)]) for i, rg in enumerate(sh["rungs"]) if str(c) in rg["scores"]]
        col, lw, _ = ST_STYLE[stt]
        ax.plot(*zip(*pts), color=col, lw=lw, marker="o", ms=3.5 if stt != "sel" else 5,
                zorder=3 if stt == "sel" else 2)
        if stt in ("final", "sel"):
            finals_.append((pts[-1][1], c, stt))
    finals_.sort(reverse=True)
    top = finals_[0][0] + 0.6
    for k, (y, c, stt) in enumerate(finals_):          # finalist labels, evenly spaced with leader lines
        ly = top - 1.9 * k
        ax.annotate(f"C{c}: {y:.2f}" + ("  (selected)" if stt == "sel" else ""), (3, y), xytext=(3.12, ly),
                    va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=0, shrinkB=2))
    for i, rg in enumerate(sh["rungs"][:-1]):
        kept = min(rg["scores"][str(c)] for c in rg["promoted"])
        cut = max(v for k, v in rg["scores"].items() if int(k) not in rg["promoted"])
        ax.plot([xs[i] - 0.2, xs[i] + 0.2], [50 * (kept + cut)] * 2, color=INK2, lw=0.8, ls="--")
    ax.set_xticks(xs, [f"rung {i + 1}\n{n} × {e} epochs" for i, (n, e) in
                       enumerate(zip(sh["sizes"], sh["rung_epochs"]))])
    ax.grid(axis="x", visible=False)
    ax.set_xlim(0.7, 3.95)
    ax.set_xlabel("Successive-halving rung")
    ax.set_ylabel("Mean validation Dice of the last 5 epochs (%)")
    h = [Line2D([], [], color=ST_STYLE[k][0], lw=max(ST_STYLE[k][1], 1.2), marker="o", ms=3.5) for k in ST_STYLE]
    h.append(Line2D([], [], color=INK2, lw=0.8, ls="--"))
    ax.legend(h, [ST_STYLE[k][2] for k in ST_STYLE] + ["promotion cut-off (best half kept)"], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=1)
    save(fig, D_STG, "Dice-based selection of the architecture across the successive-halving rungs")


def fig_alpha():
    d = run_dir()
    lw = load(d / "loss_weights.json")
    runs = sorted(lw["runs"], key=lambda r: r["alpha"])
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    a = [r["alpha"] for r in runs]
    c = [100 * r["criterion"] for r in runs]
    ax.plot(a, c, color=COLOR["qenas"], marker="o", ms=4.5, zorder=2)
    ch = [r for r in runs if r["chosen"]][0]
    ax.plot([ch["alpha"]], [100 * ch["criterion"]], marker="o", ms=4.5, color=STAR, ls="", zorder=3,
            label=f"selected weight (α = {ch['alpha']:.1f})")
    for x, y in zip(a, c):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center")
    ax.set_xticks(a)
    ax.set_xlim(0.12, 0.88)
    ax.set_ylim(min(c) - 1.5, max(c) + 1.2)
    ax.set_xlabel("Loss weight α   (L = α·BCE + (1 − α)·soft-mIoU loss)")
    ax.set_ylabel("Mean validation Dice of\nthe last 5 epochs (%)")
    ax.legend(loc="upper right")
    save(fig, D_STG, "Selection of the loss-function weight α")


def fig_final_training():
    d = run_dir()
    h = pa.read_csv(d / "final_training.csv")
    f = load(d / "final_metrics.json")
    ep = [int(r["epoch"]) for r in h]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), constrained_layout=True)
    for i, (ax, key, lab) in enumerate(zip(axs, ("loss", "dice"), ("Loss", "Dice (%)"))):
        sc = 1 if key == "loss" else 100
        ax.plot(ep, [sc * float(r[f"train_{key}"]) for r in h], color=COLOR["qenas"], lw=1.1, label="training")
        ax.plot(ep, [sc * float(r[f"val_{key}"]) for r in h], color=STAR, lw=1.1, label="validation")
        ax.axvline(f["best_epoch"], color=INK2, lw=0.8, ls="--",
                   label=f"best validation Dice (epoch {f['best_epoch']})")
        ax.set_xlabel(f"Epoch\n{caption(i, 'Loss' if key == 'loss' else 'Dice')}", linespacing=1.7)
        ax.set_ylabel(lab)
        ax.set_xlim(0, len(ep) + 1)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, D_STG, "Training and validation curves of the final QENAS model")


# =========================================================================================== review: stage extras
def fig_proxy_vs_dice():
    fig, axs = plt.subplots(2, 4, figsize=(W2, 4.8), constrained_layout=True)
    for i, ds in enumerate(DATASETS):
        d = pa.exp(ds, "qenas", BEST[ds])
        sh = load(d / "successive_halving.json")
        cands = {int(c["candidate_id"]): c for c in pa.read_csv(d / "pareto_candidates.csv")}
        sc = sh["rungs"][0]["scores"]
        ids = sorted(int(k) for k in sc)
        y = [100 * sc[str(c)] for c in ids]
        for row, (key, lab, scale) in enumerate((("zico", "ZiCo score", 1), ("params", "Parameters (M)", 1e-6))):
            ax = axs[row, i]
            x = [float(cands[c][key]) * scale for c in ids]
            rho = spearmanr(x, y).statistic
            ax.scatter(x, y, s=14, color=COLOR["qenas"], edgecolor="white", lw=0.4)
            lab_r = f"{lab} (ρ = {rho:+.2f})"
            ax.set_xlabel(lab_r if row == 0 else f"{lab_r}\n{caption(i, DSL[ds])}", linespacing=1.7)
            if i == 0:
                ax.set_ylabel("Validation Dice after\n5 epochs (%)")
    save(fig, D_REV, "Training-free objectives versus 5-epoch validation Dice of the 16 candidates")


def fig_hv_all():
    fig, axs = plt.subplots(1, 4, figsize=(W2, 2.4), sharey=True, constrained_layout=True)
    for i, (ax, ds) in enumerate(zip(axs, DATASETS)):
        cs, rs = [], []
        for s in SEEDS:
            gens, c, hr, _ = search_hv_curves(ds, s)
            cs.append(c)
            rs.append(hr)
        cs = np.array(cs)
        ax.plot(gens, cs.mean(0), color=COLOR["qenas"], marker="o", ms=3)
        ax.fill_between(gens, cs.min(0), cs.max(0), color=COLOR["qenas"], alpha=0.15, lw=0)
        ax.axhspan(min(rs), max(rs), color=MUTED, alpha=0.25, lw=0)
        ax.axhline(np.mean(rs), color=INK2, ls="--", lw=0.9)
        ax.set_xlabel(f"Generation\n{caption(i, DSL[ds])}", linespacing=1.7)
        ax.set_xticks([0, 5, 10])
    axs[0].set_ylabel("Normalised hypervolume")
    fig.legend([Line2D([], [], color=COLOR["qenas"], marker="o", ms=3), Line2D([], [], color=INK2, ls="--")],
               ["NSGA-II (mean and range of 3 runs)", "random search, same budget (mean and range)"], ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, D_REV, "Hypervolume of NSGA-II versus random search on all datasets")


def fig_ablation(ab, forest):
    rows = ab["v2"]["rows"]
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    n = len(forest)
    labs = []
    short = {"α grid vs fixed α = 0.5": "α search vs fixed α = 0.5", "vs random candidate": "vs random candidate",
             "vs zero-shot pick (max ZiCo)": "vs zero-shot (max ZiCo)"}
    for i, ((ds, nm, r), row) in enumerate(zip(forest, rows)):
        y = n - 1 - i
        p = row[8]
        sig = p == "<1e-4" or float(p) < 0.05
        col = COLOR["qenas"] if r["diff"] > 0 else STAR
        ax.plot([100 * r["lo"], 100 * r["hi"]], [y, y], color=col, lw=1.6, solid_capstyle="butt")
        ax.scatter([100 * r["diff"]], [y], s=24, color=col if sig else "white", edgecolor=col, lw=1, zorder=3)
        labs.append(f"{SHORT[ds]}: {short.get(nm, nm)}")
    ax.axvline(0, color=INK2, lw=0.7)
    ax.set_yticks(range(n)[::-1], labs)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("Δ Dice: QENAS − ablated variant (pp)\n(mean of 3 runs each, 95 % bootstrap CI)")
    ax.legend([Line2D([], [], marker="o", color=INK2, ls="", ms=5),
               Line2D([], [], marker="o", mfc="white", mec=INK2, ls="", ms=5)],
              ["significant (Wilcoxon–Holm)", "not significant"], ncol=2, loc="lower center",
              bbox_to_anchor=(0.3, 1.0))
    save(fig, D_REV, "Ablation study of the QENAS components")


def fig_blocks(counts):
    fig, ax = plt.subplots(figsize=(W1, 2.8))
    pos = [f"{p} ($g_{i + 1}$)" for i, p in enumerate(["E1", "E2", "E3", "E4", "D1", "D2", "D3", "D4"])]
    left = np.zeros(8)
    for k in range(5):
        ax.barh(range(8), counts[:, k], left=left, color=BLOCK_COLOR[k], edgecolor="white", lw=0.6, height=0.7,
                label=BLOCK_NAME[k])
        left += counts[:, k]
    ax.set_yticks(range(8), pos)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("Number of final architectures (4 datasets × 3 runs)")
    ax.set_ylabel("Position in the network (gene)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.0), handlelength=1.0, columnspacing=0.8)
    save(fig, D_REV, "Block types selected at each position of the 12 final QENAS architectures")


# =========================================================================================== 4 qualitative results
def test_images(test):
    """(sample ids, images [N,H,W,3] uint8, masks [N,H,W]) exactly as the test stage loads them."""
    from qenas.config import build_config
    from qenas.datasets.discovery import Sample, discover, locate_data_root, rebase_path
    from qenas.datasets.preprocessing import preload_split
    if test == "EXT":
        cfg = build_config("CVC", overrides={"data.root": locate_data_root(str(pa.DATA), "CVC"),
                                             "paths.cache": str(pa.CACHE.parent)})
        samples = sorted(discover(cfg).samples, key=lambda s: s.sample_id)
        imgs, masks = preload_split(samples, (256, 256), int(cfg["data"]["mask_threshold"]), "CVC", "all", pa.CACHE)
        return [s.sample_id for s in samples], imgs, masks
    ref = pa.exp(test, "qenas", SEEDS[0])
    sp = load(ref / "split.json")
    cfg = load(ref / "config.json")["config"]
    root = locate_data_root(str(pa.DATA), test)
    samples = []
    for d in sp["splits"]["test"]:
        s = Sample.from_dict(d)
        s.image = rebase_path(s.image, root, test, sp.get("data_root"))
        s.masks = [rebase_path(m, root, test, sp.get("data_root")) for m in s.masks]
        samples.append(s)
    imgs, masks = preload_split(samples, tuple(cfg["data"]["image_size"]), int(cfg["data"]["mask_threshold"]), test,
                                "test", pa.CACHE)
    return [s.sample_id for s in samples], imgs, masks


def predictions(test, m, s):
    d = pa.exp("KVASIR" if test == "EXT" else test, m, s)
    z = np.load(d / "external_CVC" / "external_predictions.npz" if test == "EXT" else d / "test_predictions.npz")
    return list(z["sample_ids"]), z["masks"]


def shown_seed(test, m):
    """QENAS: its best run; a baseline: its representative run (test Dice closest to its 3-run mean)."""
    if m == "qenas":
        return BEST[test]
    if test == "EXT":
        v = {s: ext_metrics(m, s)["dice"] for s in SEEDS}
        mu = np.mean(list(v.values()))
        return min(SEEDS, key=lambda s: abs(v[s] - mu))
    return representative_seed(test, m)


def dice(p, g):
    p, g = p.astype(bool), g.astype(bool)
    den = p.sum() + g.sum()
    return 1.0 if den == 0 else 2.0 * (p & g).sum() / den


def pick_cases(gts, preds, n=4):
    """Images where QENAS is the best of the four models, spread over the lesion size (QENAS best run)."""
    fg = np.array([g.mean() for g in gts])
    q = np.array([dice(p, g) for p, g in zip(preds["qenas"], gts)])
    o = np.array([[dice(p, g) for p, g in zip(preds[m], gts)] for m in MAIN[1:]]).max(0)
    ok = np.where((fg > 0) & (q >= o) & (q >= np.quantile(q[fg > 0], 0.5)))[0]
    if len(ok) < n:
        ok = np.argsort(-(q - o))[: 4 * n]
    ok = ok[np.argsort(fg[ok])]
    bins = np.array_split(ok, n)
    return [int(b[np.argmax(q[b] + 2 * (q[b] - o[b]))]) for b in bins if len(b)]


TP_C, FP_C, FN_C = np.array([27, 175, 122]) / 255, np.array([227, 73, 72]) / 255, np.array([57, 135, 229]) / 255


def overlay(img, pred=None, gt=None):
    base = img.astype(float) / 255
    out = base * 0.75
    if pred is None:                                     # ground truth
        mk = gt.astype(bool)
        out[mk] = 0.45 * out[mk] + 0.55 * np.array([1.0, 1.0, 1.0])
        return np.clip(out, 0, 1)
    p, g = pred.astype(bool), gt.astype(bool)
    for mk, c in ((p & g, TP_C), (p & ~g, FP_C), (~p & g, FN_C)):
        out[mk] = 0.4 * out[mk] + 0.6 * c
    return np.clip(out, 0, 1)


def fig_qualitative(test, folder, title):
    ids, imgs, gts = test_images(test)
    preds = {}
    for m in MAIN:
        pid, pm = predictions(test, m, shown_seed(test, m))
        assert pid == ids, (test, m)
        preds[m] = pm
    cases = pick_cases(gts, preds)
    h_img, w_img = imgs.shape[1:3]
    crops = {}
    if test == "IDRID":                                  # the optic disc is small: show a square crop around it
        for k in cases:
            ys, xs = np.nonzero(gts[k])
            side = int(min(h_img, w_img, max(96, 2.6 * max(np.ptp(ys), np.ptp(xs)))))
            cy, cx = int(ys.mean()), int(xs.mean())
            y0, x0 = np.clip(cy - side // 2, 0, h_img - side), np.clip(cx - side // 2, 0, w_img - side)
            crops[k] = (slice(y0, y0 + side), slice(x0, x0 + side))
        h_img = w_img = 1
    cols = 2 + len(MAIN)
    cw = (W2 - 0.05) / cols
    ch = cw * h_img / w_img
    row_h = ch + 0.2
    top = 0.22
    H = top + len(cases) * row_h + 0.32
    fig = plt.figure(figsize=(W2, H))
    heads = ["Image", "Ground truth"] + [NAME[m] for m in MAIN]
    for j, hd in enumerate(heads):
        fig.text((0.025 + j * cw + cw / 2) / W2, 1 - 0.1 / H, hd, ha="center", va="center")
    for i, k in enumerate(cases):
        y0 = H - top - (i + 1) * row_h + 0.2
        c = crops.get(k, (slice(None), slice(None)))
        panels = [imgs[k][c] / 255, overlay(imgs[k][c], gt=gts[k][c])] + [overlay(imgs[k][c], preds[m][k][c], gts[k][c])
                                                                          for m in MAIN]
        for j, im in enumerate(panels):
            ax = fig.add_axes([(0.025 + j * cw + 0.02) / W2, y0 / H, (cw - 0.04) / W2, ch / H])
            ax.imshow(im, interpolation="lanczos")
            ax.set_axis_off()
            if j >= 2:
                m = MAIN[j - 2]
                fig.text((0.025 + j * cw + cw / 2) / W2, (y0 - 0.1) / H, f"Dice = {100 * dice(preds[m][k], gts[k]):.1f} %",
                         ha="center", va="center")
    h = [Patch(facecolor=TP_C), Patch(facecolor=FP_C), Patch(facecolor=FN_C), Patch(facecolor="white", edgecolor=MUTED)]
    fig.legend(h, ["true positive", "false positive", "false negative", "ground-truth region"], ncol=4,
               loc="center", bbox_to_anchor=(0.5, 0.13 / H))
    save(fig, folder, title)
    return [ids[k] for k in cases]


def fig_all_qualitative():
    out = {}
    for t in DATASETS:
        out[t] = fig_qualitative(t, D_QUA, f"Qualitative comparison of the predicted masks on {DSL[t]}"
                                 + (" (optic-disc region)" if t == "IDRID" else ""))
    out["EXT"] = fig_qualitative("EXT", D_QUA, "Qualitative comparison of the predicted masks on CVC-ClinicDB "
                                                "(external test of the Kvasir-SEG models)")
    print("   cases:", out)


# =========================================================================================== main
def main(only=None):
    global EXTENDED
    EXTENDED = pa.extended_metrics()
    print("best QENAS runs:", BEST)
    jobs = {
        "cmp": lambda: (fig_dice_bars(), fig_all_metrics(), fig_dice_vs_cost("params"), fig_dice_vs_cost("flops"),
                        fig_search_time(), fig_total_time(), fig_cost_bars(), fig_dice_vs_time(),
                        fig_distribution(), fig_stats()),
        "stg": lambda: (fig_objective_space(), fig_front_evolution(), fig_hypervolume(), fig_successive_halving(),
                        fig_rung_selection(), fig_alpha(), fig_final_training()),
        "qua": fig_all_qualitative,
        "rev": lambda: (fig_dice_bars_fair(), fig_proxy_vs_dice(), fig_hv_all(), fig_ablation(*pa.ablations()),
                        fig_blocks(pa.architectures()[1])),
    }
    for k, f in jobs.items():
        if only is None or k in only:
            f()


def fig_stats():
    rows, ranks = comparison_stats()
    fig_forest(rows)
    fig_cd(ranks)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
