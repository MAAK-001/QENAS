"""Architecture diagrams of QENAS 3.0 (same style as scripts/paper_figures.py).

    python scripts/paper_diagrams.py

Writes to results/figures/3 Architecture/ (and review/):
* Complete architecture of QENAS               search space, candidate blocks and the full search/selection pipeline;
* Architecture found by QENAS on BUSI           the network selected in the best run (BUSI, seed 47);
* review/Architectures selected by QENAS ...    the chromosomes of all 12 final runs.
All drawing is in inches on a full-figure axes, so every text is exactly the global font size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import paper_figures as pf  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from paper_figures import (AXIS, BLOCK_COLOR, BLOCK_FILL, BLOCK_NAME, BLOCK_SHORT, COLOR, D_ARC, D_REV, DSL,  # noqa
                           INK, INK2, MUTED, W2, load, plt, save)

TRAIN_FREE_FILL, TRAIN_FREE_EDGE = "#eef4fc", COLOR["qenas"]
TRAIN_FILL, TRAIN_EDGE = "#fdf1ea", COLOR["nnunet"]
GENE_FILL = "#f4f7fb"


class Canvas:
    """A figure whose data coordinates are inches."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_axis_off()
        self.limits()

    def limits(self):
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)

    def box(self, cx, cy, w, h, text="", fc="white", ec=INK2, lw=0.7, ls="-", r=0.035, z=2, weight="normal",
            ha="center"):
        self.ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                         fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
        if text:
            x = cx if ha == "center" else cx - w / 2 + 0.06
            self.ax.text(x, cy, text, ha=ha, va="center", zorder=z + 1, linespacing=1.15, weight=weight)
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def line(self, pts, arrow=True, color=INK2, lw=0.75, ls="-", z=1):
        xs, ys = zip(*pts)
        if not arrow:
            self.ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt")
            return
        if len(pts) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, zorder=z, solid_capstyle="butt",
                         solid_joinstyle="miter")
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=0.36,head_width=0.18",
                                          mutation_scale=10, color=color, lw=lw, ls=ls, zorder=z, shrinkA=0,
                                          shrinkB=0))

    def text(self, x, y, s, **kw):
        kw = {"ha": "center", "va": "center", "zorder": 4, **kw}
        self.ax.text(x, y, s, **kw)

    def add(self, cx, cy, r=0.075):
        self.ax.add_patch(Circle((cx, cy), r, fc="white", ec=INK2, lw=0.7, zorder=3))
        self.text(cx, cy - 0.005, "+")

    def image(self, img, x0, y0, w, h, cmap=None):
        self.ax.imshow(img, extent=(x0, x0 + w, y0, y0 + h), cmap=cmap, interpolation="lanczos", zorder=2,
                       aspect="auto")
        self.ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="square,pad=0", fc="none", ec=AXIS, lw=0.5,
                                         zorder=3))
        self.limits()


# ------------------------------------------------------------------------------------------- shared pieces
def example_images():
    """A BUSI test image, its ground truth and the prediction of the best QENAS run."""
    ds, s = pf.STAGE_RUN
    ids, imgs, gts = pf.test_images(ds)
    pid, pm = pf.predictions(ds, "qenas", s)
    k = ids.index("benign/benign (33)") if "benign/benign (33)" in ids else int(np.argmax([g.mean() for g in gts]))
    return imgs[k], gts[k], pm[k]


def u_network(c, xs_enc, xs_dec, lv, bw, bh, texts_enc, texts_dec, fills_enc=None, fills_dec=None, edges_enc=None,
              edges_dec=None, sca_w=0.36):
    """U-shaped network: encoder boxes (stem + 4) at levels lv[0..4], decoder boxes at levels lv[3..0]."""
    for k, (x, t) in enumerate(zip(xs_enc, texts_enc)):
        c.box(x, lv[k], bw, bh, t, fc=(fills_enc or [GENE_FILL] * 5)[k], ec=(edges_enc or [INK2] * 5)[k],
              lw=0.8 if edges_enc else 0.7)
    for j, (x, t) in enumerate(zip(xs_dec, texts_dec)):
        c.box(x, lv[3 - j], bw, bh, t, fc=(fills_dec or [GENE_FILL] * 4)[j], ec=(edges_dec or [INK2] * 4)[j],
              lw=0.8 if edges_dec else 0.7)
    for k in range(4):                                   # encoder: down-sampling path
        x0 = xs_enc[k] + 0.32 * bw / 2
        c.line([(x0, lv[k] - bh / 2), (x0, lv[k + 1]), (xs_enc[k + 1] - bw / 2, lv[k + 1])])
    srcs = [(xs_enc[4], lv[4])] + [(xs_dec[j], lv[3 - j]) for j in range(3)]
    for j, (sx, sy) in enumerate(srcs):                  # decoder: up-sampling path
        tx = xs_dec[j] - 0.32 * bw / 2
        c.line([(sx + bw / 2, sy), (tx, sy), (tx, lv[3 - j] - bh / 2)])
    for k in range(4):                                   # skip connections with spatial-channel attention
        x0, x1 = xs_enc[k] + bw / 2, xs_dec[3 - k] - bw / 2
        mid = (x0 + x1) / 2
        c.line([(x0, lv[k]), (mid - sca_w / 2, lv[k])], arrow=False, ls=(0, (3, 1.5)))
        c.box(mid, lv[k], sca_w, 0.2, "SCA", fc="#f1f1f1", ec=MUTED, lw=0.6, z=3)
        c.line([(mid + sca_w / 2, lv[k]), (x1, lv[k])], ls=(0, (3, 1.5)))


BLOCK_ROWS = {
    0: [("box", "3×3 GConv\nBN · ReLU"), ("box", "3×3 GConv\nBN · ReLU"), ("add", None), ("box", "ReLU")],
    1: [("box", "BN · ReLU\n3×3 GConv"), ("box", "BN · ReLU\n3×3 GConv"), ("box", "BN · ReLU\n3×3 GConv"),
        ("box", "Concat · 1×1 conv")],
    2: [("box", "3×3 GConv\nBN · ReLU"), ("branch", ["1×3\n3×1", "1×5\n5×1", "1×7\n7×1"]),
        ("box", "Concat · 1×1 conv")],
    3: [("box", "5×5 DWConv · BN"), ("box", "Linear (4C) · GELU"), ("box", "Linear (C) · BN"), ("add", None),
        ("box", "GELU · 1×1 conv")],
    4: [("branch", ["$s_0$", "$s_1$"]), ("box", "ReLU · 1×1 conv\nBN (each input)"), ("box", "4 nodes: Σ of 2 ops"),
        ("box", "Concat of 4 nodes")],
}
BLOCK_SKIP = {0: "add", 1: "dense", 3: "add"}


def block_column(c, k, x0, top, col_w=1.38, box_w=1.13, rh=0.19, gap=0.1):
    """One candidate block: header + its layers (top = upper edge of the header)."""
    c.box(x0 + col_w / 2, top - 0.11, col_w, 0.22, f"{k} · {BLOCK_NAME[k]}", fc=BLOCK_FILL[k], ec=BLOCK_COLOR[k],
          lw=0.9)
    cx = x0 + box_w / 2 + 0.02
    lane = x0 + box_w + 0.12
    y = top - 0.22 - gap - rh / 2
    prev_bottom = top - 0.22
    entry = (cx, top - 0.22 - gap * 0.45)
    centers = []
    for kind, val in BLOCK_ROWS[k]:
        multi = (kind == "branch" and "\n" in val[0]) or (kind == "box" and "\n" in val)
        h = 0.34 if multi else rh
        cy = y - (h - rh) / 2
        if kind == "box":
            c.box(cx, cy, box_w, h, val, fc="white", ec=BLOCK_COLOR[k], lw=0.6)
            c.line([(cx, prev_bottom), (cx, cy + h / 2)])
        elif kind == "add":
            c.add(cx, cy)
            c.line([(cx, prev_bottom), (cx, cy + 0.075)])
        else:
            n = len(val)
            bwid = (box_w - 0.06 * (n - 1)) / n
            for i, t in enumerate(val):
                bx = cx - box_w / 2 + bwid / 2 + i * (bwid + 0.06)
                c.box(bx, cy, bwid, h, t, fc="white", ec=BLOCK_COLOR[k], lw=0.6)
                c.line([(cx if k != 4 else bx, prev_bottom), (bx, cy + h / 2)])
        centers.append((kind, cy, h))
        prev_bottom = cy - h / 2
        y = cy - h / 2 - gap - rh / 2
    skip = BLOCK_SKIP.get(k)
    if skip == "add":
        _, ay, _ = [r for r in centers if r[0] == "add"][0]
        c.line([entry, (lane, entry[1]), (lane, ay), (cx + 0.075, ay)])
    elif skip == "dense":
        last = centers[-1][1]
        c.line([entry, (lane, entry[1]), (lane, last)], arrow=False)
        for _, cy, h in centers[1:]:
            c.line([(lane, cy), (cx + box_w / 2, cy)])
    return prev_bottom


# ------------------------------------------------------------------------------------------- complete architecture
def fig_complete_architecture():
    W, H = W2, 10.75
    c = Canvas(W, H)
    img, gt, pred = example_images()
    OA = 0.75                                            # vertical offset of part (a)

    # ---- (a) search space -------------------------------------------------------------------------------------
    lv = [9.02 + OA, 8.44 + OA, 7.86 + OA, 7.28 + OA, 6.70 + OA]
    bw, bh = 0.62, 0.44
    xs_enc = [0.92, 1.55, 2.18, 2.81, 3.44]
    xs_dec = [4.07, 4.70, 5.33, 5.96]
    enc_t = ["Stem\n32, H", "E1 · $g_1$\n64, H/2", "E2 · $g_2$\n128, H/4", "E3 · $g_3$\n256, H/8", "E4 · $g_4$\n512, H/16"]
    dec_t = ["D1 · $g_5$\n256, H/8", "D2 · $g_6$\n128, H/4", "D3 · $g_7$\n64, H/2", "D4 · $g_8$\n32, H"]
    u_network(c, xs_enc, xs_dec, lv, bw, bh, enc_t, dec_t, fills_enc=["#ececec"] + [GENE_FILL] * 4)
    c.text(xs_enc[4], lv[4] - bh / 2 - 0.12, "bottleneck")
    c.image(img, 0.05, lv[0] - 0.24, 0.48, 0.48)
    c.line([(0.53, lv[0]), (xs_enc[0] - bw / 2, lv[0])])
    c.text(0.29, lv[0] - 0.36, "input")
    ox = 6.62
    c.image(pred, ox, lv[0] - 0.24, 0.48, 0.48, cmap="gray")
    c.line([(xs_dec[3] + bw / 2, lv[0]), (ox, lv[0])])
    c.text((xs_dec[3] + bw / 2 + ox) / 2, lv[0] - 0.22, "1×1\nconv", linespacing=1.0)
    c.text(ox + 0.24, lv[0] - 0.36, "mask")
    # chromosome
    ch = [4, 0, 1, 0, 4, 4, 4, 2]
    cy = 9.62 + OA
    c.text(1.47, cy, "Chromosome", ha="right")
    gx = 1.62
    for i, g in enumerate(ch):
        x = gx + i * 0.27 + (0.1 if i >= 4 else 0)
        c.box(x + 0.12, cy, 0.24, 0.24, str(g), fc=BLOCK_FILL[g], ec=BLOCK_COLOR[g], lw=0.8, r=0.02)
        c.text(x + 0.12, cy + 0.25, f"$g_{i + 1}$")
    c.text(4.05, cy + 0.08, "$g_i \\in \\{0, 1, 2, 3, 4\\}$ selects the block of position $i$", ha="left")
    c.text(4.05, cy - 0.1, "(here: the architecture selected on BUSI);  $5^8$ = 390,625 networks", ha="left")
    # encoder / decoder position details
    c.text(0.85, 8.02 + OA, "Encoder position (gene 0–3)")
    for i, t in enumerate(["BN", "AvgPool 2×2", "3×3 GConv (C → 2C)", "Block of the gene"]):
        y = 7.74 + OA - i * 0.3
        c.box(0.85, y, 1.45, 0.2, t, fc="white", ec=INK2, lw=0.6)
        if i:
            c.line([(0.85, y + 0.2), (0.85, y + 0.1)])
    c.text(0.85, 6.6 + OA, "gene 4: NAS-Unet down cell")
    dx = 6.21
    c.text(dx, 8.02 + OA, "Decoder position (gene 0–3)")
    c.box(dx - 0.46, 7.72 + OA, 0.86, 0.2, "BN (input)", fc="white", lw=0.6)
    c.box(dx + 0.46, 7.72 + OA, 0.86, 0.2, "BN (skip)", fc="white", lw=0.6)
    c.box(dx - 0.46, 7.42 + OA, 0.86, 0.2, "2×2 TConv", fc="white", lw=0.6)
    c.box(dx + 0.46, 7.42 + OA, 0.86, 0.2, "SCA", fc="#f1f1f1", ec=MUTED, lw=0.6)
    c.box(dx, 7.12 + OA, 1.78, 0.2, "Concat · 1×1 conv", fc="white", lw=0.6)
    c.box(dx, 6.82 + OA, 1.78, 0.2, "Block of the gene", fc="white", lw=0.6)
    for sx in (dx - 0.46, dx + 0.46):
        c.line([(sx, 7.62 + OA), (sx, 7.52 + OA)])
        c.line([(sx, 7.32 + OA), (sx, 7.22 + OA)])
    c.line([(dx, 7.02 + OA), (dx, 6.92 + OA)])
    c.text(dx, 6.55 + OA, "gene 4: NAS-Unet up cell")
    c.text(3.44, 6.12 + OA, "SCA (spatial–channel attention on the skip):  $s \\leftarrow s + s\\cdot\\sigma_s(s)"
                            "\\cdot\\sigma_c(s)$")
    # candidate blocks
    top = 5.92 + OA
    bottoms = [block_column(c, k, 0.05 + k * 1.425, top) for k in range(5)]
    ya = min(bottoms) - 0.22
    c.text(W / 2, ya, "(a) Search space: U-shaped network with eight searchable positions (genes $g_1$–$g_8$) and "
                      "five candidate blocks; GConv: grouped convolution (16 groups)")

    # ---- (b) pipeline -----------------------------------------------------------------------------------------
    yt, yb = ya - 0.22, 0.5
    ctr = (yt + yb) / 2
    hh = yt - yb
    c.box(1.415, ctr, 2.73, hh, fc=TRAIN_FREE_FILL, ec=TRAIN_FREE_EDGE, lw=0.8, ls=(0, (4, 2)), r=0.06, z=0)
    c.box(3.955, ctr, 2.15, hh, fc=TRAIN_FILL, ec=TRAIN_EDGE, lw=0.8, ls=(0, (4, 2)), r=0.06, z=0)
    c.box(6.12, ctr, 1.98, hh, fc=TRAIN_FILL, ec=TRAIN_EDGE, lw=0.8, ls=(0, (4, 2)), r=0.06, z=0)
    s = yt - 0.16
    c.text(1.415, s, "Stage 1: training-free NSGA-II search", weight="bold")
    c.text(3.955, s, "Stage 2: successive halving", weight="bold")
    c.text(6.12, s, "Stages 3 and 4", weight="bold")
    # stage 1
    x1, w1 = 1.5, 2.3
    rows = [(0.4, "Initial population $P_0$:\n20 random chromosomes"),
            (0.68, "Zero-cost evaluation (no training)\n$f_1$ = number of parameters → min\n"
                   "$f_2$ = 1 / (1 + ZiCo) → min\n(ZiCo: gradients of 2 training batches)"),
            (0.24, "Non-dominated sorting + crowding distance"),
            (0.52, "Binary tournament selection →\nuniform crossover (p = 0.9) →\nrandom-reset mutation (p = 1/8)"),
            (0.24, "Survival: best 20 of $P_t \\cup Q_t$"),
            (0.4, "16 candidates: final Pareto front,\nfilled up in NSGA-II survival order")]
    y = s - 0.2
    ext = []
    for i, (h, t) in enumerate(rows):
        if i == 5:
            y -= 0.12
        cy = y - h / 2
        c.box(x1, cy, w1, h, t, fc="white", ec=TRAIN_FREE_EDGE, lw=0.7)
        if i:
            c.line([(x1, ext[-1][1] - ext[-1][2] / 2), (x1, cy + h / 2)])
        ext.append((x1, cy, h))
        y = cy - h / 2 - 0.16
    c.text(x1 + 0.08, (ext[3][1] - ext[3][2] / 2 + ext[4][1] + ext[4][2] / 2) / 2, "offspring $Q_t$", ha="left")
    c.text(x1 + 0.08, (ext[4][1] - ext[4][2] / 2 + ext[5][1] + ext[5][2] / 2) / 2, "after 10 generations", ha="left")
    lx = x1 - w1 / 2 - 0.13
    c.line([(x1 - w1 / 2, ext[4][1]), (lx, ext[4][1]), (lx, ext[1][1]), (x1 - w1 / 2, ext[1][1])])
    c.text(lx - 0.07, (ext[1][1] + ext[4][1]) / 2, "generation t → t + 1", rotation=90)
    # stage 2: the actual rungs of the best BUSI run
    d = pf.run_dir()
    sh = load(d / "successive_halving.json")
    sel = load(d / "selection.json")["selected_id"]
    cx2 = 3.955
    y = s - 0.25
    sq = 0.1
    rung_y = []
    for r, rg in enumerate(sh["rungs"]):
        n, e = sh["sizes"][r], sh["rung_epochs"][r]
        c.text(cx2, y, f"Rung {r + 1}: {n} candidates × {e} epochs" if r == 0 else f"Rung {r + 1}: best {n} × {e} epochs")
        y -= 0.19
        ids = rg["trained"] if r == 0 else sorted(rg["trained"])
        ids = sorted(ids)
        x0 = cx2 - (len(ids) * 0.125 - 0.025) / 2
        for i, cid in enumerate(ids):
            kept = cid in rg["promoted"] if r < 2 else cid == sel
            fc = ("#eb6834" if r == 2 else COLOR["qenas"]) if kept else "#d0d0d0"
            c.box(x0 + i * 0.125 + sq / 2, y, sq, sq, fc=fc, ec="none", r=0.01)
        rung_y.append(y)
        if r < 2:
            c.line([(cx2, y - 0.07), (cx2, y - 0.25)])
            c.text(cx2 + 0.07, y - 0.16, "keep the best half", ha="left")
            y -= 0.38
    yg = rung_y[-1] - 0.07 - 0.17
    gates = ("Eligibility gates of the finalists\nG1 finite loss · G2 validation loss\ndecreased · G3 no over-fitting ·\n"
             "G4 significant learning trend ·\nG5 better than a trivial predictor")
    c.line([(cx2, rung_y[-1] - 0.07), (cx2, yg)])
    c.box(cx2, yg - 0.43, 1.95, 0.86, gates, fc="white", ec=TRAIN_EDGE, lw=0.7)
    ys = yg - 0.86 - 0.17
    c.line([(cx2, yg - 0.86), (cx2, ys)])
    c.box(cx2, ys - 0.3, 1.95, 0.6, "Dice-based selection: highest\nmean validation Dice of the last\n5 epochs → one "
                                    "architecture", fc="white", ec=TRAIN_EDGE, lw=0.7)
    sel_y = ys - 0.3
    # stage 3 and 4
    cx3, w3 = 6.12, 1.8
    y = s - 0.2
    boxes3 = [(0.86, "Stage 3: loss-weight selection\nα $\\in$ {0.2, 0.4, 0.5, 0.6, 0.8}\nL = α·BCE + (1 − α)·soft-mIoU\n"
                     "20 epochs per α; highest mean\nvalidation Dice of last 5 epochs"),
              (0.68, "Stage 4: final training\n120 epochs, AdamW, lr 10⁻³,\npoly schedule with warm-up,\n"
                     "best validation-Dice weights"),
              (0.68, "Single test evaluation\nDice, mIoU, IoU, HD95,\nsensitivity, parameters,\nFLOPs, inference time")]
    e3 = []
    for i, (h, t) in enumerate(boxes3):
        cy = y - h / 2
        c.box(cx3, cy, w3, h, t, fc="white", ec=TRAIN_EDGE, lw=0.7)
        if i:
            c.line([(cx3, e3[-1][0] - e3[-1][1] / 2), (cx3, cy + h / 2)])
        e3.append((cy, h))
        y = cy - h / 2 - 0.17
    ty = e3[-1][0] - e3[-1][1] / 2 - 0.1
    side = min(0.5, ty - yb - 0.08)
    c.image(img, cx3 - 0.35 - side / 2 - 0.05, ty - 0.04 - side, side, side)
    c.line([(cx3 - 0.35 + side / 2 - 0.02, ty - 0.04 - side / 2), (cx3 + 0.33 - side / 2 + 0.02, ty - 0.04 - side / 2)])
    c.image(pred, cx3 + 0.33 - side / 2 + 0.05, ty - 0.04 - side, side, side, cmap="gray")
    # links between the stages
    c.line([(x1 + w1 / 2, ext[5][1]), (2.83, ext[5][1]), (2.83, rung_y[0]), (cx2 - 1.0 - 0.02, rung_y[0])])
    c.line([(cx2 + 0.975, sel_y), (5.08, sel_y), (5.08, e3[0][0]), (cx3 - w3 / 2, e3[0][0])])
    # legend and caption
    leg = c.fig.legend([Patch(fc=TRAIN_FREE_FILL, ec=TRAIN_FREE_EDGE, ls=(0, (4, 2))),
                        Patch(fc=TRAIN_FILL, ec=TRAIN_EDGE, ls=(0, (4, 2)))],
                       ["training-free: no network weight is trained",
                        "trained on the training split, chosen on the validation split; test split used once"],
                       ncol=2, loc="center", bbox_to_anchor=(0.5, 0.31 / H), handlelength=1.6, columnspacing=1.2)
    leg.set_zorder(5)
    c.text(W / 2, 0.08, "(b) Search, selection and training pipeline (Stage 2 shows the rungs of the BUSI run)")
    c.limits()
    save(c.fig, D_ARC, "Complete architecture of QENAS")


# ------------------------------------------------------------------------------------------- found architectures
def position_rows(ds, s):
    import torch  # noqa: F401
    from qenas.models.network import QENASNet, count_parameters
    from qenas.models.darts_cell import genotype_name_for
    f = load(pf.pa.exp(ds, "qenas", s) / "final_metrics.json")
    cfg = load(pf.pa.exp(ds, "qenas", s) / "config.json")["config"]
    ch = f["architecture"]["chromosome"]
    net = QENASNet(ch, base_channels=int(cfg["model"]["base_channels"]), scale=int(cfg["model"]["scale"]),
                   genotype_name=genotype_name_for(ds))
    hw = tuple(cfg["data"]["image_size"])
    return f, ch, net.position_info(hw), count_parameters(net.stem), count_parameters(net.head), hw


def fig_found_architecture(ds="BUSI", s=None):
    s = s or pf.BEST[ds]
    f, ch, rows, p_stem, p_head, hw = position_rows(ds, s)
    W, H = W2, 4.1
    c = Canvas(W, H)
    lv = [3.55, 2.85, 2.15, 1.45, 0.75]
    bw, bh = 0.7, 0.62
    xs_enc = [0.45, 1.19, 1.93, 2.67, 3.41]
    xs_dec = [4.15, 4.89, 5.63, 6.37]

    def res(h, w):
        return f"{h}²" if h == w else f"{h}×{w}"

    def t(i, name, r):
        h, w = (int(v) for v in r["out_resolution"].split("x"))
        return (f"{name}\n{BLOCK_NAME[r['gene']].replace(' cell', '')}\n{r['out_channels']} ch, {res(h, w)}\n"
                f"{r['params'] / 1e6:.3f} M")

    enc_t = [f"Stem\n3×3 conv\n32 ch, {res(*hw)}\n{p_stem / 1e6:.3f} M"] + [t(i, f"E{i + 1}", rows[i]) for i in range(4)]
    dec_t = [t(4 + j, f"D{j + 1}", rows[4 + j]) for j in range(4)]
    u_network(c, xs_enc, xs_dec, lv, bw, bh, enc_t, dec_t,
              fills_enc=["#ececec"] + [BLOCK_FILL[g] for g in ch[:4]], fills_dec=[BLOCK_FILL[g] for g in ch[4:]],
              edges_enc=[INK2] + [BLOCK_COLOR[g] for g in ch[:4]], edges_dec=[BLOCK_COLOR[g] for g in ch[4:]],
              sca_w=0.34)
    c.box(6.88, lv[0] - 0.62, 0.42, 0.34, "1×1\nconv", fc="#ececec", lw=0.7)
    c.line([(xs_dec[3], lv[0] - bh / 2), (xs_dec[3], lv[0] - 0.62), (6.67, lv[0] - 0.62)])
    info = (f"{DSL[ds]}, best run (seed {s})\n{f['params_m']:.2f} M parameters · {f['gflops_per_image']:.2f} "
            f"GFLOPs per image\nα = {f['alpha']:.1f} · test Dice {100 * f['test']['dice']:.2f} %")
    c.text(6.0, 0.8, info, ha="center", linespacing=1.3)
    h = [Patch(fc=BLOCK_FILL[k], ec=BLOCK_COLOR[k]) for k in range(5)] + [Patch(fc="#f1f1f1", ec=MUTED)]
    c.fig.legend(h, [BLOCK_NAME[k] for k in range(5)] + ["SCA: spatial–channel attention"], ncol=6, loc="center",
                 bbox_to_anchor=(0.5, 0.12 / H), handlelength=1.2, columnspacing=1.0)
    c.limits()
    save(c.fig, D_ARC, f"Architecture found by QENAS on {DSL[ds]}")


def fig_all_found():
    runs = [(ds, s) for ds in pf.DATASETS for s in pf.SEEDS]
    W, rh = W2, 0.27
    H = 0.45 + rh * len(runs) + 0.42
    c = Canvas(W, H)
    x_ds, x_seed, x0, cw = 0.6, 1.35, 1.75, 0.5
    xp, xd = x0 + 8 * cw + 0.15 + 0.55, x0 + 8 * cw + 0.15 + 1.45
    yh = H - 0.2
    c.text(x_ds, yh, "Dataset")
    c.text(x_seed, yh, "Seed")
    for i in range(8):
        c.text(x0 + i * cw + cw / 2 + (0.1 if i >= 4 else 0), yh, f"{'E' if i < 4 else 'D'}{i % 4 + 1} ($g_{i + 1}$)")
    c.text(xp, yh, "Params (M)")
    c.text(xd, yh, "Test Dice (%)")
    for r, (ds, s) in enumerate(runs):
        y = H - 0.45 - r * rh - rh / 2 + 0.02
        f = load(pf.pa.exp(ds, "qenas", s) / "final_metrics.json")
        best = s == pf.BEST[ds]
        if s == pf.SEEDS[0]:
            c.text(x_ds, y - rh, DSL[ds])
            if r:
                c.line([(0.1, y + rh / 2), (W - 0.1, y + rh / 2)], arrow=False, color=AXIS, lw=0.5)
        c.text(x_seed, y, f"{s}" + (" *" if best else ""))
        for i, g in enumerate(f["architecture"]["chromosome"]):
            c.box(x0 + i * cw + cw / 2 + (0.1 if i >= 4 else 0), y, cw - 0.05, rh - 0.06, BLOCK_SHORT[g],
                  fc=BLOCK_FILL[g], ec=BLOCK_COLOR[g], lw=0.6, r=0.02)
        c.text(xp, y, f"{f['params_m']:.2f}")
        c.text(xd, y, f"{100 * f['test']['dice']:.2f}")
    h = [Patch(fc=BLOCK_FILL[k], ec=BLOCK_COLOR[k]) for k in range(5)]
    c.fig.legend(h, [f"{BLOCK_SHORT[k]}: {BLOCK_NAME[k]}" for k in range(5)] + [], ncol=5, loc="center",
                 bbox_to_anchor=(0.45, 0.17 / H), handlelength=1.2)
    c.text(W - 0.15, 0.17, "* best run", ha="right")
    c.limits()
    save(c.fig, D_REV, "Architectures selected by QENAS in all 12 runs")


if __name__ == "__main__":
    pass  # replaced by scripts/paper_architecture.py
    fig_found_architecture()
    fig_all_found()
