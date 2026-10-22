"""Architecture figures of QENAS 3.0 (style of scripts/paper_figures.py: Times New Roman, one font size, no titles).

    python scripts/paper_architecture.py

Writes three images to results/figures/3 Architecture/:
* QENAS network architecture      the U-shaped search space with every layer of every position, the skip connections
                                   with spatial-channel attention (SCA, detailed inset), the chromosome, head and loss;
* QENAS candidate blocks           the five candidate blocks with their layers (Residual, Dense, Inception, ConvNeXt,
                                   DARTS down and up cell of the BUSI genotype);
* QENAS search and training stages the four stages of the QENAS algorithm (search, successive halving and selection,
                                   loss weight, final training and test) with the GPU time of the BUSI run.
All drawing is in inches on a full-figure axes, so every text is exactly the global font size.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib.colors as mcolors  # noqa: E402
import paper_figures as pf  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch, Rectangle  # noqa: E402
from paper_diagrams import Canvas, example_images  # noqa: E402
from paper_figures import BLOCK_COLOR, BLOCK_SHORT, D_ARC, DSL, INK, INK2, MUTED, W2, load, save  # noqa: E402

DOWN_C, UP_C, SKIP_C, CONV_C = "#c8473f", "#24935f", "#6f6f6f", "#d99a00"
SCA_FILL, SCA_EDGE = "#eaf2fc", "#2a78d6"
FREE_FILL, FREE_EDGE, FREE_HEAD = "#edf4fd", "#2a78d6", "#cfe1f7"
TRAIN_FILL, TRAIN_EDGE, TRAIN_HEAD = "#fdf1e9", "#e0662d", "#f8d5c0"
SEL_C = "#e0662d"                               # the selected item, everywhere
OP_C = {"down_cweight": "#b8336a", "down_dep_conv": "#0f7f8c", "dil_conv": "#7a5a2f", "up_cweight": "#2a78d6"}
OP_TEXT = {"down_cweight": "down cweight: channel (SE) weighting, 3×3 conv, stride 2",
           "down_dep_conv": "down dep. conv: 3×3 depthwise + 1×1 conv, stride 2",
           "dil_conv": "dil. conv: 3×3 conv, dilation 2",
           "up_cweight": "up cweight: channel (SE) weighting, 3×3 transposed conv, ×2"}
BLOCK_FULL = {0: "Residual", 1: "Dense", 2: "Inception", 3: "ConvNeXt", 4: "DARTS cell"}


def mix(c, t):
    """Colour c blended towards white by t (0 = c, 1 = white)."""
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


# ------------------------------------------------------------------------------------------- primitives
def region(c, x0, y0, x1, y1, fc, ec, label=None, ls=(0, (4, 2)), lw=0.8, z=0):
    c.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.05", fc=fc,
                                  ec=ec, lw=lw, ls=ls, zorder=z))
    if label:
        c.text(x0 + 0.1, y1 - 0.13, label, ha="left", weight="bold", zorder=5)


def sym(c, cx, cy, kind, r=0.065, ec=INK2, z=6):
    c.ax.add_patch(Circle((cx, cy), r, fc="white", ec=ec, lw=0.7, zorder=z))
    k = r * 0.62
    if kind == "add":
        c.ax.plot([cx - k, cx + k], [cy, cy], color=ec, lw=0.8, zorder=z + 1)
        c.ax.plot([cx, cx], [cy - k, cy + k], color=ec, lw=0.8, zorder=z + 1)
    elif kind == "mul":
        q = k * 0.75
        c.ax.plot([cx - q, cx + q], [cy - q, cy + q], color=ec, lw=0.8, zorder=z + 1)
        c.ax.plot([cx - q, cx + q], [cy + q, cy - q], color=ec, lw=0.8, zorder=z + 1)
    elif kind == "cat":
        c.text(cx, cy - 0.003, "C", zorder=z + 1)
    elif kind == "sig":
        c.text(cx, cy + 0.005, "σ", zorder=z + 1)


def arrow(c, pts, color=INK2, lw=0.8, ls="-", z=2):
    c.line(pts, color=color, lw=lw, ls=ls, z=z)


def seg(c, pts, color=INK2, lw=0.8, ls="-", z=2):
    c.line(pts, arrow=False, color=color, lw=lw, ls=ls, z=z)


def lbox(c, cx, cy, w, h, text, fc="white", ec=INK2, lw=0.6, z=4):
    return c.box(cx, cy, w, h, text, fc=fc, ec=ec, lw=lw, z=z, r=0.03)


def thumb(c, img, x0, y0, s, cmap=None):
    c.image(img, x0, y0, s, s, cmap=cmap)


def slot(c, cx, cy, w, h, label, z=4):
    """Searchable block: the five candidate-block colours as stripes."""
    x0, y0 = cx - w / 2, cy - h / 2
    for i in range(5):
        c.ax.add_patch(Rectangle((x0 + i * w / 5, y0), w / 5, h, fc=mix(BLOCK_COLOR[i], 0.45), ec="none", zorder=z))
    c.ax.add_patch(Rectangle((x0, y0), w, h, fc="none", ec=INK, lw=0.8, zorder=z + 1))
    if label:
        c.ax.text(cx, cy, label, ha="center", va="center", zorder=z + 2,
                  bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))


def chain(c, x, y, items, gap=0.11, col=INK2):
    """Left-to-right chain of chips. items: (kind, width, text[, fill, edge]); kind 'box' (h 0.3) or 'slot'.
    Returns [(x0, cx, x1)] of the chips."""
    out = []
    for it in items:
        kind, w, text = it[:3]
        fc = it[3] if len(it) > 3 else "white"
        ec = it[4] if len(it) > 4 else INK2
        cx = x + w / 2
        if kind == "slot":
            slot(c, cx, y, w, 0.3, text)
        else:
            lbox(c, cx, y, w, 0.3, text, fc=fc, ec=ec)
        if out:
            arrow(c, [(out[-1][2], y), (x, y)], color=col)
        out.append((x, cx, x + w))
        x += w + gap
    return out


# ------------------------------------------------------------------------------------------- image 1: network
ROW_Y = [6.04, 5.18, 4.32, 3.44, 2.58]     # level centres: H, H/2, H/4, H/8, H/16
PH = 0.72                                   # panel height
ENC_X, DEC_X, BOT_X = (0.05, 2.5), (4.6, 7.11), (2.35, 4.75)
DBN, DTC, DCHAIN = 4.76, 5.24, 5.62             # decoder: BN and TConv (lower lane), start of the main chain
ENC_FILL, DEC_FILL = mix(DOWN_C, 0.93), mix(UP_C, 0.93)


def panel(c, x0, x1, yc, fc, ec, label, label_x=None):
    c.ax.add_patch(FancyBboxPatch((x0, yc - PH / 2), x1 - x0, PH, boxstyle="round,pad=0,rounding_size=0.05", fc=fc,
                                  ec=ec, lw=0.8, zorder=1))
    c.text(label_x if label_x is not None else x0 + 0.1, yc + PH / 2 - 0.11, label, ha="left", zorder=5)


def sca_inset(c, x0, y0, x1, y1):
    region(c, x0, y0, x1, y1, SCA_FILL, SCA_EDGE, "SCA: spatial–channel attention on each skip connection")
    yA, yB, yS, yC = y1 - 0.4, y1 - 0.66, y1 - 1.07, y1 - 1.5
    c.text(x0 + 0.2, yB, "skip")
    arrow(c, [(x0 + 0.36, yB), (x0 + 0.44, yB)])
    lbox(c, x0 + 0.61, yB, 0.34, 0.22, "BN")
    P = x0 + 0.92
    seg(c, [(x0 + 0.78, yB), (P, yB)])
    seg(c, [(P, yB), (P, yC)])
    c.text(x0 + 0.45, yS, "spatial\ngate $\\sigma_s$", linespacing=1.1)
    c.text(x0 + 0.45, yC, "channel\ngate $\\sigma_c$", linespacing=1.1)
    arrow(c, [(P, yS), (x0 + 1.1, yS)])
    lbox(c, x0 + 1.41, yS, 0.62, 0.34, "Conv 1×1\nC → 1")
    arrow(c, [(x0 + 1.72, yS), (x0 + 1.91, yS)])
    sym(c, x0 + 1.98, yS, "sig")
    xm1 = x0 + 2.46
    arrow(c, [(x0 + 2.045, yS), (xm1, yS), (xm1, yB - 0.065)])
    arrow(c, [(P, yC), (x0 + 1.05, yC)])
    lbox(c, x0 + 1.23, yC, 0.36, 0.22, "GAP")
    arrow(c, [(x0 + 1.41, yC), (x0 + 1.48, yC)])
    lbox(c, x0 + 1.78, yC, 0.6, 0.34, "Conv 1×1\nC → C/4")
    arrow(c, [(x0 + 2.08, yC), (x0 + 2.13, yC)])
    lbox(c, x0 + 2.32, yC, 0.38, 0.22, "ReLU")
    arrow(c, [(x0 + 2.51, yC), (x0 + 2.56, yC)])
    lbox(c, x0 + 2.86, yC, 0.6, 0.34, "Conv 1×1\nC/4 → C")
    arrow(c, [(x0 + 3.16, yC), (x0 + 3.2, yC)])
    xm2 = x0 + 3.27
    sym(c, xm2, yC, "sig")
    arrow(c, [(xm2, yC + 0.065), (xm2, yB - 0.065)])
    arrow(c, [(P, yB), (xm1 - 0.065, yB)])
    sym(c, xm1, yB, "mul")
    arrow(c, [(xm1 + 0.065, yB), (xm2 - 0.065, yB)])
    sym(c, xm2, yB, "mul")
    xa = x0 + 3.5
    arrow(c, [(xm2 + 0.065, yB), (xa - 0.065, yB)])
    sym(c, xa, yB, "add")
    arrow(c, [(P, yB), (P, yA), (xa, yA), (xa, yB + 0.065)])
    arrow(c, [(xa + 0.065, yB), (x1 - 0.04, yB)])
    c.text(P - 0.07, yB + 0.12, "$s$")
    c.text(x1 - 0.1, yB + 0.13, "$\\hat{s}$")
    c.text((x0 + x1) / 2, y0 + 0.15, "$\\hat{s} = s + s \\otimes \\sigma_s(s) \\otimes \\sigma_c(s)$")


def network_legend(c, x0, y0, x1, y1):
    region(c, x0, y0, x1, y1, "white", "#8c8c8c", ls="-", lw=0.6)
    xa, xb = x0 + 0.1, x0 + (x1 - x0) / 2 + 0.05
    y = y1 - 0.16
    st = 0.19

    def item(x, y, kind, text):
        if kind in ("down", "up", "skip"):
            col = {"down": DOWN_C, "up": UP_C, "skip": SKIP_C}[kind]
            arrow(c, [(x, y), (x + 0.3, y)], color=col, lw=1.2 if kind in ("down", "up") else 0.8,
                  ls=(0, (3, 1.6)) if kind == "skip" else "-")
        elif kind == "slot":
            slot(c, x + 0.15, y, 0.3, 0.15, None)
        else:
            sym(c, x + 0.15, y, kind)
        c.text(x + 0.4, y, text, ha="left")

    rows = [(("down", "encoder path (down-sampling)"), ("cat", "concatenation")),
            (("up", "decoder path (up-sampling)"), ("add", "addition")),
            (("skip", "skip connection"), ("mul", "multiplication")),
            (("slot", "searchable block (gene $g_i$)"), ("sig", "sigmoid"))]
    for left, right in rows:
        item(xa, y, *left)
        item(xb + 0.35, y, *right)
        y -= st
    for t in ("BN: batch norm.   GConv: grouped conv. (16 groups)", "TConv: transposed conv.   GAP: global avg. pooling",
              "$g_i$ = 4: a DARTS cell replaces the whole position", "(transition, SCA and block); its inputs: the two",
              "preceding feature maps (encoder) or skip and input", "from below (decoder)"):
        c.text(xa, y, t, ha="left")
        y -= st


def fig_network():
    W, H = W2, 7.3
    c = Canvas(W, H)
    img, gt, pred = example_images()
    # ---- top band: input, chromosome, prediction
    thumb(c, img, 0.21, 6.5, 0.5)
    c.text(0.46, 7.13, "input image")
    xs = [2.45 + i * 0.3 + (0.12 if i >= 4 else 0) for i in range(8)]
    c.text(2.35, 6.96, "chromosome", ha="right")
    for i, x in enumerate(xs):
        enc = i < 4
        c.box(x + 0.13, 6.96, 0.26, 0.24, f"$g_{i + 1}$", fc=ENC_FILL if enc else DEC_FILL,
              ec=DOWN_C if enc else UP_C, lw=0.8, r=0.03, z=4)
    c.text(3.73, 6.71, "$g_i \\in$ {0, 1, 2, 3, 4}: block of position $i$")
    c.text(3.73, 6.53, "$5^8$ = 390,625 architectures")
    lbox(c, 6.64, 6.72, 0.78, 0.22, "1×1 conv · σ", ec=CONV_C)
    thumb(c, pred, 5.6, 6.47, 0.5, cmap="gray")
    c.text(5.85, 7.13, "prediction")
    arrow(c, [(6.25, 6.72), (6.1, 6.72)], color=CONV_C, lw=1.0)
    # ---- encoder (stem, E1-E3), bottleneck (E4), decoder (D1-D4)
    outs = {}
    panel(c, *ENC_X, ROW_Y[0], "#f1f1f1", "#8c8c8c", "Stem · 32 ch · H × W", label_x=0.82)
    yu = ROW_Y[0] - 0.04
    ch = chain(c, 0.15, yu, [("box", 0.62, "3×3 conv"), ("box", 0.3, "BN"), ("box", 0.42, "ReLU")])
    arrow(c, [(0.46, 6.5), (0.46, yu + 0.15)])
    outs[0] = (ch[-1][1], ch[-1][2])
    for k in (1, 2, 3, 4):
        yc = ROW_Y[k]
        yu = yc - 0.04
        x0, x1 = BOT_X if k == 4 else ENC_X
        panel(c, x0, x1, yc, ENC_FILL, DOWN_C, (f"E{k} · gene $g_{k}$ · {32 * 2 ** k} ch · H/{2 ** k}" if k < 4 else "E4 · $g_4$ · 512 ch · H/16")
              + (" (bottleneck)" if k == 4 else ""), label_x=x0 + (0.45 if k == 4 else 0.5))
        ch = chain(c, x0 + 0.1, yu, [("box", 0.3, "BN"), ("box", 0.5, "AvgPool\n2×2"), ("box", 0.5, "3×3\nGConv"),
                                     ("slot", 0.62, f"$g_{k}$")])
        px, _ = outs[k - 1]
        gy = ROW_Y[k - 1] - 0.43
        arrow(c, [(px, ROW_Y[k - 1] - 0.19), (px, gy), (ch[0][1], gy), (ch[0][1], yu + 0.15)], color=DOWN_C, lw=1.2)
        outs[k] = (ch[-1][1], ch[-1][2])
    prev_out = outs[4]
    for k in (3, 2, 1, 0):                        # decoder bottom-up: D1 (level 3) ... D4 (level 0)
        j = 4 - k
        yc = ROW_Y[k]
        yu, yl = yc - 0.04, yc - 0.25
        panel(c, *DEC_X, yc, DEC_FILL, UP_C, f"D{j} · gene $g_{4 + j}$ · {32 * 2 ** k} ch · "
              + (f"H/{2 ** k}" if k else "H"))
        lbox(c, DBN, yl, 0.24, 0.17, "BN", ec=UP_C)
        lbox(c, DTC, yl, 0.62, 0.17, "2×2 TConv", ec=UP_C)
        arrow(c, [(DBN + 0.12, yl), (DTC - 0.31, yl)])
        sym(c, DTC, yu, "cat")
        arrow(c, [(DTC, yl + 0.085), (DTC, yu - 0.065)])
        ch = chain(c, DCHAIN, yu, [("box", 0.62, "1×1 conv\n2C → C"), ("slot", 0.62, f"$g_{4 + j}$")])
        arrow(c, [(DTC + 0.065, yu), (DCHAIN, yu)])
        if k == 3:                                # D1 receives the bottleneck output from its right side
            yb = ROW_Y[4] - 0.04
            arrow(c, [(prev_out[1], yb), (DBN, yb), (DBN, yl - 0.085)], color=UP_C, lw=1.2)
        else:
            px, gy = prev_out[0], yc - 0.43
            start_y = ROW_Y[k + 1] - 0.04 + 0.15
            arrow(c, [(px, start_y), (px, gy), (DBN, gy), (DBN, yl - 0.085)], color=UP_C, lw=1.2)
        prev_out = (ch[-1][1], ch[-1][2])
        # skip connection from the encoder of the same level through BN and SCA
        ex1 = outs[k][1]
        seg(c, [(ex1, yu), (2.8, yu)], color=SKIP_C, ls=(0, (3, 1.6)))
        lbox(c, 2.95, yu, 0.3, 0.22, "BN", ec=SKIP_C)
        arrow(c, [(3.1, yu), (3.35, yu)], color=SKIP_C)
        lbox(c, 3.6, yu, 0.5, 0.22, "SCA", fc=SCA_FILL, ec=SCA_EDGE)
        arrow(c, [(3.85, yu), (DTC - 0.065, yu)], color=SKIP_C, ls=(0, (3, 1.6)))
    arrow(c, [(prev_out[0], ROW_Y[0] - 0.04 + 0.15), (prev_out[0], 6.61)], color=CONV_C, lw=1.0)
    # ---- beside the bottleneck: block slot and loss
    y0, y1 = ROW_Y[4] - PH / 2, ROW_Y[4] + PH / 2
    region(c, 0.05, y0, 2.25, y1, "white", "#8c8c8c", ls="-", lw=0.6)
    c.text(0.15, y1 - 0.13, "searchable block: one of five", ha="left")
    for i in range(5):
        c.box(0.33 + i * 0.425, y0 + 0.24, 0.39, 0.22, f"{i} {BLOCK_SHORT[i]}", fc=mix(BLOCK_COLOR[i], 0.45),
              ec=BLOCK_COLOR[i], lw=0.7, r=0.03, z=4)
    region(c, 5.02, y0, 7.11, y1, "#fff7e3", CONV_C, ls="-", lw=0.6)
    thumb(c, pred, 5.1, y0 + 0.08, 0.44, cmap="gray")
    thumb(c, gt, 5.72, y0 + 0.08, 0.44, cmap="gray")
    arrow(c, [(5.56, y0 + 0.3), (5.7, y0 + 0.3)], color=INK2)
    arrow(c, [(5.7, y0 + 0.3), (5.56, y0 + 0.3)], color=INK2)
    c.text(5.32, y1 - 0.11, "pred.", zorder=6)
    c.text(5.94, y1 - 0.11, "label", zorder=6)
    c.text(6.63, y0 + 0.36, "loss:\n$\\alpha$ BCE + (1 − $\\alpha$)\n× soft-mIoU", linespacing=1.15)
    # ---- bottom band
    sca_inset(c, 0.05, 0.05, 3.75, 2.1)
    network_legend(c, 3.85, 0.05, 7.11, 2.1)
    c.limits()
    save(c.fig, D_ARC, "QENAS network architecture")


# ------------------------------------------------------------------------------------------- image 2: blocks
BLOCKS = {
    0: [("box", "3×3 GConv\nBN · ReLU"), ("box", "3×3 GConv\nBN · ReLU"), ("add", None), ("box", "ReLU")],
    1: [("box", "BN · ReLU\n3×3 GConv (32)"), ("cat", None), ("box", "BN · ReLU\n3×3 GConv (32)"), ("cat", None),
        ("box", "BN · ReLU\n3×3 GConv (32)"), ("cat", None), ("box", "1×1 conv\nC + 96 → C")],
    2: [("box", "3×3 GConv\nBN · ReLU"), ("branch", ["1×3\n3×1\nBN\nReLU", "1×5\n5×1\nBN\nReLU",
                                                    "1×7\n7×1\nBN\nReLU"]), ("cat", None), ("box", "1×1 conv\n3C → C")],
    3: [("box", "5×5 DWConv\nBN"), ("box", "Linear C → 4C\nGELU"), ("box", "Linear 4C → C\nBN"), ("add", None),
        ("box", "GELU\n1×1 conv")],
}


def block_panel(c, k, x0, top, bottom, col_w):
    col = BLOCK_COLOR[k]
    c.ax.add_patch(FancyBboxPatch((x0, bottom), col_w, top - bottom, boxstyle="round,pad=0,rounding_size=0.05",
                                  fc=mix(col, 0.9), ec=col, lw=0.8, zorder=1))
    c.box(x0 + col_w / 2, top - 0.14, col_w - 0.1, 0.22, f"{k} · {BLOCK_FULL[k]}", fc=mix(col, 0.55), ec=col, lw=0.7,
          z=4, weight="bold", r=0.03)
    bw = col_w - (0.2 if k == 2 else 0.42)
    cx, lane = x0 + 0.1 + bw / 2, x0 + col_w - 0.13
    gap = 0.09
    y = top - 0.36
    c.text(cx, y, "x: C × H × W")
    entry = y - 0.1
    prev = entry
    centers = []
    for kind, val in BLOCKS[k]:
        if kind == "box":
            h = 0.34
            cy = prev - gap - h / 2
            lbox(c, cx, cy, bw, h, val, ec=col)
            arrow(c, [(cx, prev), (cx, cy + h / 2)])
            prev = cy - h / 2
        elif kind in ("add", "cat"):
            cy = prev - gap - 0.065
            sym(c, cx, cy, kind)
            arrow(c, [(cx, prev), (cx, cy + 0.065)])
            prev = cy - 0.065
        else:
            h = 0.64
            cy = prev - 0.12 - h / 2
            n = len(val)
            bwi = (bw - 0.05 * (n - 1)) / n
            for i, t in enumerate(val):
                bx = cx - bw / 2 + bwi / 2 + i * (bwi + 0.05)
                lbox(c, bx, cy, bwi, h, t, ec=col)
                arrow(c, [(cx, prev), (bx, cy + h / 2)])
                arrow(c, [(bx, cy - h / 2), (cx, cy - h / 2 - 0.1)])
            prev = cy - h / 2 - 0.1
            centers.append(("branch", cy))
            continue
        centers.append((kind, cy))
    arrow(c, [(cx, prev), (cx, prev - 0.13)])
    c.text(cx, prev - 0.24, "y: C × H × W")
    if k in (0, 3):                                               # identity path
        ay = [cy for kind, cy in centers if kind == "add"][0]
        arrow(c, [(cx, entry), (lane, entry), (lane, ay), (cx + 0.065, ay)])
    if k == 1:                                                    # dense connections
        cats = [cy for kind, cy in centers if kind == "cat"]
        boxes = [cy for kind, cy in centers if kind == "box"]
        seg(c, [(cx, entry), (lane, entry), (lane, cats[-1])])
        for cy in cats:
            arrow(c, [(lane, cy), (cx + 0.065, cy)])
        lane2 = cx + bw / 2 + 0.1
        for i in range(2):
            seg(c, [(cx + bw / 2, boxes[i]), (lane2, boxes[i]), (lane2, cats[-1])], color=MUTED, lw=0.6)
        for cy in cats[1:]:
            seg(c, [(lane2, cy), (cx + 0.065, cy)], color=MUTED, lw=0.6)


CELLS = {   # BUSI genotype (Mixed-GGNAS / NAS-Unet), node -> two (op, input) edges
    "down": {2: [("down_cweight", 0), ("down_cweight", 1)], 3: [("down_dep_conv", 0), ("dil_conv", 2)],
             4: [("down_cweight", 1), ("dil_conv", 2)], 5: [("dil_conv", 3), ("dil_conv", 4)]},
    "up": {2: [("dil_conv", 0), ("up_cweight", 1)], 3: [("dil_conv", 0), ("dil_conv", 2)],
           4: [("dil_conv", 3), ("dil_conv", 2)], 5: [("dil_conv", 3), ("dil_conv", 4)]},
}
CELL_POS = {"down": {2: (1.35, 0.0), 3: (2.05, 0.45), 4: (2.05, -0.45), 5: (2.7, 0.0)},
            "up": {2: (1.3, 0.0), 3: (1.95, 0.45), 4: (2.4, -0.42), 5: (2.95, 0.22)}}


def cell_panel(c, kind, x0, y0, x1, y1):
    col = BLOCK_COLOR[4]
    region(c, x0, y0, x1, y1, mix(col, 0.9), col,
           "4 · DARTS cell, " + ("down cell (encoder)" if kind == "down" else "up cell (decoder)"))
    ym = (y0 + y1) / 2 + 0.05
    ins = {0: "$s_0$: two levels up" if kind == "down" else "$s_0$: skip feature",
           1: "$s_1$: previous level" if kind == "down" else "$s_1$: from below"}
    pos = {0: (x0 + 0.55, ym + 0.42), 1: (x0 + 0.55, ym - 0.42)}
    for i, (x, y) in pos.items():
        lbox(c, x, y, 0.95, 0.22, ins[i], ec=col)
    nodes = {n: (x0 + dx, ym + dy) for n, (dx, dy) in CELL_POS[kind].items()}
    r = 0.095
    for n, edges in CELLS[kind].items():
        bx, by = nodes[n]
        for op, src in edges:
            if src in pos:
                sx, sy = pos[src][0] + 0.475, pos[src][1]
            else:
                sx, sy = nodes[src]
            dx, dy = bx - sx, by - sy
            d = (dx * dx + dy * dy) ** 0.5
            s0 = 0 if src in pos else r
            arrow(c, [(sx + s0 * dx / d, sy + s0 * dy / d), (bx - r * dx / d, by - r * dy / d)], color=OP_C[op],
                  lw=1.1)
    bus = ym - 0.85
    xc = x0 + 3.15
    for n, (x, y) in nodes.items():
        seg(c, [(x, y - r), (x, bus)], color=MUTED, lw=0.7, ls=(0, (2, 1.2)), z=1)
        c.ax.add_patch(Circle((x, y), r, fc="white", ec=col, lw=0.9, zorder=6))
        c.text(x, y, str(n - 1), zorder=7)
    xs = [x for x, _ in nodes.values()]
    seg(c, [(min(xs), bus), (xc - 0.065, bus)], color=MUTED, lw=0.7)
    sym(c, xc, bus, "cat")
    arrow(c, [(xc + 0.065, bus), (x1 - 0.06, bus)])
    c.text(x1 - 0.06, bus - 0.17, "output: 4 × C/4 = C", ha="right")
    c.text(x0 + 0.1, ym + 0.7, "each input: ReLU · 1×1 conv · GN", ha="left")


def fig_blocks():
    W, H = W2, 6.15
    c = Canvas(W, H)
    top, bottom = H - 0.05, 2.75
    cw = 1.7
    for k in range(4):
        block_panel(c, k, 0.05 + k * 1.77, top, bottom, cw)
    cell_panel(c, "down", 0.05, 0.5, 3.55, 2.65)
    cell_panel(c, "up", 3.61, 0.5, 7.11, 2.65)
    for i, op in enumerate(OP_TEXT):
        x = 0.1 + (i % 2) * 3.56
        y = 0.33 - (i // 2) * 0.2
        arrow(c, [(x, y), (x + 0.3, y)], color=OP_C[op], lw=1.1)
        c.text(x + 0.38, y, OP_TEXT[op], ha="left")
    c.limits()
    save(c.fig, D_ARC, "QENAS candidate blocks")


# ------------------------------------------------------------------------------------------- image 3: stages
def mini_axes(c, x0, y0, w, h, xlabel, ylabel):
    ax = c.fig.add_axes([x0 / c.w, y0 / c.h, w / c.w, h / c.h])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
        ax.spines[s].set_color(INK2)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    return ax


def chip(c, x0, x1, y, text, fc, ec):
    c.box((x0 + x1) / 2, y, x1 - x0 - 0.08, 0.22, text, fc=fc, ec=ec, lw=0.7, z=4, weight="bold", r=0.04)


def flow_arrow(c, x0, x1, y):
    c.ax.add_patch(FancyArrow(x0, y, x1 - x0, 0, width=0.07, head_width=0.17, head_length=0.1,
                              length_includes_head=True, fc=INK2, ec="none", zorder=5))


def fig_stages():
    W, H = W2, 4.05
    c = Canvas(W, H)
    ds, seed = pf.STAGE_RUN
    d = pf.pa.exp(ds, "qenas", seed)
    img, gt, pred = example_images()
    PT, PB = H - 0.05, 0.82
    S1, S2, S3, S4 = (0.05, 2.5), (2.72, 4.37), (4.59, 5.69), (5.91, 7.11)
    mid = (PT + PB) / 2
    status, sel, sh = pf.candidate_status(d)
    # ---- stage 1: training-free NSGA-II search
    region(c, S1[0], PB, S1[1], PT, FREE_FILL, FREE_EDGE)
    chip(c, *S1, PT - 0.16, "1 · Training-free NSGA-II search", FREE_HEAD, FREE_EDGE)
    cx1 = (S1[0] + S1[1]) / 2
    yp, ys_, ye = PT - 0.53, PT - 1.15, PT - 1.77
    lbox(c, cx1, yp, 2.2, 0.34, "Population $P_t$: 20 architectures\n($P_0$: random chromosomes)", ec=FREE_EDGE)
    lbox(c, 0.6, ys_, 0.92, 0.6, "Selection,\ncrossover,\nmutation → $Q_t$", ec=FREE_EDGE)
    lbox(c, 1.95, ys_, 0.92, 0.6, "Sort $P_t \\cup Q_t$:\nnon-dominated\n+ crowding", ec=FREE_EDGE)
    lbox(c, cx1, ye, 2.2, 0.34, "Zero-cost evaluation of $Q_t$ (no training)\n$f_1$ = parameters,  "
                                "$f_2$ = 1 / (1 + ZiCo)", ec=FREE_EDGE)
    arrow(c, [(0.6, yp - 0.17), (0.6, ys_ + 0.3)], color=FREE_EDGE, lw=1.0)
    arrow(c, [(0.6, ys_ - 0.3), (0.6, ye + 0.17)], color=FREE_EDGE, lw=1.0)
    arrow(c, [(1.95, ye + 0.17), (1.95, ys_ - 0.3)], color=FREE_EDGE, lw=1.0)
    arrow(c, [(1.95, ys_ + 0.3), (1.95, yp - 0.17)], color=FREE_EDGE, lw=1.0)
    c.text(cx1, ys_, "10\ngen.", linespacing=1.15)
    arch = load(d / "search_results.json")
    hist = pf.pa.read_csv(d / "search_population_history.csv")
    last = max(int(r["generation"]) for r in hist)
    front = sorted((float(r["params"]) / 1e6, float(r["zico"])) for r in hist
                   if int(r["generation"]) == last and r["nondominated"] == "True")
    cands = pf.pa.read_csv(d / "pareto_candidates.csv")
    ax = mini_axes(c, 0.36, 1.05, 1.15, 0.72, "parameters", "ZiCo")
    ax.scatter([a["params"] / 1e6 for a in arch], [a["zico"] for a in arch], s=4, color=mix(FREE_EDGE, 0.55), lw=0)
    ax.plot(*zip(*front), color=INK, lw=0.5)
    oth = [r for r in cands if int(r["candidate_id"]) != sel]
    ax.scatter([float(r["params"]) / 1e6 for r in oth], [float(r["zico"]) for r in oth], s=9, marker="s",
               facecolor="none", edgecolor=INK, lw=0.5)
    sr = [r for r in cands if int(r["candidate_id"]) == sel][0]
    ax.scatter([float(sr["params"]) / 1e6], [float(sr["zico"])], s=9, marker="s", color=SEL_C, lw=0.5, zorder=3)
    c.text(2.0, 1.45, "final Pareto\nfront → 16\ncandidates", linespacing=1.15)
    # ---- stage 2: successive halving and selection
    region(c, S2[0], PB, S2[1], PT, TRAIN_FILL, TRAIN_EDGE)
    chip(c, *S2, PT - 0.16, "2 · Successive halving", TRAIN_HEAD, TRAIN_EDGE)
    cx2 = (S2[0] + S2[1]) / 2
    ty = [PT - 0.45, PT - 0.85, PT - 1.25]
    for r, rg in enumerate(sh["rungs"]):
        n, e = sh["sizes"][r], sh["rung_epochs"][r]
        c.text(cx2, ty[r], f"rung {r + 1}: {n} × {e} epochs")
        ids = sorted(rg["trained"])
        x0 = cx2 - (len(ids) * 0.094 - 0.019) / 2
        for k, cid in enumerate(ids):
            kept = cid in rg["promoted"] if r < 2 else cid == sel
            fc = (SEL_C if r == 2 else FREE_EDGE) if kept else "#cfcfcf"
            c.box(x0 + k * 0.094 + 0.0375, ty[r] - 0.16, 0.075, 0.075, fc=fc, ec="none", r=0.01, z=4)
    yg = PT - 1.95
    arrow(c, [(cx2, ty[2] - 0.22), (cx2, yg + 0.3)])
    lbox(c, cx2, yg, 1.5, 0.6, "Gates G1–G5: finite loss,\nfalling val. loss, no over-\nfitting, learning trend",
         ec=TRAIN_EDGE)
    ysel = yg - 0.6
    arrow(c, [(cx2, yg - 0.3), (cx2, ysel + 0.17)])
    lbox(c, cx2, ysel, 1.5, 0.34, "highest mean val. Dice\nof the last 5 epochs", ec=TRAIN_EDGE)
    chrom = load(d / "final_metrics.json")["architecture"]["chromosome"]
    yc = ysel - 0.35
    arrow(c, [(cx2, ysel - 0.17), (cx2, yc + 0.08)])
    for k, g in enumerate(chrom):
        x = cx2 - 0.56 + k * 0.135 + (0.04 if k >= 4 else 0)
        c.box(x + 0.06, yc, 0.12, 0.15, str(g), fc=mix(BLOCK_COLOR[g], 0.5), ec=BLOCK_COLOR[g], lw=0.6, r=0.02, z=4)
    # ---- stage 3: loss weight
    region(c, S3[0], PB, S3[1], PT, TRAIN_FILL, TRAIN_EDGE)
    chip(c, *S3, PT - 0.16, "3 · Loss weight", TRAIN_HEAD, TRAIN_EDGE)
    cx3 = (S3[0] + S3[1]) / 2
    c.text(cx3, PT - 0.55, "L = $\\alpha$ BCE +\n(1 − $\\alpha$) soft-mIoU", linespacing=1.15)
    runs = sorted(load(d / "loss_weights.json")["runs"], key=lambda r: r["alpha"])
    ax = mini_axes(c, cx3 - 0.3, 1.85, 0.78, 0.95, "$\\alpha$", "val. Dice")
    ax.plot([r["alpha"] for r in runs], [r["criterion"] for r in runs], color=INK2, lw=0.8, marker="o", ms=2.6)
    ch = [r for r in runs if r["chosen"]][0]
    ax.plot([ch["alpha"]], [ch["criterion"]], marker="o", ms=3.2, color=SEL_C, ls="", zorder=3)
    ax.set_xticks([0.2, 0.5, 0.8])
    ax.tick_params(axis="x", length=2, pad=1)
    ax.margins(x=0.12, y=0.25)
    c.text(cx3, 1.15, f"20 epochs per $\\alpha$;\nselected $\\alpha$ = {ch['alpha']:.1f}", linespacing=1.15)
    # ---- stage 4: final training and test
    region(c, S4[0], PB, S4[1], PT, TRAIN_FILL, TRAIN_EDGE)
    chip(c, *S4, PT - 0.16, "4 · Final training", TRAIN_HEAD, TRAIN_EDGE)
    cx4 = (S4[0] + S4[1]) / 2
    yt = PT - 0.86
    lbox(c, cx4, yt, 1.08, 0.9, "120 epochs,\nAdamW (lr $10^{-3}$),\npoly schedule,\nbest val. Dice\nweights kept",
         ec=TRAIN_EDGE)
    yte = yt - 0.88
    arrow(c, [(cx4, yt - 0.45), (cx4, yte + 0.3)])
    lbox(c, cx4, yte, 1.08, 0.6, "single test pass:\nDice, mIoU,\nHD95, …", ec=TRAIN_EDGE)
    arrow(c, [(cx4, yte - 0.3), (cx4, 1.3)])
    thumb(c, img, cx4 - 0.47, 0.9, 0.4)
    thumb(c, pred, cx4 + 0.07, 0.9, 0.4, cmap="gray")
    arrow(c, [(cx4 - 0.06, 1.1), (cx4 + 0.06, 1.1)])
    # ---- flow between the stages
    for (_, a), (b, _) in zip((S1, S2, S3), (S2, S3, S4)):
        flow_arrow(c, a + 0.03, b - 0.03, mid)
    # ---- GPU time of this run and colour key
    ds, seed = "IDRID", pf.BEST["IDRID"]           # all times in the paper: IDRiD (the published search times)
    parts = pf.qenas_stage_hours(ds, seed)
    tot = sum(parts)
    yr = 0.5
    c.text(0.08, yr, f"GPU time ({DSL[ds]} run):", ha="left")
    x, span = 1.45, 5.15
    fills = [FREE_HEAD, mix(TRAIN_EDGE, 0.72), mix(TRAIN_EDGE, 0.58), mix(TRAIN_EDGE, 0.44)]
    for k, (p, fc) in enumerate(zip(parts, fills)):
        w = span * p / tot
        c.box(x + w / 2, yr, w, 0.22, fc=fc, ec=FREE_EDGE if k == 0 else TRAIN_EDGE, lw=0.6, r=0.0, z=3)
        lab = f"stage {k + 1}: {60 * p:.0f} min"
        c.text(x + w / 2, yr if w > 0.7 else yr + 0.21, lab, zorder=5, ha="center")
        x += w
    c.text(6.66, yr, f"≈ {round(60 * tot / 10) * 10:.0f} min" if tot < 1 else f"≈ {tot:.1f} h", ha="left")
    ky = 0.15
    keys = [(FREE_EDGE, "promoted candidate"), ("#cfcfcf", "eliminated candidate"), (SEL_C, "selected")]
    kx = 0.1
    for col, t in keys:
        c.box(kx + 0.04, ky, 0.08, 0.08, fc=col, ec="none", r=0.01, z=4)
        c.text(kx + 0.13, ky, t, ha="left")
        kx += 0.2 + 0.056 * len(t)
    for fc, ec, t in ((FREE_FILL, FREE_EDGE, "training-free"), (TRAIN_FILL, TRAIN_EDGE, "trains networks")):
        c.ax.add_patch(Rectangle((kx, ky - 0.06), 0.2, 0.12, fc=fc, ec=ec, lw=0.6, ls=(0, (3, 1.5)), zorder=4))
        c.text(kx + 0.27, ky, t, ha="left")
        kx += 0.35 + 0.056 * len(t)
    c.limits()
    save(c.fig, D_ARC, "QENAS search and training stages")


if __name__ == "__main__":
    fig_network()
    fig_blocks()
    fig_stages()
