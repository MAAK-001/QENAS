"""U-shaped architecture diagram of a QENAS network (block type, channels, resolution, kernel, params)."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from ..models.chromosome import BLOCK_NAMES, chromosome_to_str
from .style import AXIS, BLOCK_COLORS, INK, INK_2, MUTED, NEUTRAL_FILL, apply_style, plt, save


def plot_architecture(summary: Dict[str, Any], path: Path, title: str = "") -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.6, 7.7)
    ax.axis("off")
    W, H = 2.75, 1.05
    # Encoder descends on the left, decoder ascends on the right; each decoder box sits at the
    # level (resolution) of the encoder feature it receives as skip connection.
    stem_xy = (0.5, 5.2)
    enc_xy = [(0.5 + 0.55 * k, 5.2 - 1.25 * (k + 1)) for k in range(4)]
    dec_xy = [(8.1 + 0.55 * j, 1.45 + 1.25 * j) for j in range(4)]
    head_xy = (dec_xy[3][0], 6.45)

    def box(xy, color, lines, w=W, h=H, text_color=INK):
        x, y = xy
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.95))
        ax.text(x + 0.1, y + h - 0.12, lines[0], fontsize=9, weight="bold", color=text_color, va="top")
        for i, ln in enumerate(lines[1:]):
            ax.text(x + 0.1, y + h - 0.38 - 0.2 * i, ln, fontsize=7.2, color=text_color, va="top")

    def arrow(p, q, color=MUTED, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, linewidth=1.0,
                                     linestyle=ls, shrinkA=2, shrinkB=2))

    pos = summary["positions"]
    box(stem_xy, NEUTRAL_FILL, ["Input → stem", f"3x3 conv, {summary['base_channels']} ch, BN, ReLU",
                                f"{summary['input']}", f"params {summary['stem_params']:,}"])
    for k in range(4):
        r = pos[k]
        c = BLOCK_COLORS[r["gene"]]
        box(enc_xy[k], c, [f"Block {k + 1} · enc{k + 1} · {r['block']}",
                           f"{r['in_channels']}→{r['out_channels']} ch, {r['in_resolution']}→{r['out_resolution']}",
                           textwrap.shorten(r["kernel_scale"], 38, placeholder="…"),
                           f"params {r['params']:,}"], text_color="white" if r["gene"] in (0, 4) else INK)
    for j in range(4):
        r = pos[4 + j]
        c = BLOCK_COLORS[r["gene"]]
        box(dec_xy[j], c, [f"Block {5 + j} · dec{j + 1} · {r['block']}",
                           f"{r['in_channels']}→{r['out_channels']} ch, {r['in_resolution']}→{r['out_resolution']}",
                           textwrap.shorten(r["kernel_scale"], 38, placeholder="…"),
                           f"params {r['params']:,}"], text_color="white" if r["gene"] in (0, 4) else INK)
    box(head_xy, NEUTRAL_FILL, ["Segmentation head", "1x1 conv → 1 logit / pixel", "sigmoid > 0.5",
                                f"params {summary['head_params']:,}"])
    # main path
    chain = [stem_xy] + enc_xy
    for a, b in zip(chain[:-1], chain[1:]):
        arrow((a[0] + 0.6, a[1]), (b[0] + 0.6, b[1] + H))
    arrow((enc_xy[3][0] + W, enc_xy[3][1] + H / 2), (dec_xy[0][0], dec_xy[0][1] + H / 2))
    for a, b in zip(dec_xy[:-1], dec_xy[1:]):
        arrow((a[0] + W - 0.6, a[1] + H), (b[0] + W - 0.6, b[1]))
    arrow((dec_xy[3][0] + W - 0.6, dec_xy[3][1] + H), (head_xy[0] + W - 0.6, head_xy[1]))
    # skips: E3->dec1, E2->dec2, E1->dec3, E0(stem)->dec4
    srcs = [enc_xy[2], enc_xy[1], enc_xy[0], stem_xy]
    for s, d in zip(srcs, dec_xy):
        arrow((s[0] + W, s[1] + H * 0.35), (d[0], d[1] + H * 0.35), color=AXIS, ls="--")
    ax.text(6.4, 6.9, "dashed: skip connections (encoder feature at the decoder's output resolution)",
            ha="center", fontsize=8, color=INK_2)
    handles = [Patch(color=BLOCK_COLORS[g], label=f"{g} {BLOCK_NAMES[g]}") for g in range(5)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, title="block types")
    ttl = title or "Selected QENAS architecture"
    ax.set_title(f"{ttl}\nchromosome {chromosome_to_str(summary['chromosome'])} · "
                 f"{summary['total_params'] / 1e6:.3f} M parameters · {summary['gmacs']:.2f} GMACs · "
                 f"scale {summary['scale']} · DARTS genotype {summary['darts_genotype']}", fontsize=11)
    save(fig, path)
