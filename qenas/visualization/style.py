"""Shared publication style: validated categorical order, sequential ramp, recessive chrome."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated reference palette; never cycled or re-ordered).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Sequential single-hue ramp (blue), ordinal-safe start (step 250) -> dark.
SEQ_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
            "#104281", "#0d366b"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL_FILL = "#d6d5ce"
SURFACE = "#ffffff"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]

# Block-type identity colours (5 categories -> first five slots, fixed mapping).
BLOCK_COLORS = {0: SERIES[0], 1: SERIES[1], 2: SERIES[2], 3: SERIES[3], 4: SERIES[6]}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "font.family": ["DejaVu Sans"],
        "text.color": INK,
    })


def series_color(i: int) -> str:
    """Colour for the i-th entity (0-based). Beyond 8 entities callers must add secondary encoding."""
    return SERIES[i % len(SERIES)]


def save(fig, path: Path, formats: Iterable[str] = ("png", "pdf")) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"), bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
