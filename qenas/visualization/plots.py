"""Publication figures for every QENAS stage (PNG at 300 dpi + vector PDF)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

from .style import (AXIS, GRID, INK, INK_2, MARKERS, MUTED, NEUTRAL_FILL, SEQ_BLUE, SERIES, apply_style, plt,
                    save, series_color)

F2_LABEL = "SynFlow objective  f₂ = 1 / (1 + ln(1+S))   (lower is better)"
LOGS_LABEL = "log-SynFlow  ln(1 + S)   (higher is better)"


def _staircase(points: np.ndarray) -> np.ndarray:
    """Attainment staircase through a minimisation front sorted by f1."""
    p = points[np.argsort(points[:, 0])]
    xs, ys = [p[0, 0]], [p[0, 1]]
    for a, b in p[1:]:
        xs += [a, a]
        ys += [ys[-1], b]
    return np.column_stack([xs, ys])


# ---------------------------------------------------------------------------------------------- search
def plot_pareto_population(pop_rows: List[Dict[str, Any]], archive_rows: List[Dict[str, Any]], generation: int,
                           n_generations: int, path: Path, highlight: Optional[List[Dict[str, Any]]] = None,
                           title: Optional[str] = None, candidate_epochs: int = 20) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if archive_rows:
        a = np.array([[r["params"] / 1e6, r["f2_synflow"]] for r in archive_rows])
        ax.scatter(a[:, 0], a[:, 1], s=14, color=NEUTRAL_FILL, edgecolor="none", label="evaluated so far", zorder=1)
    pop = np.array([[r["params"] / 1e6, r["f2_synflow"]] for r in pop_rows])
    nd = np.array([r["nondominated"] for r in pop_rows])
    if (~nd).any():
        ax.scatter(pop[~nd, 0], pop[~nd, 1], s=26, color=MUTED, edgecolor="white", linewidth=0.8,
                   label="population (dominated)", zorder=2)
    if nd.any():
        st = _staircase(pop[nd])
        ax.plot(st[:, 0], st[:, 1], color=SERIES[0], linewidth=1.2, alpha=0.8, zorder=3)
        ax.scatter(pop[nd, 0], pop[nd, 1], s=40, color=SERIES[0], edgecolor="white", linewidth=1.0,
                   label=f"non-dominated (rank 0): {int(nd.sum())}", zorder=4)
    if highlight:
        h = np.array([[r["params"] / 1e6, r["f2_synflow"]] for r in highlight])
        ax.scatter(h[:, 0], h[:, 1], s=110, facecolor="none", edgecolor=SERIES[1], linewidth=1.6,
                   label=f"selected for {candidate_epochs}-epoch evaluation", zorder=5)
        for r in highlight:
            ax.annotate(f"C{r['candidate_id']}", (r["params"] / 1e6, r["f2_synflow"]), textcoords="offset points",
                        xytext=(6, 5), fontsize=8, color=INK_2)
    ax.set_xlabel("Trainable parameters (millions)   (lower is better)")
    ax.set_ylabel(F2_LABEL)
    gen_txt = "initial population (generation 0)" if generation == 0 else f"generation {generation}/{n_generations}"
    ax.set_title(title or f"NSGA-II population — {gen_txt}")
    ax.legend(loc="upper right")
    save(fig, path)


def plot_pareto_evolution(history: List[Dict[str, Any]], n_generations: int, path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    gens = sorted({r["generation"] for r in history})
    idx = np.linspace(0, len(SEQ_BLUE) - 1, num=max(1, len(gens))).round().astype(int)
    for k, g in enumerate(gens):
        rows = [r for r in history if r["generation"] == g and r["nondominated"]]
        if not rows:
            continue
        p = np.array([[r["params"] / 1e6, r["f2_synflow"]] for r in rows])
        st = _staircase(p)
        c = SEQ_BLUE[idx[k]]
        last = g == gens[-1]
        ax.plot(st[:, 0], st[:, 1], color=c, linewidth=2.0 if last else 1.1, zorder=2 + k)
        ax.scatter(p[:, 0], p[:, 1], s=34 if last else 16, color=c, edgecolor="white", linewidth=0.7, zorder=2 + k,
                   label=("gen 0 (initial)" if g == 0 else f"gen {g}") if (g in (gens[0], gens[-1]) or
                                                                           len(gens) <= 6 or k % 2 == 0) else None)
    ax.set_xlabel("Trainable parameters (millions)   (lower is better)")
    ax.set_ylabel(F2_LABEL)
    ax.set_title("Pareto-front evolution across generations")
    ax.legend(title="non-dominated front", loc="upper right", ncol=2 if len(gens) > 6 else 1)
    save(fig, path)


def plot_front_log_synflow(front: List[Dict[str, Any]], archive_rows: List[Dict[str, Any]], path: Path,
                           candidates: Optional[List[Dict[str, Any]]] = None) -> None:
    """Same final front on the interpretable log-SynFlow scale (monotone transform of f2)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    a = np.array([[r["params"] / 1e6, r["synflow_log"]] for r in archive_rows])
    ax.scatter(a[:, 0], a[:, 1], s=14, color=NEUTRAL_FILL, edgecolor="none", label="all evaluated architectures")
    f = np.array([[r["params"] / 1e6, r["synflow_log"]] for r in front])
    o = np.argsort(f[:, 0])
    ax.plot(f[o, 0], f[o, 1], color=SERIES[0], linewidth=1.2)
    ax.scatter(f[:, 0], f[:, 1], s=40, color=SERIES[0], edgecolor="white", linewidth=1.0, label="final Pareto front")
    if candidates:
        for r in candidates:
            ax.scatter([r["params"] / 1e6], [r["synflow_log"]], s=110, facecolor="none", edgecolor=SERIES[1],
                       linewidth=1.6)
            ax.annotate(f"C{r['candidate_id']}", (r["params"] / 1e6, r["synflow_log"]), textcoords="offset points",
                        xytext=(6, -10), fontsize=8, color=INK_2)
    ax.set_xlabel("Trainable parameters (millions)")
    ax.set_ylabel(LOGS_LABEL)
    ax.set_title("Final Pareto front on the log-SynFlow scale")
    ax.legend(loc="lower right")
    save(fig, path)


# ---------------------------------------------------------------------------------------------- candidates
def _cand_label(c: Dict[str, Any]) -> str:
    return f"C{c['candidate_id']}"


def plot_candidate_curves(cands: List[Dict[str, Any]], metric: str, ylabel: str, title: str, path: Path,
                          selected_id: Optional[int]) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, c in enumerate(cands):
        h = c.get("history") or []
        if not h:
            continue
        e = [r["epoch"] for r in h]
        v = [r[metric] for r in h]
        sel = c["candidate_id"] == selected_id
        ax.plot(e, v, color=series_color(i), linewidth=2.6 if sel else 1.3, alpha=1.0 if sel else 0.85,
                marker=MARKERS[i % len(MARKERS)], markersize=4 if sel else 3, markevery=max(1, len(e) // 5),
                label=_cand_label(c) + ("  (selected)" if sel else "") + (" — diverged" if c.get("diverged") else ""),
                zorder=5 if sel else 2)
        ax.annotate(_cand_label(c), (e[-1], v[-1]), textcoords="offset points", xytext=(4, 0), fontsize=7.5,
                    color=INK_2, va="center")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    save(fig, path)


def plot_delta_dice(analysis: List[Dict[str, Any]], path: Path) -> None:
    apply_style()
    rows = sorted(analysis, key=lambda a: a["candidate_id"])
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    xs = np.arange(len(rows))
    for x, a in zip(xs, rows):
        d = a.get("delta_dice")
        if d is None:
            ax.text(x, 0, "diverged", ha="center", va="bottom", fontsize=7.5, color=MUTED, rotation=90)
            continue
        color = SERIES[0] if a["selected"] else (SERIES[0] + "55" if a["eligible"] else NEUTRAL_FILL)
        ax.bar(x, d, width=0.62, color=color, edgecolor="white", linewidth=1.0,
               hatch=None if a["eligible"] else "///")
        ax.annotate(f"{d:+.3f}", (x, d), textcoords="offset points", xytext=(0, 3 if d >= 0 else -11),
                    ha="center", fontsize=7.5, color=INK if a["selected"] else INK_2,
                    weight="bold" if a["selected"] else "normal")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(xs, [f"C{a['candidate_id']}" + ("\n(selected)" if a["selected"] else "") for a in rows])
    ax.set_ylabel("ΔDice = Dice(epoch E) − Dice(epoch 1)   (validation)")
    ax.set_title("Learning improvement of the Pareto candidates")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="selected"), Patch(color=SERIES[0] + "55", label="eligible"),
                       Patch(facecolor=NEUTRAL_FILL, hatch="///", edgecolor="white", label="failed a gate")],
              loc="upper right")
    save(fig, path)


def plot_scatter_vs_dice(analysis: List[Dict[str, Any]], xkey: str, xlabel: str, title: str, path: Path) -> None:
    apply_style()
    rows = [a for a in analysis if a.get("final_val_dice") is not None]
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    x = np.array([a[xkey] / (1e6 if xkey == "params" else 1.0) for a in rows])
    y = np.array([a["final_val_dice"] for a in rows])
    for i, a in enumerate(rows):
        sel = a["selected"]
        ax.scatter([x[i]], [y[i]], s=70 if sel else 40, color=SERIES[0] if sel else MUTED,
                   marker=MARKERS[(a["candidate_id"] - 1) % len(MARKERS)], edgecolor="white", linewidth=1.0, zorder=3)
        ax.annotate(f"C{a['candidate_id']}" + (" (selected)" if sel else ""), (x[i], y[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=INK if sel else INK_2)
    if len(rows) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        rho, p = spearmanr(x, y)
        title = f"{title}\nSpearman ρ = {rho:.2f} (p = {p:.2g}, n = {len(rows)})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final validation Dice (epoch E)")
    ax.set_title(title)
    save(fig, path)


def plot_capability_ranking(selection: Dict[str, Any], path: Path) -> None:
    """Ranked ΔDice with gate outcomes — shows which candidate was selected and why."""
    apply_style()
    rows = sorted(selection["analysis"], key=lambda a: a["capability_rank"])
    gates = ["G1_finite", "G2_val_loss_decreased", "G3_no_overfitting", "G4_learning_trend", "G5_beats_trivial"]
    short = ["G1\nfinite", "G2\nΔLoss>0", "G3\nno overfit", "G4\nDice trend", "G5\n>trivial"]
    fig, (ax, axg) = plt.subplots(1, 2, figsize=(10.5, 0.55 * len(rows) + 2.0), sharey=True,
                                  gridspec_kw={"width_ratios": [3, 2], "wspace": 0.05})
    ys = np.arange(len(rows))[::-1]
    vals = [a["delta_dice"] for a in rows if a.get("delta_dice") is not None] or [0.0]
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    span = max(hi - lo, 1e-3)
    ax.set_xlim(lo - 0.05 * span, hi + 0.55 * span)   # room on the right for the value labels
    for y, a in zip(ys, rows):
        d = a.get("delta_dice")
        if d is not None:
            color = SERIES[0] if a["selected"] else (SERIES[0] + "55" if a["eligible"] else NEUTRAL_FILL)
            ax.barh(y, d, height=0.6, color=color, edgecolor="white", hatch=None if a["eligible"] else "///")
            ax.annotate(f"{d:+.3f} · final {a['final_val_dice']:.3f}", (max(d, 0), y), textcoords="offset points",
                        xytext=(4, 0), va="center", fontsize=7.5, color=INK if a["selected"] else INK_2)
        for j, g in enumerate(gates):
            v = a["gates"].get(g)
            sym, col = ("✓", "#006300") if v else (("✗", "#d03b3b") if v is False else ("–", MUTED))
            relaxed = g in selection.get("relaxed_gates", [])
            axg.text(j, y, sym, ha="center", va="center", fontsize=11, color=MUTED if relaxed else col)
    ax.set_yticks(ys, [f"#{a['capability_rank']}  C{a['candidate_id']}" + ("  ◀ selected" if a["selected"] else "")
                       for a in rows])
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("Validation ΔDice (epoch 1 → E)")
    axg.set_xticks(range(len(gates)), short, fontsize=7.5)
    axg.set_xlim(-0.6, len(gates) - 0.4)
    axg.grid(False)
    axg.tick_params(axis="y", length=0)
    for s in ("left", "bottom"):
        axg.spines[s].set_visible(False)
    rel = selection.get("relaxed_gates") or []
    axg.set_title("Eligibility gates" + (f" (relaxed: {', '.join(r.split('_')[0] for r in rel)})" if rel else ""),
                  fontsize=9.5)
    fig.suptitle("Learning-capability ranking of Pareto candidates", fontsize=11, weight="semibold", x=0.06,
                 ha="left", y=1.0)
    import textwrap
    fig.text(0.06, -0.02 - 0.012 * len(rows), "\n".join(textwrap.wrap(selection["reason"], 150)), fontsize=7.5,
             color=INK_2, ha="left", va="top")
    save(fig, path)


# ---------------------------------------------------------------------------------------------- alpha / final
def plot_loss_weight_search(rows: List[Dict[str, Any]], chosen_alpha: float, path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ok = [r for r in rows if r.get("criterion") is not None]
    a = [r["alpha"] for r in ok]
    c = [r["criterion"] for r in ok]
    ax.plot(a, c, color=SERIES[0], marker="o", markersize=6)
    for r in ok:
        sel = abs(r["alpha"] - chosen_alpha) < 1e-12
        ax.annotate(f"{r['criterion']:.4f}" + (" (chosen)" if sel else ""), (r["alpha"], r["criterion"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5,
                    color=INK if sel else INK_2, weight="bold" if sel else "normal")
    ax.set_xlabel("α (BCE weight);  β = 1 − α (soft-mIoU weight)")
    ax.set_ylabel("Mean validation Dice, last epochs")
    ax.set_title("Loss-weight selection on the validation set")
    save(fig, path)


def plot_final_curves(history: List[Dict[str, Any]], best_epoch: int, out_dir: Path) -> None:
    apply_style()
    e = [h["epoch"] for h in history]

    def panel(ax, key, ylabel):
        ax.plot(e, [h[f"train_{key}"] for h in history], color=SERIES[0], label="train")
        ax.plot(e, [h[f"val_{key}"] for h in history], color=SERIES[1], label="validation")
        ax.axvline(best_epoch, color=MUTED, linestyle="--", linewidth=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    panel(axs[0, 0], "loss", "Loss (α·BCE + β·soft-mIoU)")
    panel(axs[0, 1], "dice", "Dice")
    panel(axs[1, 0], "iou", "IoU (foreground)")
    axs[1, 1].plot(e, [h["learning_rate"] for h in history], color=SERIES[2])
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Learning rate (end of epoch)")
    axs[1, 1].set_yscale("log")
    fig.suptitle(f"Final training of the selected architecture (best validation Dice at epoch {best_epoch}, dashed)",
                 fontsize=11, weight="semibold")
    fig.tight_layout()
    save(fig, out_dir / "final_training_curves")

    for key, ylabel, name in (("dice", "Dice", "train_vs_val_dice"), ("loss", "Loss", "train_vs_val_loss")):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        panel(ax, key, ylabel)
        ax.set_title(f"Train vs validation {ylabel.lower()} — final training")
        save(fig, out_dir / name)
