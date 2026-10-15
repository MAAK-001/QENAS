"""Publication figures for every QENAS stage (PNG at 300 dpi)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

from .style import INK, INK_2, MARKERS, MUTED, NEUTRAL_FILL, SERIES, apply_style, plt, save, series_color

PARAMS_LABEL = "Trainable parameters (M)   (lower is better)"
ZICO_LABEL = "ZiCo   (higher is better)"


def _v(r: Dict[str, Any], key: str) -> float:
    return r[key] / 1e6 if key == "params" else r[key]


def _xy(rows: List[Dict[str, Any]], kx: str = "params", ky: str = "zico") -> np.ndarray:
    return np.array([[_v(r, kx), _v(r, ky)] for r in rows], dtype=float).reshape(-1, 2)


# ---------------------------------------------------------------------------------------------- search
def plot_pareto_population(pop_rows: List[Dict[str, Any]], archive_rows: List[Dict[str, Any]], generation: int,
                           n_generations: int, path: Path, highlight: Optional[List[Dict[str, Any]]] = None,
                           title: Optional[str] = None) -> None:
    """Population of one generation in the objective space (parameters vs ZiCo)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    nd = np.array([r["nondominated"] for r in pop_rows], dtype=bool)
    if archive_rows:
        a = _xy(archive_rows)
        ax.scatter(a[:, 0], a[:, 1], s=12, color=NEUTRAL_FILL, edgecolor="none", label="evaluated so far", zorder=1)
    p = _xy(pop_rows)
    if (~nd).any():
        ax.scatter(p[~nd, 0], p[~nd, 1], s=26, color=MUTED, edgecolor="white", linewidth=0.8,
                   label="population (dominated)", zorder=2)
    if nd.any():
        ax.scatter(p[nd, 0], p[nd, 1], s=40, color=SERIES[0], edgecolor="white", linewidth=1.0,
                   label=f"non-dominated: {int(nd.sum())}", zorder=4)
    if highlight:
        h = _xy(highlight)
        ax.scatter(h[:, 0], h[:, 1], s=110, facecolor="none", edgecolor=SERIES[1], linewidth=1.6,
                   label="candidates (successive halving)", zorder=5)
        for r, (x, y) in zip(highlight, h):
            ax.annotate(f"C{r['candidate_id']}", (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8,
                        color=INK_2)
    ax.set_xlabel(PARAMS_LABEL)
    ax.set_ylabel(ZICO_LABEL)
    ax.legend(loc="lower right")
    gen_txt = "initial population (generation 0)" if generation == 0 else f"generation {generation}/{n_generations}"
    ax.set_title(title or f"NSGA-II population — {gen_txt}")
    save(fig, path)


def plot_search_progress(history: List[Dict[str, Any]], gen_table: List[Dict[str, Any]], path: Path) -> None:
    """Hypervolume of each generation's non-dominated set and the population's objective distribution."""
    apply_style()
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.0))
    gens = [g["generation"] for g in gen_table]
    axs[0].plot(gens, [g["hypervolume"] for g in gen_table], color=SERIES[0], marker="o", markersize=4)
    axs[0].set_ylabel("Hypervolume (normalised objectives, ref. 1.1)")
    axs[0].set_title("Non-dominated set quality")
    for ax, key, label in ((axs[1], "params", "Parameters (M)"), (axs[2], "zico", "ZiCo")):
        med, q1, q3 = [], [], []
        for g in gens:
            v = np.array([_v(r, key) for r in history if r["generation"] == g])
            med.append(np.median(v))
            q1.append(np.percentile(v, 25))
            q3.append(np.percentile(v, 75))
        ax.fill_between(gens, q1, q3, color=SERIES[0], alpha=0.18, linewidth=0, label="population IQR")
        ax.plot(gens, med, color=SERIES[0], marker="o", markersize=3.5, label="population median")
        ax.set_ylabel(label)
        ax.set_title(f"{label} of the surviving population")
        ax.legend(loc="best")
    from matplotlib.ticker import MaxNLocator
    for ax in axs.flat:
        ax.set_xlabel("Generation")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Training-free NSGA-II search progress", fontsize=11, weight="semibold")
    fig.tight_layout()
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


def plot_successive_halving(sh: Dict[str, Any], analysis: List[Dict[str, Any]], path: Path) -> None:
    """Criterion of every candidate at every rung it reached; the selected one is highlighted."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    rungs = [r["epochs"] for r in sh["rungs"]]
    sel = next((a["candidate_id"] for a in analysis if a.get("selected")), None)
    for i, a in enumerate(sorted(analysis, key=lambda a: a["candidate_id"])):
        pts = [(int(e), v) for e, v in sorted(a.get("rung_scores", {}).items(), key=lambda kv: int(kv[0]))]
        if not pts:
            continue
        x, y = zip(*pts)
        is_sel = a["candidate_id"] == sel
        full = len(pts) == len(rungs)
        ax.plot(x, y, color=SERIES[0] if is_sel else (series_color(i) if full else MUTED),
                linewidth=2.6 if is_sel else 1.1, marker="o", markersize=4 if is_sel else 3,
                alpha=1.0 if is_sel or full else 0.55, zorder=5 if is_sel else 2)
        ax.annotate(f"C{a['candidate_id']}" + (" (selected)" if is_sel else ""), (x[-1], y[-1]),
                    textcoords="offset points", xytext=(5, 0), fontsize=7.5, color=INK if is_sel else INK_2,
                    va="center")
    ax.set_xticks(rungs, [f"{e} epochs\n({n} trained)" for e, n in zip(rungs, sh["sizes"])])
    ax.set_xlabel("Rung")
    ax.set_ylabel(f"Mean validation Dice, last {sh['window']} epochs")
    ax.set_title("Successive halving of the candidates")
    save(fig, path)


def plot_scatter_vs_dice(analysis: List[Dict[str, Any]], xkey: str, xlabel: str, title: str, path: Path,
                         rung: str) -> None:
    """Objective ``xkey`` against the criterion at rung ``rung`` (epochs, reached by every candidate)."""
    apply_style()
    rows = [a for a in analysis if rung in a.get("rung_scores", {})]
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    x = np.array([a[xkey] / (1e6 if xkey == "params" else 1.0) for a in rows])
    y = np.array([a["rung_scores"][rung] for a in rows])
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
    ax.set_ylabel(f"Mean validation Dice after {rung} epochs")
    ax.set_title(title)
    save(fig, path)


def plot_capability_ranking(selection: Dict[str, Any], window: int, path: Path) -> None:
    """Finalists ranked by the criterion, with the gate outcomes - shows which candidate was selected and why."""
    apply_style()
    rows = sorted([a for a in selection["analysis"] if a.get("finalist")], key=lambda a: a["capability_rank"])
    gates = ["G1_finite", "G2_val_loss_decreased", "G3_no_overfitting", "G4_learning_trend", "G5_beats_trivial"]
    short = ["G1\nfinite", "G2\nΔLoss>0", "G3\nno overfit", "G4\nDice trend", "G5\n>trivial"]
    fig, (ax, axg) = plt.subplots(1, 2, figsize=(10.5, 0.55 * len(rows) + 2.0), sharey=True,
                                  gridspec_kw={"width_ratios": [3, 2], "wspace": 0.05})
    ys = np.arange(len(rows))[::-1]
    vals = [a["criterion"] for a in rows]
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-3)
    base = lo - 0.5 * span
    ax.set_xlim(base, hi + 0.6 * span)
    for y, a in zip(ys, rows):
        color = SERIES[0] if a["selected"] else (SERIES[0] + "55" if a["eligible"] else NEUTRAL_FILL)
        ax.barh(y, a["criterion"] - base, left=base, height=0.6, color=color, edgecolor="white",
                hatch=None if a["eligible"] else "///")
        ax.annotate(f"{a['criterion']:.4f}", (a["criterion"], y), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=7.5, color=INK if a["selected"] else INK_2)
        for j, g in enumerate(gates):
            v = a["gates"].get(g)
            sym, col = ("✓", "#006300") if v else (("✗", "#d03b3b") if v is False else ("–", MUTED))
            relaxed = g in selection.get("relaxed_gates", [])
            axg.text(j, y, sym, ha="center", va="center", fontsize=11, color=MUTED if relaxed else col)
    ax.set_yticks(ys, [f"#{a['capability_rank']}  C{a['candidate_id']}" + ("  ◀ selected" if a["selected"] else "")
                       for a in rows])
    ax.set_xlabel(f"Mean validation Dice over the last {window} epochs")
    axg.set_xticks(range(len(gates)), short, fontsize=7.5)
    axg.set_xlim(-0.6, len(gates) - 0.4)
    axg.grid(False)
    axg.tick_params(axis="y", length=0)
    for side in ("left", "bottom"):
        axg.spines[side].set_visible(False)
    rel = selection.get("relaxed_gates") or []
    axg.set_title("Eligibility gates" + (f" (relaxed: {', '.join(r.split('_')[0] for r in rel)})" if rel else ""),
                  fontsize=9.5)
    fig.suptitle("Learning-capability ranking of the successive-halving finalists", fontsize=11, weight="semibold",
                 x=0.06, ha="left", y=1.0)
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

    def panel(ax, key, ylabel, train=True):
        if train:
            ax.plot(e, [h[f"train_{key}"] for h in history], color=SERIES[0], label="train")
        ax.plot(e, [h[f"val_{key}"] for h in history], color=SERIES[1], label="validation")
        ax.axvline(best_epoch, color=MUTED, linestyle="--", linewidth=0.9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")

    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    panel(axs[0, 0], "loss", "Loss (α·BCE + β·soft-mIoU)")
    panel(axs[0, 1], "dice", "Dice")
    panel(axs[0, 2], "miou", "mIoU (class mean)")
    panel(axs[1, 0], "sensitivity", "Sensitivity (recall)")
    panel(axs[1, 1], "hd95", "HD95 (pixels, lower is better)", train=False)
    axs[1, 2].plot(e, [h["learning_rate"] for h in history], color=SERIES[2])
    axs[1, 2].set_xlabel("Epoch")
    axs[1, 2].set_ylabel("Learning rate (end of epoch)")
    from matplotlib.ticker import MaxNLocator
    for ax in axs.flat:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"Final training of the selected architecture (best validation Dice at epoch {best_epoch}, dashed)",
                 fontsize=11, weight="semibold")
    fig.tight_layout()
    save(fig, out_dir / "final_training_curves")

    for key, ylabel, name in (("dice", "Dice", "train_vs_val_dice"), ("loss", "Loss", "train_vs_val_loss")):
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        panel(ax, key, ylabel)
        ax.set_title(f"Train vs validation {ylabel.lower()} — final training")
        save(fig, out_dir / name)
