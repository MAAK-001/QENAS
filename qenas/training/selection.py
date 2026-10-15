r"""Learning-capability selection: successive halving and the final choice of one architecture.

Successive halving (Jamieson & Talwalkar, AISTATS 2016; the rung schedule of Hyperband / ASHA)
    All candidates are trained to the first rung (5 epochs); the best half by the criterion continues to the next
    rung (10 epochs); the best half of those is trained to the full candidate length (20 epochs). With 16
    candidates: 16 x 5 + 8 x 5 + 4 x 10 = 160 epochs, the budget of 8 candidates x 20 epochs. Candidate runs use a
    constant learning rate, the same initialisation seed and the same per-epoch data order, so a run continued
    from one rung to the next is the run that would have been trained without interruption. A run that diverges
    (non-finite loss) is eliminated at its rung (gate G1).

Criterion (every rung and the final choice)
    mean validation Dice over the last ``window`` (5) epochs reached - the end-of-run validation quality, averaged
    over epochs to reduce single-epoch noise. It replaces QENAS 2.x' ΔDice = Dice(E) - Dice(1), which rewards a
    poor start as much as a good end: in evaluation study v1 ΔDice chose the candidate with the best end-of-run
    Dice in 1 of 12 runs (DESIGN_DECISIONS D21). Ties are broken by fewer parameters.

Eligibility gates of the final choice (unchanged from QENAS 2.x; computed on the full candidate run)
    G1 finite      - training completed without NaN/Inf.
    G2 generalises - ΔLoss = Loss_val(1) - Loss_val(E) > 0.
    G3 no overfit  - NOT [validation loss significantly increasing and training loss significantly decreasing over
                     the second half] (one-sided Kendall tau tests, p < ``significance``).
    G4 genuine learning - the validation Dice has a significant increasing trend over all epochs.
    G5 competence  - final validation Dice above the Dice of the all-foreground predictor on the validation masks.

    If no finalist passes every gate, gates are relaxed in a fixed, recorded order (G4, G3, G2, G5); G1 is never
    relaxed, and only the gates that actually blocked every finalist are reported as relaxed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np

from .early_stopping import trend_test

GATE_RELAX_ORDER = ["G4_learning_trend", "G3_no_overfitting", "G2_val_loss_decreased", "G5_beats_trivial"]
ALL_GATES = ["G1_finite"] + ["G2_val_loss_decreased", "G3_no_overfitting", "G4_learning_trend", "G5_beats_trivial"]


def criterion(history: List[Dict[str, Any]], window: int) -> float:
    """Mean validation Dice over the last ``window`` epochs of ``history``."""
    return float(np.mean([h["val_dice"] for h in history[-window:]]))


def rung_sizes(n: int, n_rungs: int, keep_fraction: float) -> List[int]:
    """Number of candidates trained at each rung: n, ceil(n * keep), ceil(n * keep^2), ..."""
    sizes = [n]
    for _ in range(n_rungs - 1):
        sizes.append(max(1, math.ceil(sizes[-1] * keep_fraction)))
    return sizes


def promote(scored: Sequence[Dict[str, Any]], n_keep: int) -> List[int]:
    """Ids of the ``n_keep`` best candidates by criterion (ties: fewer parameters, then id)."""
    order = sorted(scored, key=lambda c: (-c["score"], c["params"], c["candidate_id"]))
    return [c["candidate_id"] for c in order[:n_keep]]


def analyse_candidate(history: List[Dict[str, Any]], trivial_dice: float, significance: float,
                      window: int) -> Dict[str, Any]:
    vd = [h["val_dice"] for h in history]
    vl = [h["val_loss"] for h in history]
    n = len(history)
    half = history[n // 2:]
    tau_v, p_v = trend_test([h["val_loss"] for h in half], "greater")
    tau_t, p_t = trend_test([h["train_loss"] for h in half], "less")
    tau_d, p_d = trend_test(vd, "greater")
    out = {
        "criterion": criterion(history, window), "epochs_trained": n,
        "epoch1_val_dice": vd[0], "final_val_dice": vd[-1], "delta_dice": vd[-1] - vd[0],
        "epoch1_val_loss": vl[0], "final_val_loss": vl[-1], "delta_loss": vl[0] - vl[-1],
        "final_val_iou": history[-1]["val_iou"], "best_val_dice": max(vd),
        "final_train_dice": history[-1]["train_dice"],
        "generalisation_gap_dice": history[-1]["train_dice"] - vd[-1],
        "dice_trend_tau": tau_d, "dice_trend_p": p_d,
        "val_loss_trend_tau_2nd_half": tau_v, "val_loss_trend_p_2nd_half": p_v,
        "train_loss_trend_tau_2nd_half": tau_t, "train_loss_trend_p_2nd_half": p_t,
    }
    overfit = (p_v < significance) and (p_t < significance)
    out["gates"] = {
        "G1_finite": True,
        "G2_val_loss_decreased": out["delta_loss"] > 0,
        "G3_no_overfitting": not overfit,
        "G4_learning_trend": (tau_d > 0) and (p_d < significance),
        "G5_beats_trivial": vd[-1] > trivial_dice,
    }
    return out


def select_by_learning_capability(candidates: List[Dict[str, Any]], trivial_dice: float, significance: float,
                                  window: int, full_epochs: int) -> Dict[str, Any]:
    """``candidates``: every candidate of the successive halving (id, chromosome, objectives, ``history`` up to the
    rung it reached, ``diverged``). The finalists are the candidates that completed ``full_epochs``."""
    analysed = []
    for c in candidates:
        rec = {k: c[k] for k in ("candidate_id", "chromosome", "blocks", "params", "zico", "f2_zico")}
        rec["rung_scores"] = c.get("rung_scores", {})
        if c.get("diverged"):
            rec.update({"diverged": True, "divergence": c.get("divergence", ""), "finalist": False,
                        "gates": {g: (False if g == "G1_finite" else None) for g in ALL_GATES}})
        else:
            rec.update(analyse_candidate(c["history"], trivial_dice, significance, window))
            rec["diverged"] = False
            rec["finalist"] = len(c["history"]) >= full_epochs
        analysed.append(rec)

    finalists = [a for a in analysed if a["finalist"]]
    if not finalists:
        raise RuntimeError("No candidate completed the full candidate run (all diverged); no architecture can be "
                           "selected.")

    def eligible(active: List[str]) -> List[Dict[str, Any]]:
        return [a for a in finalists if all(a["gates"][g] for g in active)]

    relaxed: List[str] = []
    active = [g for g in ALL_GATES if g != "G1_finite"]
    pool = eligible(active)
    for g in GATE_RELAX_ORDER:
        if pool:
            break
        active.remove(g)
        relaxed.append(g)
        pool = eligible(active)
    # Re-instate every relaxed gate that is not actually needed (most important first), so that only the gates
    # that truly blocked all finalists are reported as relaxed.
    for g in reversed(list(relaxed)):
        trial = active + [g]
        if eligible(trial):
            active = trial
            relaxed.remove(g)
    pool = eligible(active) or finalists

    winner = sorted(pool, key=lambda a: (-a["criterion"], a["params"], a["candidate_id"]))[0]
    pool_ids = {a["candidate_id"] for a in pool}
    for a in analysed:
        a["eligible"] = a["candidate_id"] in pool_ids
        a["selected"] = a["candidate_id"] == winner["candidate_id"]
    ranking = sorted(finalists, key=lambda a: (not a["eligible"], -a["criterion"], a["params"]))
    for r, a in enumerate(ranking, 1):
        a["capability_rank"] = r
    for a in analysed:
        a.setdefault("capability_rank", None)

    reason = (f"Candidate {winner['candidate_id']} selected: highest mean validation Dice over the last {window} of "
              f"{full_epochs} epochs among the {len(pool)} eligible of {len(finalists)} successive-halving finalists "
              f"({winner['criterion']:.4f}; ΔLoss {winner['delta_loss']:.4f}, Dice trend tau "
              f"{winner['dice_trend_tau']:.2f}, p={winner['dice_trend_p']:.2g}).")
    if relaxed:
        reason += f" No finalist passed all gates; relaxed: {', '.join(relaxed)}."
    return {"selected_id": winner["candidate_id"], "reason": reason, "relaxed_gates": relaxed,
            "active_gates": ["G1_finite"] + active, "trivial_val_dice": trivial_dice, "analysis": analysed}
