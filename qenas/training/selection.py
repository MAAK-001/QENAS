r"""Learning-capability selection of one architecture from the evaluated Pareto candidates.

Primary criterion (as specified):     ΔDice = Dice_val(epoch E) - Dice_val(epoch 1)
Secondary quantities:                 ΔLoss = Loss_val(epoch 1) - Loss_val(epoch E),
                                      final validation Dice, stability, overfitting.

No weighted score is used. Instead a candidate must first pass four *eligibility gates*,
each a falsifiable statement about pathological behaviour, and only then is ΔDice
compared. Thresholds are either standard statistical significance levels or derived
from the data itself:

G1 finite      — training completed without NaN/Inf (diverged runs are disqualified).
G2 generalises — ΔLoss > 0: the validation loss actually decreased. A Dice gain with a
                 rising validation loss indicates the gain does not generalise.
G3 no overfit  — NOT [validation loss has a significant increasing trend over the second
                 half of training while training loss has a significant decreasing trend]
                 (one-sided Kendall tau tests, p < ``significance``). This is the textbook
                 overfitting signature.
G4 genuine, stable learning — the validation Dice curve has a significant increasing
                 monotone trend over all epochs (Kendall tau > 0, p < ``significance``).
                 A large ΔDice produced by an erratic/oscillating curve or a lucky final
                 epoch fails this test.
G5 competence  — final validation Dice exceeds the Dice of the trivial predictor that
                 labels every pixel foreground (computed on the validation masks). An
                 architecture that cannot beat this floor has not learned to segment.

Among eligible candidates the largest ΔDice wins. Because single-epoch Dice values are
noisy, candidates whose ΔDice is within the *measured noise* of the leader are treated
as tied: noise_i = median |Dice_t - Dice_{t-1}| over candidate i's last
``noise_window`` epochs; candidate i is tied with leader l if
ΔDice_l - ΔDice_i <= max(noise_l, noise_i). Ties are broken by (1) higher final
validation Dice, (2) larger ΔLoss, (3) fewer parameters (the complexity objective).

If no candidate passes every gate, gates are relaxed in a fixed, recorded order
(G4, then G3, then G2, then G5); G1 is never relaxed.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .early_stopping import trend_test

GATE_RELAX_ORDER = ["G4_learning_trend", "G3_no_overfitting", "G2_val_loss_decreased", "G5_beats_trivial"]
ALL_GATES = ["G1_finite"] + ["G2_val_loss_decreased", "G3_no_overfitting", "G4_learning_trend", "G5_beats_trivial"]


def analyse_candidate(history: List[Dict[str, Any]], trivial_dice: float, significance: float,
                      noise_window: int) -> Dict[str, Any]:
    vd = [h["val_dice"] for h in history]
    vl = [h["val_loss"] for h in history]
    tl = [h["train_loss"] for h in history]
    n = len(history)
    half = history[n // 2:]
    tau_v, p_v = trend_test([h["val_loss"] for h in half], "greater")
    tau_t, p_t = trend_test([h["train_loss"] for h in half], "less")
    tau_d, p_d = trend_test(vd, "greater")
    diffs = np.abs(np.diff(vd[-(noise_window + 1):])) if n > 1 else np.array([0.0])
    noise = float(np.median(diffs)) if diffs.size else 0.0
    out = {
        "epoch1_val_dice": vd[0], "final_val_dice": vd[-1], "delta_dice": vd[-1] - vd[0],
        "epoch1_val_loss": vl[0], "final_val_loss": vl[-1], "delta_loss": vl[0] - vl[-1],
        "epoch1_val_iou": history[0]["val_iou"], "final_val_iou": history[-1]["val_iou"],
        "best_val_dice": max(vd), "min_val_loss": min(vl),
        "final_train_dice": history[-1]["train_dice"], "final_train_loss": tl[-1],
        "generalisation_gap_dice": history[-1]["train_dice"] - vd[-1],
        "dice_trend_tau": tau_d, "dice_trend_p": p_d,
        "val_loss_trend_tau_2nd_half": tau_v, "val_loss_trend_p_2nd_half": p_v,
        "train_loss_trend_tau_2nd_half": tau_t, "train_loss_trend_p_2nd_half": p_t,
        "dice_noise": noise,
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


def select_by_learning_capability(candidates: List[Dict[str, Any]], trivial_dice: float,
                                  significance: float, noise_window: int) -> Dict[str, Any]:
    """``candidates``: dicts with keys id, params, f2_synflow, synflow_log, history (list) or diverged=True."""
    analysed = []
    for c in candidates:
        rec = {k: c[k] for k in ("candidate_id", "chromosome", "blocks", "params", "synflow_log", "f2_synflow")}
        if c.get("diverged"):
            rec.update({"diverged": True, "divergence": c.get("divergence", ""),
                        "gates": {g: (False if g == "G1_finite" else None) for g in ALL_GATES}})
        else:
            rec.update(analyse_candidate(c["history"], trivial_dice, significance, noise_window))
            rec["diverged"] = False
        analysed.append(rec)

    relaxed: List[str] = []
    finite = [a for a in analysed if not a["diverged"]]
    if not finite:
        raise RuntimeError("All Pareto candidates diverged (non-finite loss); no architecture can be selected.")

    def eligible(active: List[str]) -> List[Dict[str, Any]]:
        return [a for a in finite if all(a["gates"][g] for g in active)]

    active = [g for g in ALL_GATES if g != "G1_finite"]
    pool = eligible(active)
    for g in GATE_RELAX_ORDER:
        if pool:
            break
        active.remove(g)
        relaxed.append(g)
        pool = eligible(active)
    # Re-instate every relaxed gate that is not actually needed (most important first), so that
    # only the gates that truly blocked all candidates are reported as relaxed.
    for g in reversed(list(relaxed)):
        trial = active + [g]
        if eligible(trial):
            active = trial
            relaxed.remove(g)
    pool = eligible(active) if active else finite
    if not pool:
        pool = finite

    leader = max(pool, key=lambda a: a["delta_dice"])
    tied = [a for a in pool
            if leader["delta_dice"] - a["delta_dice"] <= max(leader["dice_noise"], a["dice_noise"])]
    winner = sorted(tied, key=lambda a: (-a["final_val_dice"], -a["delta_loss"], a["params"]))[0]

    pool_ids = {a["candidate_id"] for a in pool}
    tied_ids = {a["candidate_id"] for a in tied}
    for a in analysed:
        a["eligible"] = a["candidate_id"] in pool_ids
        a["tied_with_leader"] = a["candidate_id"] in tied_ids
        a["selected"] = a["candidate_id"] == winner["candidate_id"]
    ranking = sorted([a for a in analysed if a["eligible"]], key=lambda a: -a["delta_dice"]) + \
        sorted([a for a in analysed if not a["eligible"]], key=lambda a: -(a.get("delta_dice") or -1e9))
    for r, a in enumerate(ranking, 1):
        a["capability_rank"] = r

    if len(tied) > 1:
        why = (f"largest ΔDice tier (within measured Dice noise of the leader, {len(tied)} tied); "
               f"tie broken by final validation Dice")
    else:
        why = "largest validation ΔDice among eligible candidates"
    reason = (f"Candidate {winner['candidate_id']} selected: {why}. "
              f"ΔDice={winner['delta_dice']:.4f}, final val Dice={winner['final_val_dice']:.4f}, "
              f"ΔLoss={winner['delta_loss']:.4f}, Dice trend tau={winner['dice_trend_tau']:.2f} "
              f"(p={winner['dice_trend_p']:.2g}).")
    if relaxed:
        reason += f" No candidate passed all gates; relaxed: {', '.join(relaxed)}."
    return {"selected_id": winner["candidate_id"], "reason": reason, "relaxed_gates": relaxed,
            "active_gates": ["G1_finite"] + active, "trivial_val_dice": trivial_dice, "analysis": analysed}
