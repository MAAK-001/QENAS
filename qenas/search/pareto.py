"""The candidates of the final population that are trained by successive halving.

No scalarisation, no TOPSIS, no weights. ``k`` candidates (16) are taken from the final NSGA-II population:

* If the final non-dominated front has more than ``k`` points, a *diversity-preserving* subset is chosen in the
  min-max normalised objective space (the space NSGA-II uses for crowding distance):

  1. the extreme point of every objective (fewest parameters; best ZiCo) is always kept, so the subset spans the
     whole trade-off curve;
  2. the remaining points are added by greedy farthest-point (max-min distance) sampling - Gonzalez'
     2-approximation of the k-centre problem - which maximises coverage of the front.

* If the front has at most ``k`` points, all of them are taken and the remaining places are filled with the next
  individuals of the final population in NSGA-II's own survival order (Pareto rank, then larger crowding
  distance). In evaluation study v1 the zero-cost objectives did not predict the trained quality of the front
  members (Spearman rho not significant on any dataset, DESIGN_DECISIONS D18), so the near-front elite is an
  equally good bet; successive halving then decides with real training.

The procedure is deterministic (ties broken by the sort order: the objectives, then the chromosome).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .objectives import OBJECTIVE_NAMES

_EXTREME_REASON = {"f1_params": "extreme: fewest parameters", "f2_zico": "extreme: best ZiCo objective"}


def select_pareto_candidates(front: List[Dict[str, Any]], k: int, objectives: Sequence[str] = OBJECTIVE_NAMES,
                             population: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """``front``: the final non-dominated front; ``population``: the final population with ``pareto_rank`` and
    ``crowding_distance`` (used to fill up a front smaller than ``k``)."""
    if k <= 0:
        raise ValueError("pareto.max_candidates must be positive")
    front = sorted(front, key=lambda r: tuple(r[o] for o in objectives) + (r["chromosome"],))
    if len(front) <= k:
        out = [dict(r, selection_reason="Pareto front") for r in front]
        taken = {tuple(r["chromosome"]) for r in front}

        def crowd(r):
            c = r["crowding_distance"]
            return float("inf") if c == "inf" else float(c)
        rest = sorted((r for r in population or [] if tuple(r["chromosome"]) not in taken and r["pareto_rank"] > 0),
                      key=lambda r: (r["pareto_rank"], -crowd(r), tuple(r[o] for o in objectives), r["chromosome"]))
        for r in rest[:k - len(out)]:
            rec = {key: v for key, v in r.items() if key not in ("generation", "pareto_rank", "crowding_distance",
                                                                 "nondominated")}
            out.append(dict(rec, selection_reason=f"fill-up: Pareto rank {r['pareto_rank'] + 1} of the final "
                                                  f"population (NSGA-II survival order)"))
        return out
    F = np.array([[r[o] for o in objectives] for r in front], dtype=float)
    lo, hi = F.min(0), F.max(0)
    Z = (F - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    chosen: List[int] = []
    reasons: Dict[int, str] = {}
    for j, name in enumerate(objectives):
        i = int(np.argmin(F[:, j]))
        if i in reasons:
            reasons[i] += " + " + _EXTREME_REASON[name].split(": ")[1]
        elif len(chosen) < k:
            chosen.append(i)
            reasons[i] = _EXTREME_REASON[name]
    while len(chosen) < k:
        d = np.min(np.linalg.norm(Z[:, None, :] - Z[None, chosen, :], axis=2), axis=1)
        d[chosen] = -1.0
        nxt = int(np.argmax(d))
        reasons[nxt] = f"Pareto front, farthest-point (min normalised distance {d[nxt]:.3f})"
        chosen.append(nxt)
    return [dict(front[i], selection_reason=reasons[i]) for i in sorted(chosen)]
