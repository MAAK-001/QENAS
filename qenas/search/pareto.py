"""Representative subset of the final Pareto front for supervised evaluation.

No scalarisation, no TOPSIS, no weights. If the front has at most ``k`` points all
are evaluated. Otherwise a *diversity-preserving* subset is chosen in the min-max
normalised objective space (the space NSGA-II uses for crowding distance):

1. the extreme point of every objective (fewest parameters; best SynFlow; best ZiCo)
   is always kept, so the subset spans the whole trade-off surface;
2. the remaining points are added by greedy farthest-point (max-min distance)
   sampling - Gonzalez' 2-approximation of the k-centre problem - which maximises
   coverage of the front and avoids clusters of near-identical trade-offs.

The procedure is deterministic (ties broken by the sort order: parameters, then the
other objectives, then the chromosome).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .objectives import OBJECTIVE_NAMES

_EXTREME_REASON = {"f1_params": "extreme: fewest parameters", "f2_synflow": "extreme: best SynFlow objective",
                   "f3_zico": "extreme: best ZiCo objective"}


def select_pareto_candidates(front: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if k <= 0:
        raise ValueError("pareto.max_candidates must be positive")
    front = sorted(front, key=lambda r: tuple(r[o] for o in OBJECTIVE_NAMES) + (r["chromosome"],))
    if len(front) <= k:
        return [dict(r, selection_reason="entire front (|front| <= k)") for r in front]
    F = np.array([[r[o] for o in OBJECTIVE_NAMES] for r in front], dtype=float)
    lo, hi = F.min(0), F.max(0)
    Z = (F - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    chosen: List[int] = []
    reasons: Dict[int, str] = {}
    for j, name in enumerate(OBJECTIVE_NAMES):
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
        reasons[nxt] = f"farthest-point (min normalised distance {d[nxt]:.3f})"
        chosen.append(nxt)
    return [dict(front[i], selection_reason=reasons[i]) for i in sorted(chosen)]
