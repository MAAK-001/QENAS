"""Representative subset of the final Pareto front for supervised evaluation.

No scalarisation, no TOPSIS, no weights. If the front has at most ``k`` points all
are evaluated. Otherwise a *diversity-preserving* subset is chosen in the min-max
normalised objective space (the same space NSGA-II uses for crowding distance):

1. both extreme points (fewest parameters; best SynFlow objective) are always kept,
   so the subset spans the whole trade-off;
2. the remaining points are added by greedy farthest-point (max-min distance)
   sampling — Gonzalez' 2-approximation of the k-centre problem — which maximises
   coverage of the front and avoids clusters of near-identical trade-offs.

The procedure is deterministic (ties broken by parameter count, then chromosome).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def select_pareto_candidates(front: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    if k <= 0:
        raise ValueError("pareto.max_candidates must be positive")
    front = sorted(front, key=lambda r: (r["f1_params"], r["f2_synflow"], r["chromosome"]))
    if len(front) <= k:
        out = [dict(r, selection_reason="entire front (|front| <= k)") for r in front]
        return out
    F = np.array([[r["f1_params"], r["f2_synflow"]] for r in front], dtype=float)
    lo, hi = F.min(0), F.max(0)
    Z = (F - lo) / np.where(hi - lo > 0, hi - lo, 1.0)
    chosen = [int(np.argmin(F[:, 0])), int(np.argmin(F[:, 1]))]
    chosen = list(dict.fromkeys(chosen))
    reasons = {chosen[0]: "extreme: fewest parameters"}
    if len(chosen) > 1:
        reasons[chosen[1]] = "extreme: best SynFlow objective"
    while len(chosen) < k:
        d = np.min(np.linalg.norm(Z[:, None, :] - Z[None, chosen, :], axis=2), axis=1)
        d[chosen] = -1.0
        nxt = int(np.argmax(d))   # first maximum -> lowest parameter count among ties (front is sorted)
        reasons[nxt] = f"farthest-point (min normalised distance {d[nxt]:.3f})"
        chosen.append(nxt)
    return [dict(front[i], selection_reason=reasons[i]) for i in sorted(chosen)]
