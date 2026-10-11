"""Analysis of evidence/objective_agreement/<DATASET>/results.json (written by study.py).

Usage (from QENAS/):  python evidence/objective_agreement/analyse.py BUSI

Questions answered (all on the *minimised* objectives f1 = params, f2 = 1/(1+ln(1+SynFlow)), f3 = 1/(1+ZiCo)):
1. Harmony or conflict? Kendall tau between every pair of objectives. tau > 0: the two objectives improve
   together (harmony); tau < 0: improving one worsens the other (conflict = a genuine trade-off).
2. Do SynFlow and ZiCo agree beyond model size? Kendall tau between the residuals of ln(1+S) and ZiCo after
   a rank regression on the parameter count.
3. Which block types drive each criterion? Least-squares effect of one gene of each block type (counts of
   the five block types in the chromosome; Residual = reference).
4. How large must the population be? Mean fraction of non-dominated individuals in random sub-populations of
   size N drawn from the sample, for 2 objectives (params, SynFlow) and 3 objectives (+ ZiCo).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting  # noqa: E402

ds = sys.argv[1].upper() if len(sys.argv) > 1 else "BUSI"
d = Path(__file__).resolve().parent / ds
rows = json.loads((d / "results.json").read_text())["rows"]
n = len(rows)
P = np.array([r["params"] for r in rows], float)
L = np.array([r["synflow_log"] for r in rows], float)
Z = np.array([r["zico"] for r in rows], float)
F = np.array([[r["f1_params"], r["f2_synflow"], r["f3_zico"]] for r in rows], float)
names = ["f1 params", "f2 SynFlow", "f3 ZiCo"]
out = {"dataset": ds, "n_architectures": n}

# 1. pairwise Kendall tau of the minimised objectives
tau = {}
for i in range(3):
    for j in range(i + 1, 3):
        t, p = kendalltau(F[:, i], F[:, j])
        tau[f"{names[i]} vs {names[j]}"] = {"tau": float(t), "p": float(p),
                                           "relation": "harmony" if t > 0 else "conflict (trade-off)"}
out["kendall_tau_minimised_objectives"] = tau


# 2. agreement of SynFlow and ZiCo beyond size: residuals after rank regression on params
def resid(y, x):
    ry, rx = rankdata(y), rankdata(x)
    b = np.polyfit(rx, ry, 1)
    return ry - np.polyval(b, rx)


t, p = kendalltau(resid(L, P), resid(Z, P))
out["synflow_vs_zico_given_params"] = {"tau": float(t), "p": float(p)}
for nm, v in (("ln(1+SynFlow)", L), ("ZiCo", Z)):
    t, p = kendalltau(v, P)
    out[f"{nm} vs params"] = {"tau": float(t), "p": float(p)}

# 3. block-type effects (counts of each block type; Residual is the reference category)
C = np.array([[r["chromosome"].count(g) for g in range(5)] for r in rows], float)
X = np.column_stack([np.ones(n), C[:, 1:]])
effects = {}
for nm, v in (("params_M", P / 1e6), ("ln(1+SynFlow)", L), ("ZiCo", Z)):
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    pred = X @ beta
    r2 = 1 - ((v - pred) ** 2).sum() / ((v - v.mean()) ** 2).sum()
    effects[nm] = {"intercept (all Residual)": float(beta[0]),
                   **{f"+1 {b} gene (vs Residual)": float(c) for b, c in
                      zip(["Dense", "Inception", "ConvNeXt", "DARTS"], beta[1:])}, "R2": float(r2)}
out["block_type_effects"] = effects

# 4. non-dominated fraction in random sub-populations
rng = np.random.default_rng(0)
nds = NonDominatedSorting()
frac = {}
for N in (10, 20, 30, 40):
    if N > n:
        continue
    f2o, f3o = [], []
    for _ in range(2000):
        idx = rng.choice(n, N, replace=False)
        f2o.append(len(nds.do(F[idx][:, :2], only_non_dominated_front=True)) / N)
        f3o.append(len(nds.do(F[idx], only_non_dominated_front=True)) / N)
    frac[str(N)] = {"2 objectives (params, SynFlow)": float(np.mean(f2o)), "3 objectives (+ ZiCo)": float(np.mean(f3o))}
out["nondominated_fraction_of_random_population"] = frac
out["timing_seconds_mean"] = {"synflow": float(np.mean([r["synflow_seconds"] for r in rows])),
                              "zico": float(np.mean([r["zico_seconds"] for r in rows]))}
(d / "analysis.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
