"""pymoo problem definition and genetic operators for the categorical QENAS chromosome.

* Algorithm: the official pymoo ``NSGA2`` (non-dominated sorting + crowding-distance
  survival, binary tournament mating selection) — not re-implemented here.
* Sampling: ``IntegerRandomSampling`` (uniform over {0..4} per gene).
* Crossover: ``UniformCrossover`` — every gene is inherited from either parent with
  probability 0.5. This equals the binomial crossover with CR = 0.5 used by Mixed-GGNAS
  and is appropriate for nominal genes (block types have no order, so ordinal operators
  such as SBX/polynomial mutation are not used).
* Mutation: pymoo ``ChoiceRandomMutation`` (random resetting of categorical genes,
  per-gene probability ``prob_var``), wrapped only to return integer arrays.
* ``eliminate_duplicates=True``: offspring identical to existing individuals are regenerated.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling

from ..models.chromosome import CHROMOSOME_LENGTH, N_BLOCK_TYPES


class QENASProblem(Problem):
    """Two objectives, both minimised: f1 = #parameters, f2 = 1 / (1 + ln(1 + SynFlow))."""

    def __init__(self) -> None:
        super().__init__(
            n_var=CHROMOSOME_LENGTH, n_obj=2, n_ieq_constr=0,
            xl=np.zeros(CHROMOSOME_LENGTH, dtype=int), xu=np.full(CHROMOSOME_LENGTH, N_BLOCK_TYPES - 1, dtype=int),
            vtype=int,
            vars={f"g{i + 1}": Choice(options=list(range(N_BLOCK_TYPES))) for i in range(CHROMOSOME_LENGTH)},
        )

    def _evaluate(self, X, out, *args, **kwargs):  # pragma: no cover - guarded
        raise RuntimeError("QENAS evaluates individuals explicitly (ask/tell) with caching and checkpointing.")


class IntegerChoiceRandomMutation(ChoiceRandomMutation):
    """pymoo's categorical random-resetting mutation, returning an integer array."""

    def _do(self, problem, X, random_state=None, **kwargs):
        return np.asarray(super()._do(problem, X, random_state=random_state, **kwargs)).astype(int)


def build_algorithm(cfg: Dict[str, Any]) -> NSGA2:
    s = cfg["search"]
    return NSGA2(
        pop_size=int(s["population_size"]),
        sampling=IntegerRandomSampling(),
        crossover=UniformCrossover(prob=float(s["crossover_prob"])),
        mutation=IntegerChoiceRandomMutation(prob=float(s["mutation_prob"]), prob_var=float(s["mutation_prob_var"])),
        eliminate_duplicates=bool(s["eliminate_duplicates"]),
    )
