from pathlib import Path

import numpy as np
import pytest

from qenas.config import build_config
from qenas.datasets.discovery import Sample
from qenas.datasets.splits import _group_select, check_no_leakage, DatasetError
from qenas.search.pareto import select_pareto_candidates
from qenas.search.problem import QENASProblem, build_algorithm

DATA_ROOT = Path(__file__).resolve().parents[2] / "Datasets"


def test_group_split_keeps_groups_together_and_is_deterministic():
    samples = [Sample(f"s{i}", f"img{i}", [], group=f"g{i // 5}") for i in range(100)]
    a, rest_a = _group_select(samples, 0.2, np.random.default_rng(1))
    b, _ = _group_select(samples, 0.2, np.random.default_rng(1))
    assert [s.sample_id for s in a] == [s.sample_id for s in b]
    assert {s.group for s in a}.isdisjoint({s.group for s in rest_a})
    assert abs(len(a) - 20) <= 5


def test_leakage_detector():
    s = Sample("a", "img_a", [], group="g")
    with pytest.raises(DatasetError):
        check_no_leakage({"train": [s], "test": [Sample("b", "img_b", [], group="g")]})


def test_pareto_subset_keeps_extremes_and_size():
    front = [{"chromosome": [i] * 8, "f1_params": float(i), "f2_synflow": 1.0 / (1 + i), "params": i}
             for i in range(1, 21)]
    sub = select_pareto_candidates(front, 6)
    assert len(sub) == 6
    p = [r["f1_params"] for r in sub]
    assert min(p) == 1 and max(p) == 20
    assert select_pareto_candidates(front[:4], 6) == [dict(r, selection_reason="entire front (|front| <= k)")
                                                      for r in front[:4]]


def test_pymoo_operators_produce_valid_integer_chromosomes():
    from pymoo.core.evaluator import Evaluator
    from pymoo.problems.static import StaticProblem
    cfg = build_config("BUSI")
    alg = build_algorithm(cfg)
    prob = QENASProblem()
    alg.setup(prob, termination=("n_gen", 4), seed=1, verbose=False)
    rng = np.random.default_rng(0)
    while alg.has_next():
        pop = alg.ask()
        X = pop.get("X")
        assert X.dtype.kind in "iu" and X.shape[1] == 8 and X.min() >= 0 and X.max() <= 4
        assert len({tuple(r) for r in X.tolist()}) == len(X)   # duplicates eliminated
        Evaluator().eval(StaticProblem(prob, F=rng.random((len(X), 2)) + 0.1), pop)
        alg.tell(infills=pop)


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="datasets not available")
@pytest.mark.parametrize("ds", ["CVC", "IDRID"])
def test_real_splits_have_no_leakage(ds):
    from qenas.datasets.discovery import discover
    from qenas.datasets.splits import make_splits
    cfg = build_config(ds, overrides={"data.root": str(DATA_ROOT)})
    splits = make_splits(discover(cfg), cfg)
    check_no_leakage(splits)
    if ds == "IDRID":
        assert len(splits["test"]) == 27 and len(splits["train"]) + len(splits["val"]) == 54
    if ds == "CVC":
        seqs = {k: {s.group for s in v} for k, v in splits.items()}
        assert seqs["train"].isdisjoint(seqs["test"]) and seqs["val"].isdisjoint(seqs["test"])
