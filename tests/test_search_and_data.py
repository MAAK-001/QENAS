from pathlib import Path

import numpy as np
import pytest

from qenas.config import build_config, config_fingerprint
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


def _front(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [{"chromosome": [i % 5, i // 5] * 4, "f1_params": float(rng.integers(1, 100)), "f2_zico": float(rng.random()),
             "params": i} for i in range(n)]


def test_pareto_subset_keeps_every_objective_extreme_and_size():
    front = _front()
    sub = select_pareto_candidates(front, 6)
    assert len(sub) == 6
    for k in ("f1_params", "f2_zico"):
        assert min(r[k] for r in sub) == min(r[k] for r in front)
    assert any(r["selection_reason"].startswith("extreme") for r in sub)
    small = select_pareto_candidates(front[:4], 6)
    assert len(small) == 4 and all(r["selection_reason"] == "Pareto front" for r in small)
    assert len(select_pareto_candidates(front, 1)) == 1   # never more than k, even with more extremes than k


def test_small_front_is_filled_up_in_nsga2_survival_order():
    front = _front(3)
    pop = [dict(r, pareto_rank=0, crowding_distance="inf") for r in front]
    rest = _front(8, seed=1)[3:]
    pop += [dict(r, pareto_rank=1 + i // 3, crowding_distance=float(10 - i)) for i, r in enumerate(rest)]
    sub = select_pareto_candidates(front, 6, population=pop)
    assert len(sub) == 6
    assert [r["selection_reason"] for r in sub[:3]] == ["Pareto front"] * 3
    filled = [r["chromosome"] for r in sub[3:]]
    assert filled == [r["chromosome"] for r in rest[:3]]                # rank 2 (largest crowding first)
    assert all("pareto_rank" not in r for r in sub)


def test_pymoo_operators_produce_valid_integer_chromosomes_with_two_objectives():
    from pymoo.core.evaluator import Evaluator
    from pymoo.problems.static import StaticProblem
    cfg = build_config("BUSI")
    alg = build_algorithm(cfg)
    prob = QENASProblem()
    assert prob.n_obj == 2
    alg.setup(prob, termination=("n_gen", 4), seed=1, verbose=False)
    rng = np.random.default_rng(0)
    while alg.has_next():
        pop = alg.ask()
        X = pop.get("X")
        assert X.dtype.kind in "iu" and X.shape[1] == 8 and X.min() >= 0 and X.max() <= 4
        assert len({tuple(r) for r in X.tolist()}) == len(X)   # duplicates eliminated
        Evaluator().eval(StaticProblem(prob, F=rng.random((len(X), 2)) + 0.1), pop)
        alg.tell(infills=pop)


def test_split_does_not_depend_on_the_run_seed():
    from qenas.datasets.discovery import Sample
    from qenas.datasets.splits import make_splits
    from qenas.datasets.discovery import DiscoveryResult
    samples = [Sample(f"s{i:03d}", f"img{i}", [], group=f"s{i:03d}") for i in range(50)]
    disc = DiscoveryResult("BUSI", "", samples, False, {})
    a = make_splits(disc, build_config("BUSI", overrides={"seed": 42}))
    b = make_splits(disc, build_config("BUSI", overrides={"seed": 7}))
    assert {k: [s.sample_id for s in v] for k, v in a.items()} == {k: [s.sample_id for s in v] for k, v in b.items()}


def test_config_is_a_single_protocol():
    a = build_config("BUSI")
    b = build_config("BUSI", overrides={"device": "cpu", "data.root": "elsewhere", "data.num_workers": 4})
    assert config_fingerprint(a) == config_fingerprint(b)          # run settings are not scientific
    assert config_fingerprint(a) != config_fingerprint(build_config("BUSI", overrides={"seed": 1}))
    with pytest.raises(KeyError):
        build_config("BUSI", overrides={"evaluation.tta": "on"})   # no alternative modes exist
    with pytest.raises(ValueError):
        build_config("BUSI", overrides={"candidate_training.loss_alpha": 0.3})   # must be one of the alphas


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="datasets not available")
@pytest.mark.parametrize("ds", ["CVC", "IDRID", "KVASIR"])
def test_real_splits_have_no_leakage(ds):
    from qenas.datasets.discovery import discover
    from qenas.datasets.splits import make_splits
    cfg = build_config(ds, overrides={"data.root": str(DATA_ROOT)})
    splits = make_splits(discover(cfg), cfg)
    check_no_leakage(splits)
    if ds == "IDRID":
        assert len(splits["test"]) == 27 and len(splits["train"]) + len(splits["val"]) == 54
    if ds == "KVASIR":   # official split: val.txt is the test split, validation carved from train.txt
        assert len(splits["test"]) == 120 and len(splits["train"]) + len(splits["val"]) == 880
    if ds == "CVC":
        seqs = {k: {s.group for s in v} for k, v in splits.items()}
        assert seqs["train"].isdisjoint(seqs["test"]) and seqs["val"].isdisjoint(seqs["test"])
