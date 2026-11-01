"""Fairness tests of the journal revision (REVISION.md): the new conditions differ from the existing ones only in the
documented way, and they resume. Synthetic data and the shrunk protocol of test_study.py; CPU (bit-exact)."""

import copy
import importlib.util
import json

import numpy as np
import pytest
import torch

from qenas.config import build_config, config_fingerprint, diff_configs
from qenas.experiment import experiment_dir
from qenas.methods import METHODS, stages_for
from qenas.pipeline import study_config
from qenas.runner import run_experiment
from qenas.search.nsga2_search import NSGA2Search
from qenas.search.problem import build_algorithm
from qenas.study.plan import revision_plan
from test_study import SMALL, _make_cvc, _make_kvasir


def _base(tmp):
    return {**SMALL, "data.root": str(tmp / "Datasets"), "paths.results": str(tmp / "results"),
            "paths.cache": str(tmp / "cache")}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("revision")
    _make_kvasir(tmp / "Datasets")
    _make_cvc(tmp / "Datasets")
    return tmp


# ------------------------------------------------------------------------------- (a) Supervised NSGA-II: only f2
def test_supervised_nsga2_differs_from_qenas_only_in_f2():
    q, s = METHODS["qenas"], METHODS["supervised_nsga2"]
    assert {k for k in set(q) | set(s) if q.get(k) != s.get(k)} == {"objectives", "fitness", "label", "doc"}
    assert q["objectives"][0] == s["objectives"][0] == "f1_params" and s["objectives"][1] == "f2_supervised"
    assert stages_for("supervised_nsga2") == stages_for("qenas")
    cq, cs = build_config("BUSI", {"method": "qenas", "seed": 45}), build_config("BUSI", {"method": "supervised_nsga2",
                                                                                         "seed": 45})
    assert diff_configs(cq, cs) == {"method": ("qenas", "supervised_nsga2")}      # same protocol, seed, split, search
    aq, as_ = build_algorithm(cq), build_algorithm(cs)                              # same pymoo NSGA-II and operators
    ops = [(a.pop_size, type(a.mating.crossover), a.mating.crossover.prob.value, type(a.mating.mutation),
            a.mating.mutation.prob_var.value, a.eliminate_duplicates is not None) for a in (aq, as_)]
    assert ops[0] == ops[1]


class _TableFitness:
    """Stand-in for SupervisedFitness that returns QENAS' own f2 (ZiCo objective) for every chromosome."""

    def __init__(self, table):
        self.table, self.trained, self.epochs = table, 0, 5

    def evaluate(self, ch):
        self.trained += 1
        f2 = self.table[tuple(int(g) for g in ch)]              # KeyError = the search asked another architecture
        return {"s1_dice": 1 - f2, "f2_supervised": f2, "fitness_epochs": 0, "fitness_seconds": 0.0,
                "fitness_diverged": False, "fitness_divergence": ""}


def test_supervised_search_is_the_qenas_search_when_f2_is_the_same(tmp_path):
    """With f2 = QENAS' ZiCo objective, Supervised NSGA-II must reproduce QENAS' search exactly: same sampling,
    operators, survival, archive, populations and hypervolume -> the only difference is the f2 values."""
    from qenas.search.supervised import SupervisedNSGA2Search
    ov = {**SMALL, "paths.results": str(tmp_path / "r"), "paths.cache": str(tmp_path / "c"),
          "search.population_size": 6, "search.generations": 3}
    g = torch.Generator().manual_seed(0)
    zb = [(torch.randn(4, 3, 64, 64, generator=g), (torch.rand(4, 1, 64, 64, generator=g) > 0.7).float())
          for _ in range(2)]
    cq = study_config("KVASIR", "qenas", 42, ov)
    q = NSGA2Search(cq, tmp_path / "q", torch.device("cpu"), config_fingerprint(cq), zb, METHODS["qenas"]["objectives"])
    rq = q.run()
    table = {tuple(r["chromosome"]): r["f2_zico"] for r in rq["archive"]}
    cs = study_config("KVASIR", "supervised_nsga2", 42, ov)
    s = SupervisedNSGA2Search(cs, tmp_path / "s", torch.device("cpu"), config_fingerprint(cs), zb,
                              METHODS["supervised_nsga2"]["objectives"], _TableFitness(table))
    rs = s.run()
    strip = ("first_generation", "blocks", "eval_seconds")
    assert [r["chromosome"] for r in rq["population_history"]] == [r["chromosome"] for r in rs["population_history"]]
    assert [(r["pareto_rank"], r["crowding_distance"]) for r in rq["population_history"]] == \
        [(r["pareto_rank"], r["crowding_distance"]) for r in rs["population_history"]]
    assert sorted(map(tuple, (r["chromosome"] for r in rq["archive"]))) == \
        sorted(map(tuple, (r["chromosome"] for r in rs["archive"])))
    assert [g["hypervolume"] for g in rq["generation_table"]] == [g["hypervolume"] for g in rs["generation_table"]]
    assert all({k: v for k, v in a.items() if k not in strip} == {k: v for k, v in b.items() if k in a and k not in strip}
               for a, b in zip(rq["pareto_front"], rs["pareto_front"]))
    assert len(rs["archive"]) <= cs["search"]["population_size"] * (cs["search"]["generations"] + 1)


@pytest.fixture(scope="module")
def supervised(data):
    cfg = study_config("KVASIR", "supervised_nsga2", 42, _base(data))
    assert run_experiment(cfg) == 0
    return cfg


def test_supervised_fitness_is_exactly_the_rung1_score(supervised):
    """S1 of the search = the rung-1 successive-halving score of the same architecture (retrained there), bit for bit
    on the CPU: same initialisation, protocol, data order and augmentation."""
    d = experiment_dir(supervised)
    s1 = {tuple(r["chromosome"]): r["s1_dice"] for r in json.loads((d / "search_results.json").read_text())}
    rung1 = str(supervised["candidate_training"]["rungs"][0])
    cands = json.loads((d / "candidate_results.json").read_text())
    assert cands and all(c["rung_scores"][rung1] == s1[tuple(c["chromosome"])] for c in cands)
    for r in json.loads((d / "search_results.json").read_text()):
        assert r["f2_supervised"] == pytest.approx(1 - r["s1_dice"]) and r["fitness_epochs"] == int(rung1)


def test_supervised_nsga2_outputs_match_qenas_format(supervised):
    d = experiment_dir(supervised)
    st = json.loads((d / "state.json").read_text())
    assert list(st["stages"]) == stages_for("qenas") and st["status"] == "completed"
    for f in ("search_results.csv", "search_population_history.json", "nsga2_generations.json", "pareto_front.json",
              "pareto_candidates.json", "successive_halving.json", "selection.json", "loss_weights.json",
              "final_metrics.json", "test_per_image.csv", "test_predictions.npz", "model_summary.json"):
        assert (d / f).exists(), f
    summ = json.loads((d / "search_summary.json").read_text())
    assert summ["fitness"] == "supervised" and summ["objectives"] == ["f1_params", "f2_supervised"]
    assert summ["fitness_epochs_total"] == summ["unique_evaluations"] * supervised["candidate_training"]["rungs"][0]
    gens = json.loads((d / "nsga2_generations.json").read_text())
    assert all("hypervolume" in g and "best_s1_dice" in g for g in gens)
    assert len(list((d / "supervised_fitness").glob("*.json"))) == summ["unique_evaluations"]
    assert not (d / "checkpoints" / "supervised_search").exists()          # pruned after the search
    final = json.loads((d / "final_metrics.json").read_text())
    assert final["architecture"]["type"] == "qenas" and final["params_m"] > 0 and final["gflops_mixed_ggnas"] > 0


def test_supervised_search_resumes_mid_generation(data, monkeypatch):
    """Interrupt the search after a few fitness trainings; the re-run reuses the stored records and ends with exactly
    the archive of an uninterrupted search."""
    from qenas.search import supervised as sup
    base = _base(data)
    base["paths.results"] = str(data / "results_resume")
    cfg = study_config("KVASIR", "supervised_nsga2", 43, base)
    calls = {"n": 0}
    orig = sup.Trainer.fit

    def flaky(self, until=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return orig(self, until=until)

    monkeypatch.setattr(sup.Trainer, "fit", flaky)
    assert run_experiment(cfg, stop_after="search") == 130
    monkeypatch.setattr(sup.Trainer, "fit", orig)
    d = experiment_dir(cfg)
    stored = len(list((d / "supervised_fitness").glob("*.json")))
    assert stored == 2
    assert run_experiment(cfg, stop_after="search") == 0
    resumed = json.loads((d / "search_results.json").read_text())
    ref_base = dict(base, **{"paths.results": str(data / "results_ref")})
    ref = study_config("KVASIR", "supervised_nsga2", 43, ref_base)
    assert run_experiment(ref, stop_after="search") == 0
    full = json.loads((experiment_dir(ref) / "search_results.json").read_text())
    key = lambda r: (tuple(r["chromosome"]), r["s1_dice"], r["params"])                      # noqa: E731
    assert sorted(map(key, resumed)) == sorted(map(key, full))


# ------------------------------------------------------------------------------- (b) UNeXt: the U-Net code path
needs_timm = pytest.mark.skipif(importlib.util.find_spec("timm") is None, reason="UNeXt needs timm")


@needs_timm
def test_unext_is_the_official_network():
    from qenas.models.baselines import build_baseline
    from qenas.models.network import count_parameters
    from qenas.models.summary import cost_summary
    m = build_baseline("unext", 3, 1)
    assert count_parameters(m) == 1_471_921 and round(count_parameters(m) / 1e6, 2) == 1.47      # paper: 1.47 M
    assert round(cost_summary(m, [256, 256], 3, 8)["gflops_per_image"], 2) == 0.57              # paper: 0.57 G
    for h, w in ((256, 256), (320, 512)):
        assert tuple(m(torch.zeros(1, 3, h, w)).shape) == (1, 1, h, w)
    with pytest.raises(ValueError):
        build_config("BUSI", {"method": "unext", "data.image_size": [240, 240]})       # 240 = 16 x 15: not / 32
    assert build_config("IDRID", {"method": "unext"})["data"]["image_size"] == [320, 512]


@needs_timm
def test_unext_trains_through_the_unet_code_path(data, monkeypatch):
    """UNeXt and U-Net: same stages, same Trainer calls (protocol, loss weight, seed, data, early stopping, checkpoints)
    - only the network differs; same output files; external validation works."""
    from qenas import pipeline as pl
    calls = {}
    real = pl.Trainer

    class Spy(real):
        def __init__(self, **kw):
            calls.setdefault(kw["tag"].split("_")[0] + "|" + type(kw["model"]).__name__, []).append(kw)
            super().__init__(**kw)

    monkeypatch.setattr(pl, "Trainer", Spy)
    out = {}
    for m in ("unet", "unext"):
        cfg = study_config("KVASIR", m, 42, _base(data))
        assert run_experiment(cfg) == 0
        out[m] = cfg
    sig = {}
    for m, net in (("unet", "UNet"), ("unext", "UNext")):
        sig[m] = [(k.split("|")[0], kw["tcfg"], kw["alpha"], kw["seed"], kw.get("early_stopping"), kw["ckpt_dir"].name,
                   kw.get("progress_prefix"), kw["history_csv"].name if kw.get("history_csv") else None)
                  for k, v in sorted(calls.items()) if k.endswith("|" + net) for kw in v]
        assert sig[m]
    assert sig["unet"] == sig["unext"]
    du, dx = experiment_dir(out["unet"]), experiment_dir(out["unext"])
    su, sx = (json.loads((d / "state.json").read_text()) for d in (du, dx))
    assert list(su["stages"]) == list(sx["stages"]) == stages_for("unet")
    assert sorted(p.name for p in du.iterdir() if p.is_file()) == sorted(p.name for p in dx.iterdir() if p.is_file())
    fx = json.loads((dx / "final_metrics.json").read_text())
    assert fx["method_label"] == "UNeXt" and fx["architecture"] == {"type": "baseline", "network": "unext"}
    assert fx["params_m"] == pytest.approx(1.471921)
    from qenas.study.external import external_validation
    ext = external_validation(out["unext"], torch.device("cpu"))
    assert ext["n_images"] == 20 and 0 <= ext["metrics"]["dice"] <= 1


# ------------------------------------------------------------------------------- plan, scheduler, splits
def test_revision_plan_and_session_spanning_jobs():
    jobs = revision_plan()
    assert len(jobs) == 21
    by = {}
    for j in jobs:
        by.setdefault(j.method, []).append(j)
    assert {(j.dataset, j.seed) for j in by["supervised_nsga2"]} == {(d, s) for d in ("BUSI", "IDRID") for s in (45, 46, 47)}
    assert {(j.dataset, j.seed) for j in by["nnunet_full"]} == {("IDRID", s) for s in (45, 46, 47)}
    assert {(j.dataset, j.seed) for j in by["unext"]} == {(d, s) for d in ("BUSI", "CVC", "IDRID", "KVASIR")
                                                          for s in (45, 46, 47)}
    assert all(j.spans_sessions == (j.method == "nnunet_full") for j in jobs)
    assert sum(j.external for j in jobs) == 3                                            # Kvasir-SEG UNeXt
    for w in "123":
        assert revision_plan(w)[0].spans_sessions and len({j.seed for j in revision_plan(w)}) == 1
    assert len(revision_plan(optional=True)) == 24


def test_scheduler_starts_spanning_jobs_and_handles_pause(tmp_path):
    import sys

    from qenas.study.plan import Job
    from qenas.study.scheduler import run_study
    jobs = [Job("IDRID", "nnunet_full", 45), Job("IDRID", "unext", 45)]
    seen = {}

    def command(job, device):
        seen[job.id] = True
        code = 75 if job.method == "nnunet_full" else 0
        return [sys.executable, "-c", f"import os, sys; assert 'QENAS_STOP_AT' in os.environ; sys.exit({code})"]

    st = run_study(tmp_path, [None, None], budget_hours=0.6, jobs=jobs, command=command, poll_seconds=0.2,
                   log=lambda *_: None)
    assert seen == {"IDRID/nnunet_full/seed_45": True, "IDRID/unext/seed_45": True}   # 0.6 h << 700 min: started
    assert st["IDRID/nnunet_full/seed_45"]["status"] == "paused"                      # not retried, not failed


def test_reference_split_guard(data, monkeypatch, tmp_path):
    from qenas.study import revision as rv

    def cfg(name):
        return study_config("KVASIR", "attention_unet", 42, dict(_base(data), **{"paths.results": str(tmp_path / name)}))
    assert run_experiment(cfg("r0"), stop_after="split") == 0
    split = json.loads((experiment_dir(cfg("r0")) / "split.json").read_text())
    ref = {"ids": rv.split_ids(split["splits"])}
    monkeypatch.setenv(rv.REFERENCE_ENV, str(tmp_path))
    (tmp_path / "KVASIR.json").write_text(json.dumps(ref))
    assert run_experiment(cfg("r1"), stop_after="split") == 0                          # identical split: accepted
    bad = copy.deepcopy(ref)
    bad["ids"]["train"] = bad["ids"]["train"][::-1]                                     # same images, other order
    (tmp_path / "KVASIR.json").write_text(json.dumps(bad))
    assert run_experiment(cfg("r2"), stop_after="split") == 1                          # refused at the split stage
    for ds in ("BUSI", "CVC", "IDRID", "KVASIR"):                                        # the saved study-v2 splits
        r = rv.reference_split(ds)
        assert rv.ids_digest(r["ids"]) == r["sha256"]
        assert not set(r["ids"]["test"]) & (set(r["ids"]["train"]) | set(r["ids"]["val"]))


def test_tost_and_revision_tables_on_synthetic_runs(tmp_path):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import revision_tables as rt
    rng = np.random.default_rng(0)
    a = {f"i{k}": float(x) for k, x in enumerate(rng.uniform(0.7, 0.9, 120))}
    near = {k: v + float(rng.normal(0, 0.01)) for k, v in a.items()}
    far = {k: v - 0.05 + float(rng.normal(0, 0.01)) for k, v in a.items()}
    assert rt.tost(a, near, 0.02)["equivalent"] and rt.tost(a, near, 0.02)["ci90_inside_margin"]
    assert not rt.tost(a, far, 0.02)["equivalent"]
