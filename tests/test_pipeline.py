"""End-to-end run of the complete QENAS pipeline on a tiny synthetic dataset with the BUSI folder layout.

The protocol is shrunk only in size (resolution, widths, population, epochs) so that it runs in about a minute
on a CPU; the code path is exactly the one of ``python main.py --dataset BUSI``.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from qenas.config import build_config, config_fingerprint
from qenas.experiment import Experiment
from qenas.pipeline import STAGES, Pipeline


def _make_busi(root, n=(28, 14, 2), size=80, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    for cls, k in zip(("benign", "malignant", "normal"), n):
        d = root / "BUSI_synthetic" / cls
        d.mkdir(parents=True)
        for i in range(1, k + 1):
            cy, cx, r = rng.integers(20, 60), rng.integers(20, 60), rng.integers(6, 16)
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
            img = rng.normal(90, 25, (size, size)) + 80 * mask
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(d / f"{cls} ({i}).png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(d / f"{cls} ({i})_mask.png")


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("qenas")
    _make_busi(tmp / "Datasets")
    cfg = build_config("BUSI", overrides={
        "data.root": str(tmp / "Datasets"), "paths.results": str(tmp / "results"), "paths.cache": str(tmp / "cache"),
        "device": "cpu", "data.image_size": [64, 64], "model.base_channels": 16,
        "search.population_size": 4, "search.generations": 1, "search.plot_every": 1, "pareto.max_candidates": 4,
        "candidate_training.epochs": 3, "candidate_training.rungs": [1, 2, 3], "candidate_training.batch_size": 4,
        "loss_weights.alphas": [0.5, 0.8], "loss_weights.criterion_window": 2, "selection.criterion_window": 2,
        "final_training.epochs": 3, "final_training.batch_size": 4,
        "final_training.early_stopping": {"patience": 2, "min_epochs": 2, "significance": 0.05},
    })
    exp = Experiment.open(cfg, config_fingerprint(cfg))
    Pipeline(cfg, exp, torch.device("cpu")).run()
    return cfg, exp


def test_all_stages_complete_in_order(finished_run):
    cfg, exp = finished_run
    assert exp.completed and exp.state["test_evaluated"]
    assert [s for s in exp.state["stages"]] == STAGES
    assert exp.dir == Path(cfg["paths"]["results"]) / "BUSI" / "qenas" / "seed_42"
    # a finished experiment keeps only its final best-validation weights
    kept = [p.relative_to(exp.checkpoints).as_posix() for p in exp.checkpoints.rglob("*") if p.is_file()]
    assert kept == ["final/best.pt"]


def test_search_uses_two_objectives_and_records_progress(finished_run):
    _, exp = finished_run
    front = json.loads((exp.dir / "pareto_front.json").read_text())
    assert all(k in r for r in front for k in ("f1_params", "f2_zico", "zico")) and "f2_synflow" not in front[0]
    gens = json.loads((exp.dir / "nsga2_generations.json").read_text())
    assert len(gens) == 2 and all(0 < g["hypervolume"] for g in gens)
    assert gens[-1]["hypervolume"] >= gens[0]["hypervolume"] - 1e-12   # elitist survival never loses the front
    # pareto.max_candidates = population size: the front is filled up with the rest of the final population
    assert len(json.loads((exp.dir / "pareto_candidates.json").read_text())) == 4


def test_successive_halving_trains_the_best_half_further(finished_run):
    _, exp = finished_run
    sh = json.loads((exp.dir / "successive_halving.json").read_text())
    assert sh["sizes"] == [4, 2, 1] and sh["rung_epochs"] == [1, 2, 3]
    r1, r2, r3 = sh["rungs"]
    best2 = sorted(r1["scores"], key=lambda c: -r1["scores"][c])[:2]
    assert sorted(map(str, r1["promoted"])) == sorted(best2) and r2["trained"] == r1["promoted"]
    res = {r["candidate_id"]: r for r in json.loads((exp.dir / "candidate_results.json").read_text())}
    assert sorted(len(r["history"]) for r in res.values()) == [1, 1, 2, 3]
    sel = json.loads((exp.dir / "selection.json").read_text())
    assert sel["selected_id"] == r3["trained"][0] and len(res[sel["selected_id"]]["history"]) == 3
    lw = json.loads((exp.dir / "loss_weights.json").read_text())   # the alpha = 0.5 run reuses the finalist's run
    assert next(r for r in lw["runs"] if r["alpha"] == 0.5)["status"] == "reused candidate run"


def test_final_metrics_contain_mixed_ggnas_and_added_metrics(finished_run):
    _, exp = finished_run
    final = json.loads((exp.dir / "final_metrics.json").read_text())
    t = final["test"]
    for k in ("dice", "miou", "iou", "sensitivity", "hd95", "loss"):
        assert np.isfinite(t[k])
    assert 0 <= t["dice"] <= 1 and t["hd95"] >= 0
    assert final["params_m"] > 0 and final["gflops_mixed_ggnas"] == pytest.approx(4 * final["gflops_per_image"])
    row = json.loads((exp.dir / "final_result.json").read_text())[0]
    assert {"Dice / DSC (%)", "mIoU (%)", "Params (M)", "HD95 (px)", "Sensitivity (%)", "ms / image"} <= set(row)
    assert final["inference_ms_per_image"] > 0
    # per-image test metrics and predictions (statistics and qualitative figures of the study)
    per = list(csv.DictReader(open(exp.dir / "test_per_image.csv", encoding="utf-8")))
    assert len(per) == final["n_test"]
    assert abs(np.mean([float(r["dice"]) for r in per]) - t["dice"]) < 1e-9
    z = np.load(exp.dir / "test_predictions.npz")
    assert z["masks"].shape == (final["n_test"], 64, 64) and list(z["sample_ids"]) == [r["sample_id"] for r in per]
    assert "tta" not in json.dumps(final).lower() and final["alpha"] in (0.5, 0.8)
    assert (exp.dir / "summary.txt").exists() and (exp.dir / "figures" / "successive_halving.png").exists()


def test_rerun_skips_everything_and_never_retests(finished_run):
    cfg, exp = finished_run
    again = Experiment.open(cfg, exp.fingerprint)
    p = Pipeline(cfg, again, torch.device("cpu"))
    assert p.stage_test() == {"skipped": True}
