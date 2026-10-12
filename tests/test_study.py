"""End-to-end test of the evaluation study on tiny synthetic Kvasir-SEG and CVC-ClinicDB datasets (official folder
layouts). The protocol is shrunk only in size; every study condition runs the code path of study.py."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from qenas.config import config_fingerprint
from qenas.experiment import experiment_dir
from qenas.pipeline import study_config
from qenas.runner import run_experiment
from qenas.study.analysis import analyse
from qenas.study.external import external_validation
from qenas.study.plan import Job, plan
from qenas.study.scheduler import job_done, run_study

SMALL = {"device": "cpu", "data.image_size": [64, 64], "model.base_channels": 16,
         "search.population_size": 4, "search.generations": 1, "search.plot_every": 1, "pareto.max_candidates": 2,
         "candidate_training.epochs": 3, "candidate_training.batch_size": 4,
         "loss_weights.alphas": [0.5, 0.8], "loss_weights.criterion_window": 2, "selection.noise_window": 2,
         "final_training.epochs": 3, "final_training.batch_size": 4,
         "final_training.early_stopping": {"patience": 2, "min_epochs": 2, "significance": 0.05}}


def _blob(rng, size):
    yy, xx = np.mgrid[:size, :size]
    cy, cx, r = rng.integers(20, size - 20), rng.integers(20, size - 20), rng.integers(6, 16)
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < r ** 2
    img = np.clip(rng.normal(90, 25, (size, size, 3)) + 80 * mask[..., None], 0, 255).astype(np.uint8)
    return img, mask


def _make_kvasir(root, n=36, n_test=8, seed=0):
    rng = np.random.default_rng(seed)
    home = root / "Kvasir-SEG" / "Kvasir-SEG"
    (home / "images").mkdir(parents=True)
    (home / "masks").mkdir()
    ids = [f"img{i:03d}" for i in range(n)]
    for i in ids:
        img, mask = _blob(rng, 80)
        Image.fromarray(img).save(home / "images" / f"{i}.jpg", quality=95)
        Image.fromarray((mask * 255).astype(np.uint8)).save(home / "masks" / f"{i}.jpg", quality=95)
    (home / "train.txt").write_text("\n".join(ids[n_test:]) + "\n")
    (home / "val.txt").write_text("\n".join(ids[:n_test]) + "\n")


def _make_cvc(root, n=20, seed=1):
    rng = np.random.default_rng(seed)
    d = root / "CVC-ClinicDB"
    (d / "PNG" / "Original").mkdir(parents=True)
    (d / "PNG" / "Ground Truth").mkdir(parents=True)
    rows = []
    for k in range(1, n + 1):
        img, mask = _blob(rng, 80)
        Image.fromarray(img).save(d / "PNG" / "Original" / f"{k}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(d / "PNG" / "Ground Truth" / f"{k}.png")
        rows.append({"frame_id": k, "sequence_id": 1 + (k - 1) // 4, "png_image_path": f"PNG/Original/{k}.png"})
    with open(d / "metadata.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame_id", "sequence_id", "png_image_path"])
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("study")
    _make_kvasir(tmp / "Datasets")
    _make_cvc(tmp / "Datasets")
    base = {**SMALL, "data.root": str(tmp / "Datasets"), "paths.results": str(tmp / "results"),
            "paths.cache": str(tmp / "cache")}

    def cfg(ds, method, split_unit="dataset"):
        return study_config(ds, method, 42, overrides=base, split_unit=split_unit)

    for m in ("qenas", "qenas_random_pick", "qenas_zico_pick", "qenas_no_zico", "attention_unet", "random_search"):
        assert run_experiment(cfg("KVASIR", m)) == 0, m
    external_validation(cfg("KVASIR", "qenas"), torch.device("cpu"))
    assert run_experiment(cfg("CVC", "qenas", "image")) == 0
    return tmp, cfg


def _state(cfg):
    return json.loads((experiment_dir(cfg) / "state.json").read_text())


def test_every_condition_completes_with_its_own_stages(study):
    _, cfg = study
    assert list(_state(cfg("KVASIR", "attention_unet"))["stages"]) == ["split", "loss_weights", "final", "test", "report"]
    assert list(_state(cfg("KVASIR", "qenas_random_pick"))["stages"]) == \
        ["split", "inherit", "selection", "loss_weights", "final", "test", "report"]
    assert list(_state(cfg("KVASIR", "random_search"))["stages"]) == ["split", "search"]
    for m in ("qenas", "qenas_random_pick", "qenas_zico_pick", "qenas_no_zico", "attention_unet", "random_search"):
        assert _state(cfg("KVASIR", m))["status"] == "completed"
        assert _state(cfg("KVASIR", m))["total_seconds"] > 0          # start-to-end timer


def test_all_conditions_use_the_identical_split(study):
    _, cfg = study
    splits = [json.loads((experiment_dir(cfg("KVASIR", m)) / "split.json").read_text())["splits"]
              for m in ("qenas", "attention_unet", "random_search", "qenas_no_zico")]
    assert all(s == splits[0] for s in splits)
    assert [len(splits[0][k]) for k in ("test",)] == [8]              # the official val.txt images


def test_selection_ablations_reuse_the_qenas_candidates(study):
    _, cfg = study
    q = experiment_dir(cfg("KVASIR", "qenas"))
    for m, rule in (("qenas_random_pick", "random"), ("qenas_zico_pick", "max_zico")):
        d = experiment_dir(cfg("KVASIR", m))
        assert (d / "candidate_results.json").read_text() == (q / "candidate_results.json").read_text()
        sel = json.loads((d / "selection.json").read_text())
        assert sel["selection_rule"] == rule and "learning_capability_choice" in sel
    sel = json.loads((experiment_dir(cfg("KVASIR", "qenas_zico_pick")) / "selection.json").read_text())
    pool = [a for a in sel["analysis"] if not a["diverged"]]
    assert sel["selected_id"] == max(pool, key=lambda a: a["zico"])["candidate_id"]


def test_objective_ablation_optimises_two_objectives(study):
    _, cfg = study
    d = experiment_dir(cfg("KVASIR", "qenas_no_zico"))
    assert json.loads((d / "search_summary.json").read_text())["objectives"] == ["f1_params", "f2_synflow"]


def test_baseline_is_the_published_attention_unet(study):
    _, cfg = study
    final = json.loads((experiment_dir(cfg("KVASIR", "attention_unet")) / "final_metrics.json").read_text())
    assert final["architecture"] == {"type": "baseline", "network": "attention_unet"}
    assert abs(final["params_m"] - 8.73) < 0.01


def test_external_validation_and_leakage_condition(study):
    _, cfg = study
    ext = json.loads((experiment_dir(cfg("KVASIR", "qenas")) / "external_CVC" / "external_metrics.json").read_text())
    assert ext["n_images"] == 20 and 0 <= ext["metrics"]["dice"] <= 1
    d = experiment_dir(cfg("CVC", "qenas", "image"))
    assert d.parent.parent.name == "CVC_frame_split"
    groups = {s["group"] for s in json.loads((d / "split.json").read_text())["splits"]["test"]}
    assert all(g.startswith("frame_") for g in groups)


def test_analysis_writes_the_paper_tables(study):
    tmp, _ = study
    analyse(tmp / "results", data_root=str(tmp / "Datasets"))
    a = tmp / "results" / "analysis"
    for name in ("T3_ablations", "T5_search_vs_random", "T6_objective_agreement", "T9_proxies_vs_quality",
                 "T14_external_validation", "T16_cvc_leakage"):
        assert (a / f"{name}.md").exists(), name
    assert (a / "README.md").exists()


def test_scheduler_respects_dependencies_and_time_budget(tmp_path):
    jobs = [Job("BUSI", "qenas", 42), Job("BUSI", "qenas_random_pick", 42, deps=(Job("BUSI", "qenas", 42).id,)),
            Job("BUSI", "random_search", 42)]
    order = tmp_path / "order.txt"

    def fake(job, device):   # marks the job's experiment completed and records the start order
        code = (f"import json, pathlib; d = pathlib.Path(r'{tmp_path}') / '{job.id}'; d.mkdir(parents=True, exist_ok=True);"
                f"(d / 'state.json').write_text(json.dumps({{'status': 'completed', 'total_seconds': 1}}));"
                f"open(r'{order}', 'a').write('{job.id}\\n')")
        return [sys.executable, "-c", code]

    run_study(tmp_path, [None], budget_hours=24, jobs=jobs, command=fake, poll_seconds=0.1, log=lambda *_: None)
    started = order.read_text().split()
    assert all(job_done(tmp_path, j) for j in jobs)
    assert started.index(jobs[0].id) < started.index(jobs[1].id)
    # a job whose expected duration exceeds the remaining budget is not started
    late = tmp_path / "late"
    run_study(late, [None], budget_hours=0.01, jobs=[Job("KVASIR", "qenas", 43)], command=fake, poll_seconds=0.1,
              log=lambda *_: None)
    assert not job_done(late, Job("KVASIR", "qenas", 43))


def test_full_plan_is_consistent():
    jobs = plan()
    ids = {j.id for j in jobs}
    assert len(jobs) == 78 and all(d in ids for j in jobs for d in j.deps)
    assert sum(j.external for j in jobs) == 9
