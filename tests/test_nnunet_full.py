"""nnU-Net with the default schedule across sessions (``nnunet_full``; needs nnunetv2, skipped without it).

The real nnU-Net pipeline on a tiny synthetic Kvasir-SEG, with the official 5-epoch trainer variant in place of the
1000-epoch default: session 1 is paused by the deadline after its first epoch (exit 75), its work folder is deleted
(a new Kaggle session has only the persisted copy), session 2 restores the checkpoint and finishes the 5 epochs; a
lost checkpoint is refused instead of silently restarting."""

import json
import shutil
import time

import pytest

pytest.importorskip("nnunetv2")

from qenas.experiment import experiment_dir  # noqa: E402
from qenas.methods import METHODS  # noqa: E402
from qenas.pipeline import study_config  # noqa: E402
from qenas.runner import run_experiment  # noqa: E402
from qenas.study import nnunet_full  # noqa: E402
from test_study import SMALL, _make_kvasir  # noqa: E402


def test_nnunet_full_resumes_across_sessions(tmp_path, monkeypatch):
    monkeypatch.setitem(METHODS["nnunet_full"], "trainer", "nnUNetTrainer_5epochs")
    monkeypatch.setenv("nnUNet_n_proc_DA", "1")
    _make_kvasir(tmp_path / "Datasets")
    cfg = study_config("KVASIR", "nnunet_full", 42, overrides={
        **SMALL, "data.root": str(tmp_path / "Datasets"), "paths.results": str(tmp_path / "results"),
        "paths.cache": str(tmp_path / "cache")})
    d = experiment_dir(cfg)

    # session 1: the deadline has passed -> pause after the first epoch with nnU-Net's own checkpoint persisted
    monkeypatch.setenv(nnunet_full.STOP_ENV, str(time.time()))
    assert run_experiment(cfg) == 75
    man = json.loads((d / "nnunet_resume" / "manifest.json").read_text())
    assert man["file"] == "checkpoint_latest.pth" and man["epoch"] == 1 and man["num_epochs"] == 5
    state = json.loads((d / "state.json").read_text())
    assert state["nnunet_progress"]["epoch"] == 1 and state["nnunet_progress"]["sessions"] == 1
    assert "nnunet_train" not in state["stages"] and state["status"] == "running"

    # a lost persistent copy is refused (state.json knows epoch 1 was reached)
    shutil.copytree(d / "nnunet_resume", tmp_path / "keep")
    shutil.rmtree(d / "nnunet_resume")
    shutil.rmtree(d / "nnunet_work")
    monkeypatch.delenv(nnunet_full.STOP_ENV)
    assert run_experiment(cfg) == 1

    # session 2 (new machine): only the persisted copy exists -> restore, continue at epoch 2, finish
    shutil.rmtree(d / "nnunet_work", ignore_errors=True)
    shutil.copytree(tmp_path / "keep", d / "nnunet_resume")
    assert run_experiment(cfg) == 0
    state = json.loads((d / "state.json").read_text())
    assert state["status"] == "completed" and state["test_evaluated"]
    tr = state["stages"]["nnunet_train"]
    assert tr["num_epochs"] == 5 and tr["sessions"] == 2 and tr["seconds"] >= state["nnunet_progress"]["train_seconds"] - 1
    log = "".join(p.read_text() for p in (d / "nnunet").glob("training_log_*.txt"))
    assert "continuing from checkpoint_latest.pth" in log and "epoch 4" in log.lower()
    man = json.loads((d / "nnunet_resume" / "manifest.json").read_text())
    assert man["file"] == "checkpoint_final.pth" and man["epoch"] == 5
    final = json.loads((d / "final_metrics.json").read_text())
    assert final["architecture"]["trainer"] == "nnUNetTrainer_5epochs" and final["method_label"] == "nnU-Net (1000 epochs)"
    assert (d / "test_per_image.csv").exists() and (d / "test_per_image_mirroring_tta.csv").exists()
    assert 0 <= final["test"]["dice"] <= 1 and 0 <= final["test_with_nnunet_mirroring_tta"]["dice"] <= 1
