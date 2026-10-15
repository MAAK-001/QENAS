"""End-to-end test of the nnU-Net baseline (needs the nnunetv2 package; skipped without it).

Runs the real nnU-Net pipeline (dataset export, fingerprint, plans, preprocessing, training, prediction) on a tiny
synthetic Kvasir-SEG with the official 1-epoch trainer variant instead of 100 epochs, then the external validation
on a tiny CVC-ClinicDB, with one data-augmentation worker (``nnUNet_n_proc_DA = 1``)."""

import csv
import json

import pytest
import torch

pytest.importorskip("nnunetv2")

from qenas.experiment import experiment_dir  # noqa: E402
from qenas.pipeline import study_config  # noqa: E402
from qenas.runner import run_experiment  # noqa: E402
from qenas.study import nnunet  # noqa: E402
from qenas.study.external import external_validation  # noqa: E402
from test_study import SMALL, _make_cvc, _make_kvasir  # noqa: E402


def test_nnunet_baseline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(nnunet, "TRAINER", "nnUNetTrainer_1epoch")
    monkeypatch.setenv("nnUNet_n_proc_DA", "1")
    _make_kvasir(tmp_path / "Datasets")
    _make_cvc(tmp_path / "Datasets")
    cfg = study_config("KVASIR", "nnunet", 42, overrides={
        **SMALL, "data.root": str(tmp_path / "Datasets"), "paths.results": str(tmp_path / "results"),
        "paths.cache": str(tmp_path / "cache")})
    assert run_experiment(cfg) == 0
    d = experiment_dir(cfg)
    state = json.loads((d / "state.json").read_text())
    assert list(state["stages"]) == ["split", "nnunet_data", "nnunet_train", "nnunet_test", "report"]
    final = json.loads((d / "final_metrics.json").read_text())
    assert final["architecture"]["type"] == "nnunet" and final["architecture"]["trainer"] == "nnUNetTrainer_1epoch"
    assert 0 <= final["test"]["dice"] <= 1 and 0 <= final["test_with_nnunet_mirroring_tta"]["dice"] <= 1
    assert final["params_m"] > 0 and final["inference_ms_per_image"] > 0
    per = list(csv.DictReader(open(d / "test_per_image.csv", encoding="utf-8")))
    assert len(per) == final["n_test"] == 8
    cases = json.loads((d / "nnunet" / "splits_final.json").read_text())[0]
    split = json.loads((d / "split.json").read_text())["splits"]
    assert len(cases["train"]) == len(split["train"]) and len(cases["val"]) == len(split["val"])
    ext = external_validation(cfg, torch.device("cpu"))
    assert ext["n_images"] == 20 and 0 <= ext["metrics"]["dice"] <= 1
