"""External validation of the evaluation study: models trained on Kvasir-SEG, tested on CVC-ClinicDB.

Protocol (the cross-dataset protocol of the polyp-segmentation literature, e.g. PraNet): the finished Kvasir-SEG
experiment's best-validation checkpoint is applied, unchanged, to all 612 CVC-ClinicDB frames - a different
hospital, device and patient population that played no role in its training or model selection. Preprocessing is
that of the trained model (256 x 256, Kvasir-SEG training-split normalisation statistics), the flip-TTA decision
is the one the model's own validation set made, and the metrics are those of the test stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import build_config
from ..datasets.dataset import ArraySegDataset
from ..datasets.discovery import discover, locate_data_root
from ..datasets.preprocessing import preload_split
from ..experiment import Experiment, experiment_dir
from ..pipeline import Pipeline
from ..training.losses import BCESoftMIoULoss
from ..training.trainer import evaluate, load_best_weights
from ..utils.io import load_json, save_csv, save_json
from ..utils.logging_utils import get_logger

log = get_logger()

SOURCE, TARGET = "KVASIR", "CVC"


def result_path(cfg: Dict[str, Any]) -> Path:
    return experiment_dir(cfg) / f"external_{TARGET}" / "external_metrics.json"


def external_validation(cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """``cfg``: configuration of a finished Kvasir-SEG experiment. Writes ``external_CVC/`` into its directory."""
    if cfg["data"]["dataset"] != SOURCE:
        raise ValueError("external validation is defined for Kvasir-SEG experiments")
    out = result_path(cfg)
    if out.exists():
        return load_json(out)
    exp = Experiment.open(cfg, load_json(experiment_dir(cfg) / "state.json")["fingerprint"])
    if not exp.completed:
        raise RuntimeError(f"{exp.dir} is not finished; external validation needs its final model")
    pipe = Pipeline(cfg, exp, device)
    final = load_json(exp.dir / "final_metrics.json")
    model = pipe.build(final["architecture"])
    load_best_weights(model, exp.checkpoints / "final", device)
    model.to(device)
    mean, std = pipe.data.mean, pipe.data.std              # the model's own (Kvasir-SEG training) statistics
    tcfg = build_config(TARGET, overrides={"data.root": locate_data_root(cfg["data"]["root"], TARGET),
                                           "paths.cache": cfg["paths"]["cache"]})
    samples = sorted(discover(tcfg).samples, key=lambda s: s.sample_id)
    h, w = cfg["data"]["image_size"]
    imgs, masks = preload_split(samples, (h, w), int(tcfg["data"]["mask_threshold"]), TARGET, "all",
                                Path(cfg["paths"]["cache"]) / "preprocessed")
    loader = DataLoader(ArraySegDataset(imgs, masks, mean, std), batch_size=int(cfg["final_training"]["batch_size"]),
                        shuffle=False)
    m, rows, preds = evaluate(model, loader, BCESoftMIoULoss(float(final["alpha"])), device,
                              tta=bool(final["test_tta"]), keep=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = [s.sample_id for s in samples]
    save_csv(out.parent / "external_per_image.csv", [{"sample_id": i, **r} for i, r in zip(ids, rows)])
    np.savez_compressed(out.parent / "external_predictions.npz", sample_ids=np.array(ids), masks=np.stack(preds))
    res = {"source": SOURCE, "target": TARGET, "method": cfg["method"], "seed": int(cfg["seed"]),
           "n_images": len(ids), "tta": bool(final["test_tta"]), "metrics": m,
           "protocol": "Kvasir-SEG best-validation checkpoint applied unchanged to all 612 CVC-ClinicDB frames"}
    save_json(out, res)
    log.info("EXTERNAL %s -> %s | %s seed %d | Dice %.4f | mIoU %.4f | HD95 %.2f px | Sensitivity %.4f", SOURCE, TARGET,
             cfg["method"], cfg["seed"], m["dice"], m["miou"], m["hd95"], m["sensitivity"])
    return res
