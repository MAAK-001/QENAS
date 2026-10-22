"""nnU-Net baseline of the evaluation study (Isensee et al., Nature Methods 18, 2021; nnunetv2 2.8.1).

nnU-Net runs its own, unmodified self-configuring pipeline - fingerprint, default ``ExperimentPlanner`` plans,
preprocessing (per-image z-score normalisation), data augmentation, Dice + cross-entropy loss with deep
supervision, SGD with poly learning rate - with three study settings:

* **Data.** The same images as every other method: the QENAS split (``split.json``; fold 0 of nnU-Net =
  QENAS train / validation, written as ``splits_final.json``), at the QENAS resolution, exported from the same
  preprocessed arrays as 3-channel PNGs (``NaturalImage2DIO``, channel names R, G, B as in nnU-Net's official 2D
  example Dataset120_RoadSegmentation). The test images are exported only in the test stage (test isolation).
* **Budget.** The official trainer variant ``nnUNetTrainer_100epochs`` (100 epochs x 250 iterations) instead of
  the default 1000 epochs, i.e. at least the GPU time of the complete QENAS pipeline (DESIGN_DECISIONS D47).
* **Inference.** The final checkpoint (nnU-Net's default), the ``2d`` configuration, no test-time augmentation
  (as for every method of the study); nnU-Net's default mirroring TTA is reported in addition.

Predictions are scored with the study's own metric code at the network resolution, so every number is directly
comparable with the other methods. ``torch.compile`` is disabled (``nnUNet_compile=f``; speed only). nnU-Net's
data loading is not seeded by nnU-Net itself, so runs are not bit-reproducible; the seed sets the global
``random`` / ``numpy`` / ``torch`` generators.
"""

from __future__ import annotations

import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from ..datasets.preprocessing import preload_split
from ..pipeline import Pipeline, write_final_result
from ..training.metrics import MetricAccumulator
from ..training.trainer import inference_time_ms
from ..utils.env import environment_info
from ..utils.io import load_json, save_csv, save_json
from ..utils.logging_utils import get_logger

log = get_logger()

NNUNET_VERSION = "2.8.1"
TRAINER = "nnUNetTrainer_100epochs"
PLANS = "nnUNetPlans"
CONFIGURATION = "2d"
FOLD = 0
DATASET_IDS = {"BUSI": 901, "CVC": 902, "IDRID": 903, "KVASIR": 904}
SAVE_EVERY = 5               # checkpoint frequency (nnU-Net default 50); does not change training


def dataset_name(dataset: str) -> str:
    return f"Dataset{DATASET_IDS[dataset]}_{dataset}"


def _case(i: int, split: str) -> str:
    return f"{split}_{i:04d}"


def score(preds: List[np.ndarray], masks: np.ndarray) -> Tuple[Dict[str, float], List[Dict[str, float]], List]:
    """Study metrics (Dice, IoU, mIoU, sensitivity, HD95) of binary predictions, as ``evaluate`` computes them."""
    acc = MetricAccumulator(boundary=True, keep=True)
    for p, g in zip(preds, masks):
        logits = torch.from_numpy(np.where(p > 0, 10.0, -10.0)[None, None].astype(np.float32))
        acc.update(torch.zeros(1, dtype=torch.float64), logits, torch.from_numpy(g[None, None].astype(np.float32)))
    m = acc.compute()
    m.pop("loss", None)
    for r in acc.rows:
        r.pop("loss", None)
    return m, acc.rows, acc.predictions


class NNUNetPipeline(Pipeline):
    """Stages: split -> nnunet_data -> nnunet_train -> nnunet_test -> report."""

    def run(self, stop_after=None) -> None:
        # nnU-Net's working folder is not pushed to GitHub. If a new session finds data export or training done but
        # its files missing (and the test not yet evaluated), those stages are repeated.
        ckpt = self.model_dir / f"fold_{FOLD}" / "checkpoint_final.pth"
        lost = (self.exp.is_done("nnunet_train") and not ckpt.exists()) or                (self.exp.is_done("nnunet_data") and not (self.raw_dir / "dataset.json").exists())
        if lost and not self.exp.is_done("nnunet_test"):
            log.warning("nnU-Net working files of %s are missing (new session); repeating data export and training",
                        self.exp.dir)
            for s in ("nnunet_data", "nnunet_train"):
                self.exp.state["stages"].pop(s, None)
            self.exp.save_state()
        super().run(stop_after=stop_after)

    # ------------------------------------------------------------------ nnU-Net environment
    @property
    def work(self) -> Path:
        return self.exp.dir / "nnunet_work"

    def _env(self) -> None:
        """nnU-Net reads its folders from the environment when it is imported."""
        os.environ["nnUNet_raw"] = str(self.work / "raw")
        os.environ["nnUNet_preprocessed"] = str(self.work / "preprocessed")
        os.environ["nnUNet_results"] = str(self.work / "results")
        os.environ["nnUNet_compile"] = "f"
        os.environ.setdefault("nnUNet_n_proc_DA", os.environ.get("OMP_NUM_THREADS", "2"))
        for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
            Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

    @property
    def raw_dir(self) -> Path:
        return self.work / "raw" / dataset_name(self.cfg["data"]["dataset"])

    @property
    def model_dir(self) -> Path:
        return (self.work / "results" / dataset_name(self.cfg["data"]["dataset"]) /
                f"{TRAINER}__{PLANS}__{CONFIGURATION}")

    def _predictor(self, mirroring: bool):
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=mirroring,
                               perform_everything_on_device=self.device.type == "cuda", device=self.device,
                               verbose=False, allow_tqdm=False)
        pred.initialize_from_trained_model_folder(str(self.model_dir), use_folds=(FOLD,),
                                                  checkpoint_name="checkpoint_final.pth")
        return pred

    def predict_arrays(self, images: np.ndarray, folder: Path, mirroring: bool) -> List[np.ndarray]:
        """nnU-Net predictions (binary, at the network resolution) of ``images`` [N, H, W, 3] uint8."""
        self._env()
        src, out = folder / "images", folder / ("pred_mirroring" if mirroring else "pred")
        shutil.rmtree(folder, ignore_errors=True)
        src.mkdir(parents=True)
        out.mkdir(parents=True)
        for i, img in enumerate(images):
            Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).save(src / f"case_{i:04d}_0000.png")
        self._predictor(mirroring).predict_from_files_sequential(str(src), str(out), save_probabilities=False,
                                                                 overwrite=True)
        preds = [(np.asarray(Image.open(out / f"case_{i:04d}.png")) > 0).astype(np.uint8) for i in range(len(images))]
        shutil.rmtree(folder, ignore_errors=True)
        return preds

    # ------------------------------------------------------------------ stages
    def stage_nnunet_data(self) -> Dict[str, Any]:
        """nnU-Net raw dataset of the training and validation images (the test images are exported later)."""
        self._env()
        raw = self.raw_dir
        shutil.rmtree(raw, ignore_errors=True)
        (raw / "imagesTr").mkdir(parents=True)
        (raw / "labelsTr").mkdir(parents=True)
        data = self.data
        cases = {"train": [], "val": []}
        for split, imgs, masks in (("train", data.train_images, data.train_masks),
                                   ("val", data.val_images, data.val_masks)):
            for i, (img, m) in enumerate(zip(imgs, masks)):
                c = _case(i, split)
                Image.fromarray(np.ascontiguousarray(img, dtype=np.uint8)).save(raw / "imagesTr" / f"{c}_0000.png")
                Image.fromarray(np.ascontiguousarray(m, dtype=np.uint8)).save(raw / "labelsTr" / f"{c}.png")
                cases[split].append(c)
        from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
        generate_dataset_json(str(raw), {0: "R", 1: "G", 2: "B"}, {"background": 0, "foreground": 1},
                              len(cases["train"]) + len(cases["val"]), ".png",
                              dataset_name=dataset_name(self.cfg["data"]["dataset"]),
                              description="QENAS evaluation study: the QENAS split at the QENAS resolution",
                              converted_by="QENAS study (qenas/study/nnunet.py)")
        ids = {c: s.sample_id for split in ("train", "val") for c, s in zip(cases[split], self.splits[split])}
        save_json(self.exp.dir / "nnunet_cases.json", {"cases": cases, "sample_ids": ids})
        log.info("nnU-Net raw dataset %s: %d training + %d validation cases (%dx%d PNG)", raw.name,
                 len(cases["train"]), len(cases["val"]), *self.cfg["data"]["image_size"])
        return {"train": len(cases["train"]), "val": len(cases["val"])}

    def stage_nnunet_train(self) -> Dict[str, Any]:
        self._env()
        did = DATASET_IDS[self.cfg["data"]["dataset"]]
        nproc = max(1, int(os.environ.get("nnUNet_n_proc_DA", "2")))
        from nnunetv2.experiment_planning.plan_and_preprocess_api import (extract_fingerprints, plan_experiments,
                                                                          preprocess)
        pre = self.work / "preprocessed" / dataset_name(self.cfg["data"]["dataset"])
        if not (pre / f"{PLANS}.json").exists():
            extract_fingerprints([did], num_processes=nproc, check_dataset_integrity=True)
            plan_experiments([did])
            preprocess([did], plans_identifier=PLANS, configurations=(CONFIGURATION,), num_processes=(nproc,))
        cases = load_json(self.exp.dir / "nnunet_cases.json")["cases"]
        save_json(pre / "splits_final.json", [{"train": cases["train"], "val": cases["val"]}])
        random.seed(self.seed)
        np.random.seed(self.seed % (2 ** 32))
        torch.manual_seed(self.seed)
        from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
        trainer = get_trainer_from_args(str(did), CONFIGURATION, FOLD, TRAINER, PLANS, device=self.device)
        trainer.save_every = SAVE_EVERY
        maybe_load_checkpoint(trainer, continue_training=True, validation_only=False)
        if self.device.type == "cuda":
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        t0 = time.time()
        trainer.run_training()
        log.info("nnU-Net training finished in %.1f min (%d epochs x %d iterations, batch size %d, patch %s)",
                 (time.time() - t0) / 60, trainer.num_epochs, trainer.num_iterations_per_epoch, trainer.batch_size,
                 trainer.configuration_manager.patch_size)
        info = {"num_epochs": trainer.num_epochs, "iterations_per_epoch": trainer.num_iterations_per_epoch,
                "batch_size": trainer.batch_size, "patch_size": list(trainer.configuration_manager.patch_size)}
        # keep nnU-Net's own record (the work folder with data and weights is not pushed)
        keep = self.exp.dir / "nnunet"
        keep.mkdir(exist_ok=True)
        for f in [pre / f"{PLANS}.json", pre / "dataset_fingerprint.json", pre / "splits_final.json"] + \
                list((self.model_dir / f"fold_{FOLD}").glob("training_log_*.txt")) + \
                [self.model_dir / f"fold_{FOLD}" / "debug.json"]:
            if f.exists():
                shutil.copy2(f, keep / f.name)
        shutil.rmtree(pre / f"{PLANS}_{CONFIGURATION}", ignore_errors=True)    # preprocessed arrays
        for f in (self.model_dir / f"fold_{FOLD}").glob("checkpoint_*.pth"):
            if f.name != "checkpoint_final.pth":
                f.unlink()
        return info

    def stage_nnunet_test(self) -> Dict[str, Any]:
        if self.exp.state.get("test_evaluated"):
            log.info("Test set already evaluated once for this experiment — not re-evaluating.")
            return {"skipped": True}
        h, w = self.cfg["data"]["image_size"]
        cin = int(self.cfg["data"]["in_channels"])
        val_m, _, _ = score(self.predict_arrays(self.data.val_images, self.work / "val_tmp", False),
                            self.data.val_masks)
        test_imgs, test_masks = preload_split(self.splits["test"], (h, w), int(self.cfg["data"]["mask_threshold"]),
                                              self.cfg["data"]["dataset"], "test",
                                              Path(self.cfg["paths"]["cache"]) / "preprocessed")
        log.info("Evaluating the held-out TEST split exactly once | nnU-Net final checkpoint, no TTA "
                 "(+ nnU-Net's mirroring TTA reported separately)")
        test_m, rows, preds = score(self.predict_arrays(test_imgs, self.work / "test_tmp", False), test_masks)
        test_mirror, _, _ = score(self.predict_arrays(test_imgs, self.work / "test_tmp", True), test_masks)
        self.exp.state["test_evaluated"] = True
        self.exp.state["test_evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.exp.save_state()
        ids = [s.sample_id for s in self.splits["test"]]
        save_csv(self.path("test_per_image.csv"), [{"sample_id": i, **r} for i, r in zip(ids, rows)])
        np.savez_compressed(self.path("test_predictions.npz"), sample_ids=np.array(ids), masks=np.stack(preds))
        from ..models.summary import cost_summary
        net = self._predictor(False).network.to(self.device)
        bs = int(self.cfg["final_training"]["batch_size"])
        cost = cost_summary(net, [h, w], cin, bs)
        latency = inference_time_ms(net, [cin, h, w], self.device)
        train_info = self.exp.state["stages"]["nnunet_train"]
        fv = {"best_epoch": train_info["num_epochs"], "epochs_run": train_info["num_epochs"], "stopped_early": False,
              "stop_reason": "", "checkpoint": "final (nnU-Net default)", "alpha": None, "beta": None,
              "train_metrics_best_checkpoint": None, "val_metrics_best_checkpoint": val_m}
        save_json(self.path("final_validation.json"), fv)
        final = {
            "dataset": self.cfg["data"]["dataset"], "method": self.method_name, "method_label": self.method["label"], "seed": self.seed,
            "architecture": {"type": "nnunet", "configuration": CONFIGURATION, "trainer": TRAINER, "plans": PLANS,
                             "nnunetv2": NNUNET_VERSION},
            "architecture_label": f"nnU-Net {CONFIGURATION} ({TRAINER})", "blocks": None,
            **cost, "inference_ms_per_image": latency, "inference_device": str(self.device),
            "alpha": None, "beta": None, "best_epoch": fv["best_epoch"], "epochs_run": fv["epochs_run"],
            "train": None, "validation": val_m, "test": test_m, "n_test": len(ids),
            "test_with_nnunet_mirroring_tta": test_mirror, "nnunet_training": train_info,
        }
        save_json(self.path("final_metrics.json"), final)
        write_final_result(self.exp.dir, final)
        log.info("TEST | nnU-Net | Dice (DSC) %.4f | mIoU %.4f | HD95 %.2f px | Sensitivity %.4f | Params %.3f M | "
                 "%.2f ms/image | with mirroring TTA: Dice %.4f", test_m["dice"], test_m["miou"], test_m["hd95"],
                 test_m["sensitivity"], cost["params_m"], latency, test_mirror["dice"])
        return {"test_dice": test_m["dice"], "test_dice_mirroring_tta": test_mirror["dice"]}

    def stage_report(self) -> Dict[str, Any]:
        final = load_json(self.path("final_metrics.json"))
        save_json(self.path("reproducibility.json"), {
            "fingerprint": self.fp, "config": self.cfg, "environment": environment_info(self.device),
            "split_summary": load_json(self.path("split.json"))["summary"], "architecture": final["architecture"],
            "nnunet_training": final["nnunet_training"], "stages": self.exp.state["stages"]})
        t, v = final["test"], final["validation"]
        lines = ["================ nnU-Net RESULT ================", "",
                 f"Dataset: {self.cfg['data']['dataset']}", f"Method: {self.method['label']} — {self.method['doc']}",
                 f"Seed: {self.seed}", "",
                 f"Training: {final['nnunet_training']['num_epochs']} epochs x "
                 f"{final['nnunet_training']['iterations_per_epoch']} iterations, batch size "
                 f"{final['nnunet_training']['batch_size']}, patch {final['nnunet_training']['patch_size']}",
                 f"Validation Dice (final checkpoint): {v['dice']:.4f}", "",
                 f"Final test (n = {final['n_test']}, evaluated once, no TTA):",
                 f"Dice (DSC):        {100 * t['dice']:.2f} %", f"mIoU (class mean): {100 * t['miou']:.2f} %",
                 f"Params:            {final['params_m']:.3f} M",
                 f"FLOPs:             {final['gflops_mixed_ggnas']:.2f} G per batch of {final['flops_batch_size']} "
                 f"({final['gflops_per_image']:.2f} G per image)",
                 f"HD95:              {t['hd95']:.2f} px", f"Sensitivity:       {100 * t['sensitivity']:.2f} %",
                 f"IoU (foreground):  {100 * t['iou']:.2f} %",
                 f"Inference:         {final['inference_ms_per_image']:.2f} ms per image on {final['inference_device']}",
                 f"Dice with nnU-Net mirroring TTA: {100 * final['test_with_nnunet_mirroring_tta']['dice']:.2f} %", "",
                 "================================================", ""]
        self.path("summary.txt").write_text("\n".join(lines), encoding="utf-8")
        return {}


def external_predictions(pipe: NNUNetPipeline, images: np.ndarray) -> List[np.ndarray]:
    """Predictions of a finished nnU-Net experiment on external images (no TTA)."""
    return pipe.predict_arrays(images, pipe.work / "external_tmp", False)
