"""nnU-Net with its default schedule (``nnunet_full``: ``nnUNetTrainer``, 1000 epochs x 250 iterations) - journal
revision, IDRiD (required) and BUSI (optional), seeds 45-47.

Everything is the study's nnU-Net baseline (``qenas/study/nnunet.py``: same data export of the preprocessed images,
fold 0 = the QENAS train/validation split, default plans, 2d configuration, final checkpoint, test without TTA and,
secondary, with nnU-Net's default mirroring, scored by the study's metric code). Only the trainer differs: the
default ``nnUNetTrainer`` instead of ``nnUNetTrainer_100epochs``.

One run takes about 11.5 h (IDRiD) / 30 h (BUSI) on a Tesla T4, longer than a 12-hour Kaggle session, so training
continues across sessions exactly as nnU-Net's own ``nnUNetv2_train ... --c`` does (``maybe_load_checkpoint`` with
``continue_training=True``, i.e. from ``checkpoint_latest.pth``):

* **Session deadline.** The job reads ``QENAS_STOP_AT`` (Unix time, set by the study scheduler) and stops at the end
  of the first epoch whose successor would not finish before it, or on SIGTERM. nnU-Net's own ``on_epoch_end`` then
  writes ``checkpoint_latest.pth`` (the checkpoint interval is set to 1 for that epoch; checkpointing does not
  change training), the data-augmentation workers are shut down, and the job exits with code 75 (paused).
* **Persistent copy.** Every checkpoint nnU-Net writes (every 5 epochs and at the pause) is copied atomically to
  ``<experiment>/nnunet_resume/`` with a manifest (epoch, SHA-256 of the checkpoint, of the plans and of the split).
  The Kaggle cell stores this folder in a private Kaggle dataset of the same account and restores it at the start of
  the next session.
* **Restore.** In a new session the data export, fingerprint, plans and preprocessing are recomputed (deterministic;
  the plans' SHA-256 must equal the manifest's), the checkpoint is copied back into nnU-Net's fold folder, verified,
  and nnU-Net continues from its epoch. The experiment's ``state.json`` (pushed with the results) records the last
  persisted epoch; a restore that finds an older or no checkpoint stops with an error instead of silently starting
  again (``QENAS_NNUNET_ALLOW_RESTART=1`` overrides).
* **Time.** The training time of every session is accumulated, so the stage time is the total GPU time.

As for nnU-Net in general (DESIGN_DECISIONS D47), its data loading is not seeded, so runs are independent repetitions
but not bit-reproducible; resuming changes only the (unseeded) augmentation stream, as with ``--c``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..experiment import SessionPause
from ..utils.io import load_json, save_json
from ..utils.logging_utils import format_seconds, get_logger
from .nnunet import FOLD, PLANS, NNUNetPipeline, dataset_name

log = get_logger()

PERSIST_DIR = "nnunet_resume"
STOP_ENV = "QENAS_STOP_AT"
RESTART_ENV = "QENAS_NNUNET_ALLOW_RESTART"
EPOCH_MARGIN = 1.5            # stop when the next epoch (x this factor) would end after the deadline


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stop_time() -> float:
    try:
        return float(os.environ.get(STOP_ENV, "inf"))
    except ValueError:
        return float("inf")


class NNUNetFullPipeline(NNUNetPipeline):
    """``nnunet_full``: the nnU-Net baseline with the default 1000-epoch trainer, resumable across sessions."""

    @property
    def persist(self) -> Path:
        return self.exp.dir / PERSIST_DIR

    @property
    def fold_dir(self) -> Path:
        return self.model_dir / f"fold_{FOLD}"

    def _manifest(self) -> Optional[Dict[str, Any]]:
        p = self.persist / "manifest.json"
        return load_json(p) if p.exists() else None

    def _plans_sha(self) -> str:
        pre = self.work / "preprocessed" / dataset_name(self.cfg["data"]["dataset"])
        return sha256_file(pre / f"{PLANS}.json")

    def _split_sha(self) -> str:
        return hashlib.sha256(json_bytes(load_json(self.exp.dir / "nnunet_cases.json")["cases"])).hexdigest()

    # ------------------------------------------------------------------ session handling
    def run(self, stop_after=None) -> None:
        # A finished training whose final checkpoint is only in the persistent copy (new session before the test):
        # restore it, so the base class does not repeat the training.
        man = self._manifest()
        final = self.fold_dir / "checkpoint_final.pth"
        if self.exp.is_done("nnunet_train") and not final.exists() and man and man["file"] == "checkpoint_final.pth" \
                and not self.exp.is_done("nnunet_test"):
            self._restore(man, require_plans=False)
            # nnU-Net's predictor also needs plans.json / dataset.json of the model folder (saved with the checkpoint)
            if not (self.model_dir / "plans.json").exists():
                for f in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
                    if (self.persist / f).exists():
                        shutil.copy2(self.persist / f, self.model_dir / f)
        super().run(stop_after=stop_after)

    def _restore(self, man: Dict[str, Any], require_plans: bool = True) -> None:
        src = self.persist / man["file"]
        if not src.exists():
            raise RuntimeError(f"{src} listed in the manifest is missing (restore the account's persistent storage)")
        digest = sha256_file(src)
        if digest != man["sha256"]:
            raise RuntimeError(f"{src} is corrupted (sha256 {digest[:12]} != manifest {man['sha256'][:12]})")
        if require_plans and man.get("plans_sha256") != self._plans_sha():
            raise RuntimeError("nnU-Net plans of this session differ from the plans of the persisted checkpoint "
                               f"(sha256 {self._plans_sha()[:12]} != {str(man.get('plans_sha256'))[:12]}); refusing to "
                               "resume a different configuration")
        if require_plans and man.get("split_sha256") != self._split_sha():
            raise RuntimeError("nnU-Net cases/split differ from those of the persisted checkpoint; refusing to resume")
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.fold_dir / (man["file"] + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, self.fold_dir / man["file"])
        log.info("Restored %s (epoch %d, sha256 %s) from %s", man["file"], man["epoch"], digest[:12], self.persist)

    def _before_training(self, trainer) -> None:
        """Bring back the latest persisted checkpoint before nnU-Net's ``maybe_load_checkpoint``."""
        progress = self.exp.state.get("nnunet_progress", {})
        have = [f for f in ("checkpoint_final.pth", "checkpoint_latest.pth") if (self.fold_dir / f).exists()]
        man = self._manifest()
        if not have and man:
            if int(man["epoch"]) < int(progress.get("epoch", 0)):
                self._stale(f"the persisted checkpoint is at epoch {man['epoch']}, but this experiment already "
                            f"reached epoch {progress['epoch']}")
            self._restore(man)
        elif not have and int(progress.get("epoch", 0)) > 0:
            self._stale(f"no checkpoint found, but this experiment already reached epoch {progress['epoch']}")
        trainer.print_to_log_file(f"QENAS study: continuing from {have[0] if have else (man or {}).get('file', 'scratch')}")

    def _stale(self, why: str) -> None:
        if os.environ.get(RESTART_ENV) == "1":
            log.warning("%s; restarting nnU-Net training from scratch (%s=1)", why, RESTART_ENV)
            return
        raise RuntimeError(f"{why}. Restore {self.persist} from this account's persistent storage (the Kaggle cell "
                           f"does it automatically from the dataset of the previous session; check that the newest "
                           f"version was used), or set {RESTART_ENV}=1 to start again.")

    def _persist(self, trainer, name: str, epoch: int) -> None:
        """Atomic copy of the checkpoint nnU-Net just wrote, plus manifest and nnU-Net's model-folder files."""
        src = self.fold_dir / name
        self.persist.mkdir(parents=True, exist_ok=True)
        tmp = self.persist / (name + ".tmp")
        shutil.copy2(src, tmp)
        digest = sha256_file(tmp)
        os.replace(tmp, self.persist / name)
        for f in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
            if (self.model_dir / f).exists():
                shutil.copy2(self.model_dir / f, self.persist / f)
        old = self._manifest()
        save_json(self.persist / "manifest.json", {
            "experiment": f"{self.cfg['data']['dataset']}/{self.method_name}/seed_{self.seed}", "file": name,
            "epoch": int(epoch), "num_epochs": int(trainer.num_epochs), "sha256": digest,
            "plans_sha256": self._plans_sha(), "split_sha256": self._split_sha(), "trainer": self.trainer_name,
            "saved": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if old and old["file"] != name and (self.persist / old["file"]).exists():
            (self.persist / old["file"]).unlink()
        keep = self.exp.dir / "nnunet"                        # text logs of every session (pushed with the results)
        keep.mkdir(exist_ok=True)
        for f in self.fold_dir.glob("training_log_*.txt"):
            shutil.copy2(f, keep / f.name)
        self.exp.state["nnunet_progress"] = {**self.exp.state.get("nnunet_progress", {}), "epoch": int(epoch),
                                             "checkpoint": name, "sha256": digest, "saved": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.exp.save_state()

    def _run_training(self, trainer) -> None:
        deadline = stop_time()
        flag = {"stop": False, "t": time.time(), "epoch_s": 0.0}
        prev_handler = None
        try:
            prev_handler = signal.signal(signal.SIGTERM, lambda *_: flag.__setitem__("stop", True))
        except ValueError:                                     # not the main thread (tests)
            pass
        orig = trainer.on_epoch_end
        if trainer.current_epoch >= trainer.num_epochs:        # restored a final state: nothing left to train
            log.info("nnU-Net training already complete (%d epochs)", trainer.num_epochs)

        def on_epoch_end():
            ep = trainer.current_epoch                         # 0-based epoch that just finished
            now = time.time()
            flag["epoch_s"] = now - flag["t"]
            flag["t"] = now
            last = ep == trainer.num_epochs - 1
            pause = not last and (flag["stop"] or now + EPOCH_MARGIN * flag["epoch_s"] >= deadline)
            if pause:
                trainer.save_every = 1                         # nnU-Net writes checkpoint_latest for this epoch
            orig()
            if (ep + 1) % trainer.save_every == 0 and not last:
                self._persist(trainer, "checkpoint_latest.pth", ep + 1)
            if pause:
                raise SessionPause(f"nnU-Net paused after epoch {ep + 1}/{trainer.num_epochs} "
                                   f"({'SIGTERM' if flag['stop'] else 'session deadline'})")

        trainer.on_epoch_end = on_epoch_end
        try:
            trainer.run_training()
        except SessionPause:
            for dl in (trainer.dataloader_train, trainer.dataloader_val):
                if dl is not None and hasattr(dl, "_finish"):
                    dl._finish()
            raise
        finally:
            trainer.on_epoch_end = orig
            if prev_handler is not None:
                signal.signal(signal.SIGTERM, prev_handler)
        self._persist(trainer, "checkpoint_final.pth", trainer.num_epochs)

    def stage_nnunet_train(self) -> Dict[str, Any]:
        before = float(self.exp.state.get("nnunet_progress", {}).get("train_seconds", 0.0))   # earlier sessions
        t0 = time.time()

        def account(done: bool) -> Dict[str, Any]:
            prog = self.exp.state.setdefault("nnunet_progress", {})
            prog["train_seconds"] = round(before + time.time() - t0, 1)
            prog["sessions"] = int(prog.get("sessions", 0)) + 1
            prog["completed"] = done
            self.exp.save_state()
            log.info("nnU-Net training: %s this session, %s in %d session(s)", format_seconds(time.time() - t0),
                     format_seconds(prog["train_seconds"]), prog["sessions"])
            return prog

        try:
            info = super().stage_nnunet_train()
        except SessionPause:
            account(False)
            raise
        prog = account(True)
        info.update(seconds_before=before, sessions=prog["sessions"])
        return info


def json_bytes(obj: Any) -> bytes:
    import json
    return json.dumps(obj, sort_keys=True).encode("utf-8")
