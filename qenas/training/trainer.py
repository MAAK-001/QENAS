"""Generic supervised trainer used by candidate evaluation, loss-weight selection and final training.

Guarantees
* checkpoint after **every** epoch (model, optimiser, scheduler, AMP scaler, epoch,
  history, best metric, RNG states, configuration fingerprint), atomically, with a
  ``.prev`` fallback against corruption;
* automatic resume from the last completed epoch;
* NaN/Inf losses raise :class:`NumericalError` (never silently continue);
* the best-validation-Dice weights are stored separately (``best.pt``) and can be
  restored with :func:`load_best_weights`;
* progress (percentage, elapsed time, ETA) is logged during and after every epoch.

Training metrics are measured on the augmented mini-batches in train mode (standard
practice); validation metrics in eval mode without augmentation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..datasets.dataset import DataBundle, set_loader_epoch
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.env import is_oom_error
from ..utils.io import retry_on_lock, save_csv
from ..utils.logging_utils import ProgressTimer, format_seconds, get_logger
from ..utils.seed import get_rng_states, set_rng_states
from .early_stopping import OverfittingEarlyStopping
from .losses import BCESoftMIoULoss
from .metrics import MetricAccumulator
from .optim import build_optimizer, build_scheduler

log = get_logger()

HISTORY_FIELDS = ["epoch", "train_loss", "val_loss", "train_dice", "val_dice", "train_iou", "val_iou",
                  "train_miou", "val_miou", "learning_rate", "epoch_seconds"]


class NumericalError(RuntimeError):
    """Non-finite loss or metric during training/evaluation."""


def amp_enabled(spec: Any, device: torch.device) -> bool:
    if isinstance(spec, str):
        spec = spec.lower()
        if spec == "auto":
            return device.type == "cuda"
        return spec in ("true", "1", "yes", "on")
    return bool(spec) and device.type == "cuda"


@torch.no_grad()
def evaluate(model: nn.Module, loader, loss_fn: BCESoftMIoULoss, device: torch.device, use_amp: bool = False) -> Dict[str, float]:
    model.eval()
    acc = MetricAccumulator()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        per_sample, _ = loss_fn.per_sample(logits, y)
        if not torch.isfinite(per_sample).all():
            raise NumericalError("non-finite loss during evaluation")
        acc.update(per_sample, logits.float(), y)
    return acc.compute()


def load_best_weights(model: nn.Module, ckpt_dir: Path, device: torch.device) -> Dict[str, Any]:
    state = load_checkpoint(Path(ckpt_dir) / "best.pt", map_location=device, required=True)
    model.load_state_dict(state["model"])
    return state


class Trainer:
    def __init__(self, *, model: nn.Module, data: DataBundle, tcfg: Dict[str, Any], alpha: float,
                 device: torch.device, ckpt_dir: Path, seed: int, fingerprint: str, tag: str,
                 progress_prefix: str = "", early_stopping: Optional[Dict[str, Any]] = None,
                 history_csv: Optional[Path] = None):
        self.model = model.to(device)
        self.data = data
        self.tcfg = tcfg
        self.alpha = float(alpha)
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.seed = int(seed)
        self.fingerprint = fingerprint
        self.tag = tag
        self.prefix = progress_prefix
        self.history_csv = history_csv
        self.epochs = int(tcfg["epochs"])
        self.batch_size = int(tcfg["batch_size"])
        self.loss_fn = BCESoftMIoULoss(alpha=self.alpha)
        self.train_loader = data.train_loader(self.batch_size, seed=self.seed)
        self.val_loader = data.val_loader(self.batch_size)
        self.optimizer = build_optimizer(self.model, tcfg)
        self.scheduler = build_scheduler(self.optimizer, tcfg, steps_per_epoch=len(self.train_loader))
        self.use_amp = amp_enabled(tcfg.get("amp", "auto"), device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip = tcfg.get("grad_clip")
        self.stopper = OverfittingEarlyStopping(early_stopping)
        self.history: List[Dict[str, Any]] = []
        self.best_val_dice = -1.0
        self.best_epoch = -1
        self.completed = False
        self.stop_reason = ""

    # ------------------------------------------------------------------ checkpoint
    @property
    def last_path(self) -> Path:
        return self.ckpt_dir / "last.pt"

    def _save_last(self) -> None:
        save_checkpoint(self.last_path, {
            "tag": self.tag, "fingerprint": self.fingerprint, "alpha": self.alpha,
            "epoch": len(self.history), "completed": self.completed, "stop_reason": self.stop_reason,
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict(), "history": self.history,
            "best_val_dice": self.best_val_dice, "best_epoch": self.best_epoch,
            "rng": get_rng_states(),
        })

    def _compact_after_completion(self) -> None:
        """Drop optimiser state once a run is complete (keeps last weights + history; saves disk)."""
        save_checkpoint(self.last_path, {
            "tag": self.tag, "fingerprint": self.fingerprint, "alpha": self.alpha,
            "epoch": len(self.history), "completed": True, "stop_reason": self.stop_reason,
            "model": self.model.state_dict(), "history": self.history,
            "best_val_dice": self.best_val_dice, "best_epoch": self.best_epoch,
        })
        for p in (self.last_path, self.ckpt_dir / "best.pt"):
            prev = p.with_name(p.name + ".prev")
            if prev.exists():
                retry_on_lock(prev.unlink)

    def _try_resume(self) -> int:
        state = load_checkpoint(self.last_path, map_location=self.device)
        if state is None:
            return 0
        if state.get("fingerprint") != self.fingerprint or state.get("tag") != self.tag:
            raise RuntimeError(f"Checkpoint {self.last_path} belongs to a different run/configuration; refusing to resume.")
        self.history = state["history"]
        self.best_val_dice = state["best_val_dice"]
        self.best_epoch = state["best_epoch"]
        self.completed = bool(state["completed"])
        self.stop_reason = state.get("stop_reason", "")
        self.model.load_state_dict(state["model"])
        if not self.completed:
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None and state.get("scheduler") is not None:
                self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            set_rng_states(state.get("rng", {}))
            log.info("%sResuming %s from epoch %d/%d", self._p(), self.tag, len(self.history) + 1, self.epochs)
        return len(self.history)

    def _p(self) -> str:
        return f"{self.prefix} | " if self.prefix else ""

    # ------------------------------------------------------------------ training
    def _train_epoch(self, epoch: int, run_timer: ProgressTimer) -> Dict[str, float]:
        self.model.train()
        set_loader_epoch(self.train_loader, epoch)
        acc = MetricAccumulator()
        n_batches = len(self.train_loader)
        report_every = max(1, n_batches // 4)
        t0 = time.time()
        for b, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            try:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    logits = self.model(x)
                per_sample, _ = self.loss_fn.per_sample(logits, y)
                loss = per_sample.mean()
                if not torch.isfinite(loss):
                    raise NumericalError(f"{self.tag}: non-finite training loss ({loss.item()}) at epoch {epoch + 1}, "
                                         f"batch {b + 1}/{n_batches}")
                self.scaler.scale(loss).backward()
                if self.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
            except RuntimeError as exc:
                if is_oom_error(exc):
                    raise RuntimeError(f"Out of memory while training {self.tag} (batch size {self.batch_size}). "
                                       f"Reduce --batch-size or the image size.") from exc
                raise
            acc.update(per_sample.detach(), logits.detach().float(), y)
            if (b + 1) % report_every == 0 and b + 1 < n_batches:
                el = time.time() - t0
                log.info("%s  epoch %d/%d | batch %d/%d (%.0f%%) | running loss %.4f | epoch elapsed %s, "
                         "epoch ETA %s", self._p(), epoch + 1, self.epochs, b + 1, n_batches,
                         100.0 * (b + 1) / n_batches, acc.sums["loss"] / acc.n, format_seconds(el),
                         format_seconds(el / (b + 1) * (n_batches - b - 1)))
        return acc.compute()

    def fit(self) -> Dict[str, Any]:
        start = self._try_resume()
        if self.completed:
            log.info("%s%s already completed (%d epochs) — loaded from checkpoint", self._p(), self.tag, len(self.history))
            return self.result()
        timer = ProgressTimer(self.epochs, done_before=start)
        for epoch in range(start, self.epochs):
            t0 = time.time()
            tr = self._train_epoch(epoch, timer)
            va = evaluate(self.model, self.val_loader, self.loss_fn, self.device, self.use_amp)
            for k, v in list(tr.items()) + list(va.items()):
                if v != v or v in (float("inf"), float("-inf")):
                    raise NumericalError(f"{self.tag}: non-finite metric {k}={v} at epoch {epoch + 1}")
            row = {
                "epoch": epoch + 1,
                "train_loss": tr["loss"], "val_loss": va["loss"],
                "train_dice": tr["dice"], "val_dice": va["dice"],
                "train_iou": tr["iou"], "val_iou": va["iou"],
                "train_miou": tr["miou"], "val_miou": va["miou"],
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epoch_seconds": round(time.time() - t0, 2),
            }
            self.history.append(row)
            improved = row["val_dice"] > self.best_val_dice
            if improved:
                self.best_val_dice, self.best_epoch = row["val_dice"], epoch + 1
                save_checkpoint(self.ckpt_dir / "best.pt", {"tag": self.tag, "fingerprint": self.fingerprint,
                                                            "epoch": epoch + 1, "model": self.model.state_dict(),
                                                            "metrics": row})
            stop = self.stopper.should_stop(self.history)
            if stop:
                self.stop_reason = self.stopper.reason
            self.completed = stop or (epoch + 1 == self.epochs)
            self._save_last()
            if self.history_csv is not None:
                save_csv(self.history_csv, self.history, HISTORY_FIELDS)
            timer.step()
            log.info("%sEpoch %d/%d | Progress: %5.1f%% | Train Loss: %.4f | Val Loss: %.4f | Train Dice: %.4f | "
                     "Val Dice: %.4f | Train IoU: %.4f | Val IoU: %.4f | LR: %.2e | epoch %s | Elapsed: %s | ETA: %s%s",
                     self._p(), epoch + 1, self.epochs, timer.percent, row["train_loss"], row["val_loss"],
                     row["train_dice"], row["val_dice"], row["train_iou"], row["val_iou"], row["learning_rate"],
                     format_seconds(row["epoch_seconds"]), format_seconds(timer.elapsed), format_seconds(timer.eta),
                     "  *best*" if improved else "")
            if stop:
                log.info("%sEarly stopping at epoch %d: %s", self._p(), epoch + 1, self.stop_reason)
                break
        self._compact_after_completion()
        return self.result()

    def result(self) -> Dict[str, Any]:
        return {"tag": self.tag, "history": self.history, "best_val_dice": self.best_val_dice,
                "best_epoch": self.best_epoch, "epochs_run": len(self.history),
                "stopped_early": bool(self.stop_reason), "stop_reason": self.stop_reason}
