r"""Supervised-fitness NSGA-II: the search baseline "Supervised NSGA-II" of the journal revision (NSGA-Net style).

It is the QENAS search with exactly one change, the second objective:

    f1(a) = number of trainable parameters                                   (unchanged)
    f2(a) = 1 - S1(a),   S1(a) = mean validation Dice over epochs 1-5 of a 5-epoch training of a

S1 is computed with the candidate protocol of successive halving and is therefore *exactly* QENAS' rung-1 score:
the same seeded initialisation theta_0(a, s) (``Pipeline.init_model``), the same ``Trainer`` (AdamW, lr 1e-3, weight
decay 5e-5, constant learning rate, AMP on CUDA), alpha = 0.5, the same batch size, data order and augmentation
stream, ``fit(until=rungs[0])`` and ``criterion(history, criterion_window)``. Everything else - search space,
encoding, pymoo NSGA-II with its operators, population 20, 10 generations (at most 220 distinct evaluations),
duplicate elimination, the evaluation cache, per-generation checkpoints and the hypervolume - is the inherited
``NSGA2Search``. ZiCo is still computed (training-free, cached) and recorded for every architecture, so that every
output file has the QENAS format; it is not optimised.

Resume. Each finished fitness training is written to ``<experiment>/supervised_fitness/<key>.json`` (persistent,
pushed with the results) and its history to ``supervised_search_histories/``. An interrupted fitness training
resumes from its last epoch (``checkpoints/supervised_search/<key>/last.pt``, deleted once the record exists). With
the per-generation search checkpoint, a resumed search re-asks the same offspring (pickled pymoo RNG) and re-uses
every record already written, so it continues where it stopped.

A training with a non-finite loss is a failed evaluation: S1 = 0, f2 = 1 (the worst value), recorded as diverged -
the same consequence as gate G1 in successive halving.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from ..models.chromosome import chromosome_key, decode_short
from ..training.selection import criterion
from ..training.trainer import NumericalError, Trainer
from ..utils.io import load_json, save_json
from ..utils.logging_utils import get_logger
from .nsga2_search import RECORD_KEYS, NSGA2Search
from .objectives import ObjectiveError, evaluate_architecture

log = get_logger()

SUPERVISED_OBJECTIVE = "f2_supervised"
SUPERVISED_KEYS = ("s1_dice", SUPERVISED_OBJECTIVE)


class SupervisedFitness:
    """S1 of one architecture: a 5-epoch candidate-protocol training, stored once per experiment and chromosome."""

    def __init__(self, cfg: Dict[str, Any], exp_dir: Path, device: torch.device, fingerprint: str, data,
                 init_model: Callable[[Dict[str, Any]], torch.nn.Module]):
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        self.device = device
        self.fingerprint = fingerprint
        self.data = data
        self.init_model = init_model
        self.tcfg = cfg["candidate_training"]
        self.epochs = int(self.tcfg["rungs"][0])                     # rung 1 of successive halving (5)
        self.window = int(cfg["selection"]["criterion_window"])     # 5: epochs 1-5
        self.alpha = float(self.tcfg["loss_alpha"])                  # 0.5
        self.store = self.exp_dir / "supervised_fitness"
        self.ckpt_root = self.exp_dir / "checkpoints" / "supervised_search"
        self.histories = self.exp_dir / "supervised_search_histories"
        self.context = {"fingerprint": fingerprint, "epochs": self.epochs, "window": self.window, "alpha": self.alpha,
                        "protocol": "candidate_training (successive-halving rung 1)",
                        "torch_version": torch.__version__.split("+")[0], "device_type": device.type}
        self.trained = 0
        self.reused = 0

    def _path(self, chromosome: Sequence[int]) -> Path:
        return self.store / f"{chromosome_key(chromosome)}.json"

    def evaluate(self, chromosome: Sequence[int]) -> Dict[str, Any]:
        p = self._path(chromosome)
        if p.exists():
            rec = load_json(p)
            if rec.get("context") != self.context:
                diff = {k: (rec.get("context", {}).get(k), v) for k, v in self.context.items()
                        if rec.get("context", {}).get(k) != v}
                raise RuntimeError(f"Supervised-fitness record {p} was computed under a different context {diff}; "
                                   f"refusing to mix evaluations (same software and device type are required).")
            self.reused += 1
            return dict(rec["result"])
        key = chromosome_key(chromosome)
        t0 = time.time()
        model = self.init_model({"type": "qenas", "chromosome": [int(g) for g in chromosome]})
        ckpt = self.ckpt_root / key
        tr = Trainer(model=model, data=self.data, tcfg=self.tcfg, alpha=self.alpha, device=self.device, ckpt_dir=ckpt,
                     seed=int(self.cfg["seed"]), fingerprint=self.fingerprint, tag=f"supervised_fitness_{key}",
                     progress_prefix=f"Supervised fitness {decode_short(chromosome)}",
                     history_csv=self.histories / f"{key}.csv")
        try:
            hist = tr.fit(until=self.epochs)["history"]
            s1, diverged, why = criterion(hist[:self.epochs], self.window), False, ""
        except NumericalError as exc:
            log.error("Supervised fitness of %s diverged: %s -> S1 = 0 (worst fitness)", list(chromosome), exc)
            hist, s1, diverged, why = tr.history, 0.0, True, str(exc)
        # GPU time from the epoch times in the history, so an interrupted-and-resumed training counts every session
        seconds = sum(float(h["epoch_seconds"]) for h in hist) or (time.time() - t0)
        res = {"s1_dice": float(s1), SUPERVISED_OBJECTIVE: 1.0 - float(s1), "fitness_epochs": len(hist),
               "fitness_seconds": round(seconds, 3), "fitness_diverged": diverged, "fitness_divergence": why}
        save_json(p, {"key": key, "chromosome": [int(g) for g in chromosome], "context": self.context, "result": res})
        shutil.rmtree(ckpt, ignore_errors=True)
        del model, tr
        self.trained += 1
        return res


class SupervisedNSGA2Search(NSGA2Search):
    """``NSGA2Search`` whose archive records carry S1 and f2 = 1 - S1 (objectives ``("f1_params", "f2_supervised")``)."""

    record_keys = RECORD_KEYS + SUPERVISED_KEYS

    def __init__(self, cfg: Dict[str, Any], exp_dir: Path, device: torch.device, fingerprint: str,
                 zico_batches: List[Tuple[torch.Tensor, torch.Tensor]], objectives: Sequence[str],
                 fitness: SupervisedFitness):
        if tuple(objectives) != ("f1_params", SUPERVISED_OBJECTIVE):
            raise ValueError(f"Supervised NSGA-II optimises (f1_params, {SUPERVISED_OBJECTIVE}), not {objectives}")
        super().__init__(cfg, exp_dir, device, fingerprint, zico_batches, objectives)
        self.fitness = fitness

    def _evaluate_one(self, chromosome: List[int], generation: int) -> Tuple[Dict[str, Any], bool]:
        key = chromosome_key(chromosome)
        if key in self.archive:
            return self.archive[key], True
        rec = self.cache.get(chromosome)               # ZiCo / params: the QENAS evaluation, recorded only
        if rec is None:
            rec = evaluate_architecture(chromosome, self.cfg, self.device, self.zico_batches)
            self.cache.put(chromosome, rec)
        before = self.fitness.trained
        sup = self.fitness.evaluate(chromosome)
        if not 0.0 < sup[SUPERVISED_OBJECTIVE] <= 1.0:
            raise ObjectiveError(f"f2 = 1 - S1 = {sup[SUPERVISED_OBJECTIVE]} is outside (0, 1] for {chromosome}")
        rec = dict(rec, **sup)
        rec["first_generation"] = generation
        rec["blocks"] = decode_short(chromosome)
        self.archive[key] = rec
        log.info("    S1 (mean val Dice, epochs 1-%d) = %.4f -> f2 = %.4f%s", self.fitness.epochs, sup["s1_dice"],
                 sup[SUPERVISED_OBJECTIVE], "" if self.fitness.trained > before else " (stored record)")
        return rec, self.fitness.trained == before

    def _record_generation(self, generation: int, gen_seconds: float, n_evaluated: int) -> None:
        super()._record_generation(generation, gen_seconds, n_evaluated)
        rows = [r for r in self.population_history if r["generation"] == generation]
        self.generation_table[-1]["best_s1_dice"] = float(max(r["s1_dice"] for r in rows))
        self.generation_table[-1]["fitness_epochs_total"] = int(sum(r.get("fitness_epochs", 0)
                                                                    for r in self.archive.values()))


def search_training_summary(archive: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Training spent by the supervised search (all sessions): trainings, epochs, GPU seconds."""
    return {"fitness_trainings": len(archive),
            "fitness_epochs_total": int(sum(r["fitness_epochs"] for r in archive)),
            "fitness_seconds_total": round(sum(float(r["fitness_seconds"]) for r in archive), 1),
            "fitness_diverged": int(sum(bool(r["fitness_diverged"]) for r in archive)),
            "zico_seconds_total": round(sum(float(r.get("eval_seconds", 0.0)) for r in archive), 1)}


def fitness_store_digest(exp_dir: Path) -> str:
    """SHA-256 over all stored fitness records (reproducibility record)."""
    h = hashlib.sha256()
    for p in sorted((Path(exp_dir) / "supervised_fitness").glob("*.json")):
        h.update(json.dumps(load_json(p)["result"], sort_keys=True).encode())
    return h.hexdigest()[:24]
