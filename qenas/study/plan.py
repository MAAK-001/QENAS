"""The experiment plan of evaluation study v2 (STUDY.md).

Study v1 (seeds 42-44, QENAS 2.1, git tag ``study-v1``, results branch ``kaggle-study``) answered the questions
that do not depend on the v2 changes and is not repeated: the CVC-ClinicDB leakage study, the SynFlow analyses,
the objective relations and the proxy validity. Study v2 tests QENAS 3.0 and everything that changed or was never
tested, with new seeds so that the changes are not judged on the runs that motivated them:

RQ1  QENAS 3.0 vs U-Net, Attention U-Net (re-trained under the v2 protocol: 120 epochs, no TTA) and nnU-Net
     (new) on BUSI, CVC-ClinicDB, IDRiD, Kvasir-SEG.
RQ2  Does optimising the loss coefficients help, or does it only cost time? QENAS with the alpha grid vs the same
     architecture trained with fixed alpha = beta = 0.5 (new ablation, all four datasets).
RQ3  Does the new selection (successive halving + end-of-run validation Dice) beat a random candidate and the
     zero-shot rule (max ZiCo)? (BUSI and Kvasir-SEG, as in v1.)
RQ4  Does NSGA-II on (params, ZiCo) still beat random search with the same budget? (training-free, all datasets)
RQ5  External validation Kvasir-SEG -> CVC-ClinicDB for QENAS and the three baselines (inside the Kvasir jobs).

Jobs are ordered seed-major (each seed completes a full set of tables) and, inside a seed, by importance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

STUDY_VERSION = 2
SEEDS = (45, 46, 47)
DATASETS = ("BUSI", "CVC", "IDRID", "KVASIR")
MAIN_METHODS = ("qenas", "unet", "attention_unet", "nnunet")
LOSS_ABLATION = "qenas_fixed_alpha"                          # all four datasets
SELECTION_DATASETS = ("BUSI", "KVASIR")
SELECTION_METHODS = ("qenas_random_pick", "qenas_zico_pick")
EXTERNAL_METHODS = ("qenas", "unet", "attention_unet", "nnunet")   # Kvasir-SEG -> CVC-ClinicDB
DEPENDS_ON_QENAS = (LOSS_ABLATION,) + SELECTION_METHODS

# Duration estimates on one Kaggle T4 with two jobs in parallel (minutes): the v1 measurements scaled to the v2
# protocol (120 final epochs; nnU-Net conservative). Replaced by the median of finished jobs of the same
# (method, dataset) as soon as one exists.
_EST = {
    "qenas": {"BUSI": 95, "CVC": 85, "IDRID": 40, "KVASIR": 120},
    "qenas_fixed_alpha": {"BUSI": 32, "CVC": 27, "IDRID": 12, "KVASIR": 42},
    "unet": {"BUSI": 70, "CVC": 70, "IDRID": 35, "KVASIR": 115},
    "attention_unet": {"BUSI": 40, "CVC": 40, "IDRID": 16, "KVASIR": 65},
    "nnunet": {"BUSI": 150, "CVC": 150, "IDRID": 120, "KVASIR": 150},
    "random_search": {"BUSI": 10, "CVC": 10, "IDRID": 12, "KVASIR": 10},
    "qenas_random_pick": {"BUSI": 55, "KVASIR": 70},
    "qenas_zico_pick": {"BUSI": 55, "IDRID": 15, "KVASIR": 70},
    "abl_random_population": {"IDRID": 40, "KVASIR": 150},
    "abl_params_only": {"IDRID": 40, "KVASIR": 150},
    "abl_random_arch": {"IDRID": 15, "KVASIR": 65},
}

# ---------------------------------------------------------------------------------------- component ablation study
# Six conditions x IDRiD, Kvasir-SEG x seeds 45-47 (36 experiments). Full QENAS, the fixed-alpha variant and the
# Kvasir-SEG max-ZiCo pick are the study-v2 experiments (copied into the results branch); the rest are new. The new
# experiments are split between two Kaggle accounts (workers A and B) of roughly equal GPU time.
ABLATION_DATASETS = ("IDRID", "KVASIR")
ABLATION_METHODS = ("qenas", "abl_random_population", "abl_params_only", "abl_random_arch", "qenas_zico_pick",
                    "qenas_fixed_alpha")
ABLATION_WORKERS = {"A": ("abl_random_population", "qenas_zico_pick"),
                    "B": ("abl_params_only", "abl_random_arch")}


@dataclass(frozen=True)
class Job:
    dataset: str
    method: str
    seed: int
    deps: Tuple[str, ...] = field(default=())

    @property
    def id(self) -> str:
        return f"{self.dataset}/{self.method}/seed_{self.seed}"

    @property
    def external(self) -> bool:
        return self.dataset == "KVASIR" and self.method in EXTERNAL_METHODS

    def estimate_minutes(self) -> float:
        return float(_EST.get(self.method, {}).get(self.dataset, 120))


def plan() -> List[Job]:
    jobs: List[Job] = []
    for s in SEEDS:
        jobs += [Job(d, "qenas", s) for d in DATASETS]
        jobs += [Job(d, LOSS_ABLATION, s, deps=(Job(d, "qenas", s).id,)) for d in DATASETS]
        for m in MAIN_METHODS[1:]:
            jobs += [Job(d, m, s) for d in DATASETS]
        jobs += [Job(d, "random_search", s) for d in DATASETS]
        for d in SELECTION_DATASETS:
            jobs += [Job(d, m, s, deps=(Job(d, "qenas", s).id,)) for m in SELECTION_METHODS]
    ids = [j.id for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    return jobs


def ablation_plan(worker: Optional[str] = None) -> List[Job]:
    """The component ablation study; ``worker`` 'A' or 'B' keeps only that Kaggle account's conditions."""
    jobs: List[Job] = []
    for s in SEEDS:
        for d in ABLATION_DATASETS:
            for m in ABLATION_METHODS:
                inherit = m in DEPENDS_ON_QENAS
                jobs.append(Job(d, m, s, deps=(Job(d, "qenas", s).id,) if inherit else ()))
    if worker is not None:
        keep = ABLATION_WORKERS[worker.upper()]
        jobs = [j for j in jobs if j.method in keep]
    return jobs


def select_plan(name: str = "v2", worker: Optional[str] = None) -> List[Job]:
    if name == "v2":
        if worker is not None:
            raise ValueError("--worker applies to the ablation plan only")
        return plan()
    if name == "ablation":
        return ablation_plan(worker)
    raise ValueError(f"unknown plan {name!r} (v2 or ablation)")


def summary() -> Dict[str, int]:
    jobs = plan()
    out: Dict[str, int] = {}
    for j in jobs:
        out[j.method] = out.get(j.method, 0) + 1
    out["total"] = len(jobs)
    out["external_evaluations"] = sum(j.external for j in jobs)
    out["estimated_gpu_hours"] = round(sum(j.estimate_minutes() for j in jobs) / 60)
    return out
