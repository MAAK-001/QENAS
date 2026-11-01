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
    # journal revision (REVISION.md §4: measured v2 stage times on a T4; the scheduler adds 15 %)
    "supervised_nsga2": {"BUSI": 312, "IDRID": 87},
    "nnunet_full": {"BUSI": 1810, "IDRID": 700},
    "unext": {"BUSI": 37, "CVC": 33, "IDRID": 14, "KVASIR": 62},
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
        return self.dataset == "KVASIR" and self.method in EXTERNAL_METHODS + REVISION_EXTERNAL

    @property
    def spans_sessions(self) -> bool:
        """Longer than a Kaggle session: started regardless of the remaining time, stops cleanly at the deadline
        (exit code 75) and continues in the next session of the same account."""
        from ..methods import method_spec
        return bool(method_spec(self.method).get("spans_sessions"))

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


# ---------------------------------------------------------------------------------------- journal revision
# Three additions requested for the revision (REVISION.md), seeds 45-47, identical splits and protocol:
#   1. Supervised NSGA-II (f2 = 1 - S1) on BUSI and IDRiD;
#   2. nnU-Net with its default 1000-epoch trainer on IDRiD (BUSI optional);
#   3. UNeXt on all four datasets (+ the external Kvasir-SEG -> CVC-ClinicDB test inside the Kvasir-SEG jobs).
# Five Kaggle accounts ("workers" 1-5). A run that spans sessions must stay on one account (its checkpoint lives in
# that account's storage), so each of workers 1-3 takes one seed's nnU-Net run plus that seed's Supervised NSGA-II
# runs on its second GPU; workers 4 and 5 take UNeXt. The optional BUSI nnU-Net runs go to workers 4 and 5.
REVISION_EXTERNAL = ("unext",)
REVISION_WORKERS = {
    "1": [("IDRID", "nnunet_full", 45), ("BUSI", "supervised_nsga2", 45), ("IDRID", "supervised_nsga2", 45)],
    "2": [("IDRID", "nnunet_full", 46), ("BUSI", "supervised_nsga2", 46), ("IDRID", "supervised_nsga2", 46)],
    "3": [("IDRID", "nnunet_full", 47), ("BUSI", "supervised_nsga2", 47), ("IDRID", "supervised_nsga2", 47)],
    "4": [(d, "unext", s) for d in ("BUSI", "CVC", "IDRID") for s in SEEDS],
    "5": [("KVASIR", "unext", s) for s in SEEDS],
}
REVISION_OPTIONAL = {"4": [("BUSI", "nnunet_full", 45)],
                     "5": [("BUSI", "nnunet_full", 46), ("BUSI", "nnunet_full", 47)]}


def revision_plan(worker: Optional[str] = None, optional: bool = False) -> List[Job]:
    """The revision jobs (all workers, or one Kaggle account's share). Jobs that span sessions come first, so that
    they start at the beginning of a session; ``optional`` adds the BUSI nnU-Net (1000 epochs) runs."""
    keys = [str(worker)] if worker is not None else sorted(REVISION_WORKERS)
    jobs: List[Job] = []
    for k in keys:
        if k not in REVISION_WORKERS:
            raise ValueError(f"unknown revision worker {k!r} (1-5)")
        jobs += [Job(d, m, s) for d, m, s in REVISION_WORKERS[k]]
        if optional:
            jobs += [Job(d, m, s) for d, m, s in REVISION_OPTIONAL.get(k, [])]
    jobs.sort(key=lambda j: not j.spans_sessions)          # stable sort: the long runs first
    assert len({j.id for j in jobs}) == len(jobs), "duplicate job ids"
    return jobs


def select_plan(name: str = "v2", worker: Optional[str] = None, optional: bool = False) -> List[Job]:
    if name == "v2":
        if worker is not None:
            raise ValueError("--worker applies to the ablation and revision plans only")
        return plan()
    if name == "ablation":
        return ablation_plan(worker)
    if name == "revision":
        return revision_plan(worker, optional)
    raise ValueError(f"unknown plan {name!r} (v2, ablation or revision)")


def summary() -> Dict[str, int]:
    jobs = plan()
    out: Dict[str, int] = {}
    for j in jobs:
        out[j.method] = out.get(j.method, 0) + 1
    out["total"] = len(jobs)
    out["external_evaluations"] = sum(j.external for j in jobs)
    out["estimated_gpu_hours"] = round(sum(j.estimate_minutes() for j in jobs) / 60)
    return out
