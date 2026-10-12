"""The experiment plan of the QENAS evaluation study (STUDY.md).

Research questions and the experiments that answer them (seeds 42, 43, 44; identical split for every run):

RQ1  How does QENAS compare with hand-designed networks trained on the same splits?
     QENAS, U-Net, Attention U-Net on BUSI, CVC-ClinicDB, IDRiD, Kvasir-SEG.
RQ2  What does each component contribute? (ablations on BUSI and Kvasir-SEG)
     without ZiCo, without SynFlow, random Pareto candidate, zero-shot pick (max ZiCo).
RQ3  Does NSGA-II search better than random sampling with the same evaluation budget? (training-free)
     random search on all four datasets (also an unbiased sample for the objective analyses).
RQ4  Do the models generalise to another hospital? Kvasir-SEG-trained QENAS / U-Net / Attention U-Net tested
     on all of CVC-ClinicDB (external validation, run inside the Kvasir-SEG jobs).
RQ5  How much does a random frame split inflate CVC-ClinicDB scores? QENAS and U-Net with a frame-level split.

Jobs are ordered seed-major (each seed completes a full set of tables) and, inside a seed, by importance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

SEEDS = (42, 43, 44)
DATASETS = ("BUSI", "CVC", "IDRID", "KVASIR")
MAIN_METHODS = ("qenas", "unet", "attention_unet")
ABLATION_DATASETS = ("BUSI", "KVASIR")
ABLATION_METHODS = ("qenas_no_zico", "qenas_no_synflow", "qenas_random_pick", "qenas_zico_pick")
EXTERNAL_METHODS = ("qenas", "unet", "attention_unet")      # Kvasir-SEG -> CVC-ClinicDB
LEAKAGE_METHODS = ("qenas", "unet")                         # CVC-ClinicDB with a frame-level split

# Conservative duration estimates on one Kaggle T4 with two jobs in parallel (minutes); replaced by the median of
# finished jobs of the same (method, dataset) as soon as one exists.
_EST = {
    "qenas": {"BUSI": 125, "CVC": 85, "IDRID": 45, "KVASIR": 200},
    "unet": {"BUSI": 110, "CVC": 95, "IDRID": 35, "KVASIR": 180},
    "attention_unet": {"BUSI": 100, "CVC": 85, "IDRID": 30, "KVASIR": 160},
    "random_search": {"BUSI": 20, "CVC": 20, "IDRID": 30, "KVASIR": 20},
    "qenas_random_pick": {"BUSI": 70, "KVASIR": 115},
    "qenas_zico_pick": {"BUSI": 70, "KVASIR": 115},
}
_EST["qenas_no_zico"] = _EST["qenas"]
_EST["qenas_no_synflow"] = _EST["qenas"]


@dataclass(frozen=True)
class Job:
    dataset: str
    method: str
    seed: int
    split_unit: str = "dataset"
    deps: Tuple[str, ...] = field(default=())

    @property
    def id(self) -> str:
        ds = f"{self.dataset}_frame_split" if self.split_unit == "image" else self.dataset
        return f"{ds}/{self.method}/seed_{self.seed}"

    @property
    def external(self) -> bool:
        return self.dataset == "KVASIR" and self.split_unit == "dataset" and self.method in EXTERNAL_METHODS

    def estimate_minutes(self) -> float:
        return float(_EST[self.method].get(self.dataset, 120))


def plan() -> List[Job]:
    jobs: List[Job] = []
    for s in SEEDS:
        for m in MAIN_METHODS:
            jobs += [Job(d, m, s) for d in DATASETS]
        jobs += [Job(d, "random_search", s) for d in DATASETS]
        for d in ABLATION_DATASETS:
            parent = Job(d, "qenas", s).id
            jobs += [Job(d, "qenas_no_zico", s), Job(d, "qenas_no_synflow", s),
                     Job(d, "qenas_random_pick", s, deps=(parent,)), Job(d, "qenas_zico_pick", s, deps=(parent,))]
        jobs += [Job("CVC", m, s, split_unit="image") for m in LEAKAGE_METHODS]
    ids = [j.id for j in jobs]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    return jobs


def summary() -> Dict[str, int]:
    jobs = plan()
    out: Dict[str, int] = {}
    for j in jobs:
        out[j.method] = out.get(j.method, 0) + 1
    out["total"] = len(jobs)
    out["external_evaluations"] = sum(j.external for j in jobs)
    return out
