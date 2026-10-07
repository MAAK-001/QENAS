"""Training-free NSGA-II search with caching, per-generation checkpoints and Pareto tracking.

Uses pymoo's official ask-and-tell interface:

    algorithm.setup(problem, termination, seed)
    while algorithm.has_next():
        pop = algorithm.ask()                         # initial sampling / mating + mutation
        F   = <evaluate params & SynFlow, with cache>  # QENAS evaluation (no training, no data)
        Evaluator().eval(StaticProblem(problem, F=F), pop)
        algorithm.tell(infills=pop)                   # rank-and-crowding survival
        <record generation, checkpoint>

Generation 0 is the random initial population; generations 1..G are evolutionary
generations (pymoo counts the initialisation as its first iteration, hence
``n_gen = G + 1``). The pickled algorithm (including its ``numpy`` Generator) is
checkpointed after every completed generation, so a resumed search continues
bit-identically.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pymoo.core.evaluator import Evaluator
from pymoo.problems.static import StaticProblem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from ..models.chromosome import chromosome_key, chromosome_to_str, decode_short, validate_chromosome
from ..models.network import architecture_signature
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.io import save_csv, save_json
from ..utils.logging_utils import ProgressTimer, format_seconds, get_logger
from .cache import EvaluationCache
from .objectives import ObjectiveError, evaluate_architecture, objective_signature
from .problem import QENASProblem, build_algorithm

log = get_logger()


class NSGA2Search:
    def __init__(self, cfg: Dict[str, Any], exp_dir: Path, device: torch.device, fingerprint: str):
        self.cfg = cfg
        self.exp_dir = Path(exp_dir)
        self.device = device
        self.fingerprint = fingerprint
        self.ckpt_path = self.exp_dir / "checkpoints" / "search_state.pt"
        ctx = {"architecture": architecture_signature(cfg), "objective": objective_signature(cfg)}
        self.cache = EvaluationCache(Path(cfg["search"]["cache_dir"]), ctx)
        self.cache_context = ctx
        self.n_generations = int(cfg["search"]["generations"])
        self.pop_size = int(cfg["search"]["population_size"])
        # state
        self.algorithm = None
        self.generations_done = 0          # number of completed tells (gen 0 = initial population)
        self.archive: Dict[str, Dict[str, Any]] = {}     # key -> evaluation record (all unique individuals)
        self.population_history: List[Dict[str, Any]] = []   # survivors per generation
        self.generation_table: List[Dict[str, Any]] = []
        self.elapsed_before = 0.0

    # ------------------------------------------------------------------ checkpointing
    def _save(self) -> None:
        save_checkpoint(self.ckpt_path, {
            "fingerprint": self.fingerprint,
            "algorithm": self.algorithm,
            "generations_done": self.generations_done,
            "archive": self.archive,
            "population_history": self.population_history,
            "generation_table": self.generation_table,
            "elapsed": self.elapsed_before + (time.time() - self._t_start),
            "cache_context": self.cache_context,
        })

    def _try_resume(self) -> bool:
        state = load_checkpoint(self.ckpt_path)
        if state is None:
            return False
        if state.get("fingerprint") != self.fingerprint:
            raise RuntimeError("Search checkpoint was produced with a different configuration; refusing to resume.")
        self.algorithm = state["algorithm"]
        self.generations_done = int(state["generations_done"])
        self.archive = state["archive"]
        self.population_history = state["population_history"]
        self.generation_table = state["generation_table"]
        self.elapsed_before = float(state.get("elapsed", 0.0))
        log.info("Resumed NSGA-II search after generation %d (%d unique architectures evaluated so far)",
                 self.generations_done - 1, len(self.archive))
        return True

    # ------------------------------------------------------------------ evaluation
    def _evaluate_one(self, chromosome: List[int], generation: int) -> Tuple[Dict[str, Any], bool]:
        key = chromosome_key(chromosome)
        if key in self.archive:
            return self.archive[key], True
        cached = self.cache.get(chromosome)
        if cached is not None:
            rec, was_cached = cached, True
        else:
            try:
                rec = evaluate_architecture(chromosome, self.cfg, self.device)
            except ObjectiveError:
                raise
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(f"Out of memory while evaluating SynFlow for {chromosome}. "
                                       f"Reduce search.synflow.input_size or use --device cpu.") from exc
                raise
            self.cache.put(chromosome, rec)
            was_cached = False
        rec = dict(rec)
        rec["first_generation"] = generation
        rec["blocks"] = decode_short(chromosome)
        self.archive[key] = rec
        return rec, was_cached

    def _evaluate_population(self, X: np.ndarray, generation: int, timer: ProgressTimer) -> np.ndarray:
        F = np.zeros((X.shape[0], 2), dtype=float)
        n = X.shape[0]
        for i, row in enumerate(X):
            ch = validate_chromosome(row)
            rec, cached = self._evaluate_one(ch, generation)
            F[i, 0], F[i, 1] = rec["f1_params"], rec["f2_synflow"]
            if not np.all(np.isfinite(F[i])) or np.any(F[i] <= 0):
                raise ObjectiveError(f"Invalid objective vector {F[i]} for {ch}")
            timer.step()
            log.info("  Generation %d/%d | Evaluating %d/%d | %s %s | params %.3fM | log-SynFlow %.2f | "
                     "f2 %.5f | %s | %s", generation, self.n_generations, i + 1, n, chromosome_to_str(ch),
                     rec["blocks"], rec["params"] / 1e6, rec["synflow_log"], rec["f2_synflow"],
                     "cached" if cached else f"{rec['eval_seconds']:.1f}s", timer.summary())
        return F

    # ------------------------------------------------------------------ recording
    def _record_generation(self, generation: int, gen_seconds: float, n_evaluated: int) -> None:
        pop = self.algorithm.pop
        X = pop.get("X").astype(int)
        F = pop.get("F")
        rank = pop.get("rank")
        crowd = pop.get("crowding")
        rows = []
        for i in range(len(pop)):
            rec = self.archive[chromosome_key(X[i])]
            cd = float(crowd[i]) if crowd[i] is not None else float("nan")
            rows.append({
                "generation": generation, "chromosome": X[i].tolist(), "blocks": rec["blocks"],
                "params": rec["params"], "synflow_raw": rec["synflow_raw"], "synflow_log": rec["synflow_log"],
                "f1_params": float(F[i, 0]), "f2_synflow": float(F[i, 1]),
                "pareto_rank": int(rank[i]), "crowding_distance": cd if np.isfinite(cd) else "inf",
                "nondominated": bool(rank[i] == 0),
            })
        self.population_history.extend(rows)
        front = [r for r in rows if r["nondominated"]]
        self.generation_table.append({
            "generation": generation,
            "population": len(rows),
            "new_evaluations": n_evaluated,
            "pareto_size": len(front),
            "best_params": int(min(r["params"] for r in rows)),
            "best_synflow_objective": float(min(r["f2_synflow"] for r in rows)),
            "best_log_synflow": float(max(r["synflow_log"] for r in rows)),
            "unique_evaluated_total": len(self.archive),
            "seconds": round(gen_seconds, 2),
        })
        log.info("Generation %d/%d done | population %d | Pareto front size: %d | best params %.3fM | "
                 "best log-SynFlow %.2f | time %s", generation, self.n_generations, len(rows), len(front),
                 self.generation_table[-1]["best_params"] / 1e6, self.generation_table[-1]["best_log_synflow"],
                 format_seconds(gen_seconds))

    # ------------------------------------------------------------------ main loop
    def run(self, resume: bool = True, on_generation=None) -> Dict[str, Any]:
        self._t_start = time.time()
        resumed = resume and self._try_resume()
        problem = QENASProblem()
        if not resumed:
            self.algorithm = build_algorithm(self.cfg)
            self.algorithm.setup(problem, termination=("n_gen", self.n_generations + 1),
                                 seed=int(self.cfg["seed"]), verbose=False)
        total_gens = self.n_generations + 1
        timer = ProgressTimer(total_gens * self.pop_size, done_before=self.generations_done * self.pop_size)
        while self.algorithm.has_next():
            generation = self.generations_done
            t0 = time.time()
            log.info("=" * 100)
            log.info("NSGA-II Generation %d/%d %s", generation, self.n_generations,
                     "(initial random population)" if generation == 0 else "")
            pop = self.algorithm.ask()
            X = pop.get("X").astype(int)
            n_before = len(self.archive)
            F = self._evaluate_population(X, generation, timer)
            Evaluator().eval(StaticProblem(problem, F=F), pop)
            self.algorithm.tell(infills=pop)
            self.generations_done += 1
            self._record_generation(generation, time.time() - t0, len(self.archive) - n_before)
            self._save()
            if on_generation is not None:
                on_generation(self)
        return self.finalize()

    def finalize(self) -> Dict[str, Any]:
        pop = self.algorithm.pop
        X = pop.get("X").astype(int)
        F = pop.get("F")
        nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
        front = []
        for i in sorted(nd, key=lambda j: (F[j, 0], F[j, 1])):
            rec = self.archive[chromosome_key(X[i])]
            front.append({k: rec[k] for k in ("chromosome", "blocks", "params", "synflow_raw", "synflow_log",
                                              "f1_params", "f2_synflow")})
        for r, row in enumerate(front, 1):
            row["front_index"] = r
            row["pareto_rank"] = 0
        self.write_outputs(front)
        return {"pareto_front": front, "archive": list(self.archive.values()),
                "population_history": self.population_history, "generation_table": self.generation_table,
                "cache_hits": self.cache.hits, "cache_misses": self.cache.misses}

    def write_outputs(self, front: List[Dict[str, Any]]) -> None:
        d = self.exp_dir
        archive_rows = sorted(self.archive.values(), key=lambda r: (r["first_generation"], r["params"]))
        save_csv(d / "search_results.csv", archive_rows,
                 ["first_generation", "chromosome", "blocks", "params", "synflow_raw", "synflow_log",
                  "f1_params", "f2_synflow", "eval_seconds"])
        save_csv(d / "search_population_history.csv", self.population_history)
        save_csv(d / "nsga2_generations.csv", self.generation_table)
        save_json(d / "nsga2_generations.json", self.generation_table)
        save_csv(d / "pareto_front.csv", front, ["front_index", "chromosome", "blocks", "params", "synflow_raw",
                                                 "synflow_log", "f1_params", "f2_synflow", "pareto_rank"])
        save_json(d / "pareto_front.json", front)
