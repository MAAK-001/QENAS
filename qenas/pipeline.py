"""The QENAS pipeline and the study conditions that share its code.

QENAS (``method = "qenas"``, the only method of ``main.py``) is one fixed sequence of stages:

    split -> search (NSGA-II, training-free: params, ZiCo) -> pareto (16 candidates of the final population)
          -> candidates (successive halving: 16 x 5 -> 8 x 10 -> 4 x 20 epochs) -> selection (learning capability)
          -> loss_weights (alpha on validation) -> final (supervised training with validation; best checkpoint)
          -> test (once) -> report

The comparison conditions of the evaluation study (``qenas/methods.py``) run a subset of the *same* stage code:
the selection and loss-weight ablations re-use QENAS' search and candidate runs and change only one rule,
baselines skip the search and train a fixed network with QENAS' own protocol, random search only replaces NSGA-II.
nnU-Net has its own stages (``qenas/study/nnunet.py``).

Every stage is idempotent and resumable; completed stages are skipped on re-run.
The test split is first materialised in ``stage_test`` and nowhere else.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .datasets.dataset import DataBundle
from .datasets.discovery import DatasetError, Sample, discover, rebase_path, relative_to_root
from .datasets.splits import check_no_leakage, make_splits, split_summary
from .experiment import Experiment, experiment_dir
from .methods import method_spec, stages_for
from .models.baselines import build_baseline
from .models.chromosome import SEARCH_SPACE_SIZE, chromosome_key, chromosome_to_str, decode
from .models.network import QENASNet, build_model, count_parameters
from .models.summary import cost_summary, model_summary, summary_text
from .search.nsga2_search import NSGA2Search, RandomSearch
from .search.objectives import ZICO_BATCHES, ZICO_LOSS_ALPHA, zico_batches_digest
from .search.pareto import select_pareto_candidates
from .training.losses import BCESoftMIoULoss
from .training.selection import criterion, promote, rung_sizes, select_by_learning_capability
from .training.trainer import NumericalError, Trainer, evaluate, inference_time_ms, load_best_weights
from .utils.env import environment_info
from .utils.io import load_json, save_csv, save_json, save_markdown_table
from .utils.logging_utils import format_seconds, get_logger
from .utils.seed import derive_seed
from .visualization import plots
from .visualization.architecture import plot_architecture

log = get_logger()

STAGES = stages_for("qenas")   # the QENAS method
OBJ_COLS = ["params", "zico", "f1_params", "f2_zico"]
# Files of the parent QENAS experiment re-used by the inheriting conditions (stage "inherit").
INHERITED_FILES = ["search_results.csv", "search_results.json", "search_population_history.csv",
                   "search_population_history.json", "nsga2_generations.csv", "nsga2_generations.json",
                   "search_summary.json", "pareto_front.csv", "pareto_front.json", "pareto_candidates.csv",
                   "pareto_candidates.json", "candidate_results.json", "candidate_training.csv",
                   "successive_halving.json"]


class Pipeline:
    def __init__(self, cfg: Dict[str, Any], exp: Experiment, device: torch.device):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.fp = exp.fingerprint
        self.seed = int(cfg["seed"])
        self.method_name = cfg["method"]
        self.method = method_spec(self.method_name)
        self.stages = stages_for(self.method_name)
        self._data: Optional[DataBundle] = None
        self._splits: Optional[Dict[str, List[Sample]]] = None

    # ================================================================== helpers
    def path(self, name: str) -> Path:
        return self.exp.dir / name

    @property
    def objectives(self):
        return self.method["objectives"]

    def init_model(self, arch: Dict[str, Any]) -> torch.nn.Module:
        """Seeded initialisation; for QENAS architectures identical to the zero-cost evaluation (the scored
        network is the trained network)."""
        devs = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(self.seed)
            return self.build(arch)

    def build(self, arch: Dict[str, Any]) -> torch.nn.Module:
        if arch["type"] == "qenas":
            return build_model(arch["chromosome"], self.cfg)
        return build_baseline(arch["network"], int(self.cfg["data"]["in_channels"]),
                              int(self.cfg["model"]["num_classes"]))

    @property
    def splits(self) -> Dict[str, List[Sample]]:
        if self._splits is None:
            d = load_json(self.path("split.json"))
            self._splits = {k: [Sample.from_dict(s) for s in v] for k, v in d["splits"].items()}
            # Paths are re-based onto the *current* data root, so an experiment created on one machine
            # (e.g. Windows) can be resumed on another (e.g. Kaggle). Sample IDs and split membership
            # are unchanged; only file locations are resolved anew.
            root, ds, old_root = self.cfg["data"]["root"], self.cfg["data"]["dataset"], d.get("data_root")
            for lst in self._splits.values():
                for s in lst:
                    s.image = rebase_path(s.image, root, ds, old_root)
                    s.masks = [rebase_path(m, root, ds, old_root) for m in s.masks]
                    for p in [s.image] + s.masks:
                        if not Path(p).exists():
                            raise DatasetError(
                                f"File listed in split.json not found under the data root '{root}': {p}\n"
                                f"Pass --data-root pointing to the folder that contains the {ds} dataset folder.")
            check_no_leakage(self._splits)
        return self._splits

    @property
    def data(self) -> DataBundle:
        if self._data is None:
            self._data = DataBundle(self.splits, self.cfg)
            log.info("Data ready: train %d, validation %d (test held out) | normalisation mean %s std %s",
                     len(self.splits["train"]), len(self.splits["val"]),
                     [round(x, 4) for x in self._data.mean], [round(x, 4) for x in self._data.std])
        return self._data

    def parent_dir(self) -> Path:
        """Directory of the QENAS experiment (same dataset, seed, split) an inheriting condition builds on."""
        pcfg = dict(self.cfg)
        pcfg["method"] = self.method["parent"]
        return experiment_dir(pcfg)

    # ================================================================== driver
    def run(self, stop_after: Optional[str] = None) -> None:
        if stop_after is not None and stop_after not in self.stages:
            raise ValueError(f"--stop-after must be one of {self.stages}")
        log.info("Method %s (%s) | stages: %s", self.method_name, self.method["doc"], " -> ".join(self.stages))
        for stage in self.stages:
            if self.exp.is_done(stage):
                log.info("Stage '%s' already completed — skipping", stage)
            else:
                log.info("#" * 100)
                log.info("STAGE: %s", stage.upper())
                t0 = time.time()
                info = getattr(self, f"stage_{stage}")() or {}
                info["seconds"] = round(time.time() - t0, 1)
                self.exp.mark_done(stage, info)
                self._prune(stage)
                log.info("Stage '%s' finished in %s", stage, format_seconds(info["seconds"]))
            if stop_after == stage:
                log.info("Stopping after stage '%s' as requested (resume later with the same command).", stage)
                return
        self.exp.mark_completed()
        self._prune("completed")
        if self.path("summary.txt").exists():
            print(self.path("summary.txt").read_text(encoding="utf-8"))

    def _prune(self, stage: str) -> None:
        """Delete checkpoints that no completed stage will ever read again (their results are in the stage's
        JSON/CSV outputs). A finished experiment keeps only its final best-validation weights."""
        ck = self.exp.checkpoints
        targets = {"search": [ck / "search_state.pt", ck / "search_state.pt.prev"],
                   "candidates": [ck / "candidates"], "loss_weights": [ck / "loss_weights"]}.get(stage, [])
        if stage == "completed":
            targets = ([p for p in ck.iterdir() if p.name != "final"] +
                       [p for p in (ck / "final").glob("*") if p.name != "best.pt"]) if ck.exists() else []
        for p in targets:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()

    # ================================================================== 1. split
    def stage_split(self) -> Dict[str, Any]:
        disc = discover(self.cfg)
        splits = make_splits(disc, self.cfg)
        summ = split_summary(splits)
        save_json(self.path("dataset_report.json"), disc.report)
        root = self.cfg["data"]["root"]

        def portable(s: Sample) -> Dict[str, Any]:
            d = s.to_dict()
            d["image"] = relative_to_root(s.image, root)
            d["masks"] = [relative_to_root(m, root) for m in s.masks]
            return d

        save_json(self.path("split.json"), {
            "dataset": disc.dataset, "split_seed": int(self.cfg["data"]["split_seed"]),
            "summary": summ, "data_root": root,
            "paths": "relative to data_root (re-based onto --data-root when resumed elsewhere)",
            "splits": {k: [portable(s) for s in v] for k, v in splits.items()},
        })
        self._splits = splits
        log.info("Dataset structure: %s", disc.report["layout"])
        log.info("\n%s\nTrain: %d\nValidation: %d\nTest: %d", disc.dataset, summ["train"]["n"], summ["val"]["n"],
                 summ["test"]["n"])
        return {"train": summ["train"]["n"], "val": summ["val"]["n"], "test": summ["test"]["n"]}

    # ================================================================== 2. search
    def stage_search(self) -> Dict[str, Any]:
        s = self.cfg["search"]
        bs = int(self.cfg["candidate_training"]["batch_size"])
        random_search = self.method.get("sampler") == "random"
        log.info("Training-free %s | search space 5^8 = %s | population %d | generations %d | "
                 "crossover p=%.2f | mutation p_var=%.3f | optimised objectives: %s",
                 "random search" if random_search else "NSGA-II", f"{SEARCH_SPACE_SIZE:,}", s["population_size"],
                 s["generations"], s["crossover_prob"], s["mutation_prob_var"], ", ".join(self.objectives))
        zb = self.data.zico_batches(bs, ZICO_BATCHES, self.seed)
        digest = zico_batches_digest(zb)
        log.info("ZiCo data: the first %d augmented training batches of epoch 0 (batch size %d, sha256 %s); "
                 "loss alpha=%.2f; validation and test images are not used by the search", ZICO_BATCHES, bs, digest,
                 ZICO_LOSS_ALPHA)
        if random_search:
            res = RandomSearch(self.cfg, self.exp.dir, self.device, self.fp, zb, self.objectives).run()
            hv = res["generation_table"][0]["hypervolume"]
            info = {"unique_evaluations": len(res["archive"]), "final_front_size": len(res["pareto_front"]),
                    "cache_hits": res["cache_hits"], "cache_misses": res["cache_misses"],
                    "objectives": list(self.objectives), "sampler": "random",
                    "hypervolume_generation_0": hv, "hypervolume_final": hv, "zico_batches_sha256": digest}
            save_json(self.path("search_summary.json"), info)
            return {k: v for k, v in info.items() if k != "zico_batches_sha256"}
        search = NSGA2Search(self.cfg, self.exp.dir, self.device, self.fp, zb, self.objectives)
        every = max(1, int(s["plot_every"]))

        def on_generation(srch: NSGA2Search) -> None:
            g = srch.generations_done - 1
            if g == 0 or g % every == 0:
                rows = [r for r in srch.population_history if r["generation"] == g]
                arch = [r for r in srch.archive.values() if r["first_generation"] <= g]
                name = "pareto_initial_population" if g == 0 else f"pareto_generation_{g:03d}"
                plots.plot_pareto_population(rows, arch, g, srch.n_generations, self.exp.figures / name)

        res = search.run(on_generation=on_generation)
        hist, table = res["population_history"], res["generation_table"]
        last_gen = max(r["generation"] for r in hist)
        plots.plot_pareto_population([r for r in hist if r["generation"] == last_gen], res["archive"], last_gen,
                                     search.n_generations, self.exp.figures / "pareto_front_final",
                                     title=f"Final population and Pareto front (generation {last_gen})")
        plots.plot_search_progress(hist, table, self.exp.figures / "search_progress")
        info = {"unique_evaluations": len(res["archive"]), "final_front_size": len(res["pareto_front"]),
                "cache_hits": res["cache_hits"], "cache_misses": res["cache_misses"],
                "objectives": list(self.objectives),
                "hypervolume_generation_0": table[0]["hypervolume"], "hypervolume_final": table[-1]["hypervolume"],
                "zico_batches_sha256": digest}
        save_json(self.path("search_summary.json"), info)
        log.info("Search complete: %d unique architectures evaluated, final Pareto front size %d, hypervolume "
                 "%.4f (generation 0) -> %.4f (generation %d)", info["unique_evaluations"], info["final_front_size"],
                 info["hypervolume_generation_0"], info["hypervolume_final"], last_gen)
        return {k: v for k, v in info.items() if k != "zico_batches_sha256"}

    # ================================================================== 3. candidates of the final population
    def stage_pareto(self) -> Dict[str, Any]:
        front = load_json(self.path("pareto_front.json"))
        hist = load_json(self.path("search_population_history.json"))
        last_gen = max(r["generation"] for r in hist)
        population = [r for r in hist if r["generation"] == last_gen]
        k = int(self.cfg["pareto"]["max_candidates"])
        cands = select_pareto_candidates(front, k, self.objectives, population)
        for i, c in enumerate(cands, 1):
            c["candidate_id"] = i
        save_json(self.path("pareto_candidates.json"), cands)
        save_csv(self.path("pareto_candidates.csv"), cands, ["candidate_id", "chromosome", "blocks", *OBJ_COLS,
                                                             "selection_reason"])
        archive = load_json(self.path("search_results.json"))
        plots.plot_pareto_population(population, archive, last_gen, int(self.cfg["search"]["generations"]),
                                     self.exp.figures / "pareto_front_final_with_candidates", highlight=cands,
                                     title="Final population and the candidates trained by successive halving")
        n_front = sum(1 for c in cands if c["selection_reason"].startswith(("Pareto front", "extreme")))
        log.info("Final Pareto front: %d architectures; %d candidates for successive halving (%d from the front%s)",
                 len(front), len(cands), n_front,
                 f", {len(cands) - n_front} next-ranked individuals of the final population" if n_front < len(cands)
                 else "")
        for c in cands:
            log.info("  C%d %s %s | %.3fM params | ZiCo %.2f | %s", c["candidate_id"], chromosome_to_str(c["chromosome"]),
                     c["blocks"], c["params"] / 1e6, c["zico"], c["selection_reason"])
        return {"front_size": len(front), "n_candidates": len(cands), "from_front": n_front}

    # ================================================================== 2'. inherit (ablations)
    def stage_inherit(self) -> Dict[str, Any]:
        """Re-use the search, the candidates and the successive-halving runs of the parent QENAS experiment (same
        dataset, seed and split), so that the condition differs from QENAS only in its documented rule."""
        src = self.parent_dir()
        st = load_json(src / "state.json") if (src / "state.json").exists() else {}
        if st.get("stages", {}).get("candidates", {}).get("status") != "done":
            raise RuntimeError(f"Parent experiment {src} has not finished its candidate stage yet.")
        if load_json(src / "split.json")["splits"] != load_json(self.path("split.json"))["splits"]:
            raise RuntimeError(f"Split of {src} differs from this experiment's split.")
        for f in INHERITED_FILES:
            shutil.copy2(src / f, self.path(f))
        shutil.copytree(src / "candidate_histories", self.path("candidate_histories"), dirs_exist_ok=True)
        log.info("Inherited search, %d candidates and their successive-halving runs from %s",
                 len(load_json(self.path("candidate_results.json"))), src)
        return {"inherited_from": str(src)}

    # ================================================================== 4. candidate training (successive halving)
    def stage_candidates(self) -> Dict[str, Any]:
        cands = load_json(self.path("pareto_candidates.json"))
        tcfg = self.cfg["candidate_training"]
        rungs = [int(r) for r in tcfg["rungs"]]
        w = int(self.cfg["selection"]["criterion_window"])
        sizes = rung_sizes(len(cands), len(rungs), float(tcfg["keep_fraction"]))
        # Resume: histories saved after every candidate run (the checkpoints of eliminated runs are deleted).
        prev = ({r["candidate_id"]: r for r in load_json(self.path("candidate_results.json"))}
                if self.path("candidate_results.json").exists() else {})
        recs = {c["candidate_id"]: dict(c, history=prev.get(c["candidate_id"], {}).get("history") or [],
                                        diverged=bool(prev.get(c["candidate_id"], {}).get("diverged")),
                                        divergence=prev.get(c["candidate_id"], {}).get("divergence", ""),
                                        rung_scores={}) for c in cands}
        sh_rungs: List[Dict[str, Any]] = []
        active = [c["candidate_id"] for c in cands]
        t_all = time.time()
        log.info("Successive halving: %s candidates trained to %s epochs (criterion: mean validation Dice of the "
                 "last %d epochs; budget %d candidate-epochs)", " -> ".join(map(str, sizes)),
                 " -> ".join(map(str, rungs)), w,
                 sum(n * (r - (rungs[i - 1] if i else 0)) for i, (n, r) in enumerate(zip(sizes, rungs))))
        for ri, rung in enumerate(rungs):
            log.info("=" * 100)
            log.info("Rung %d/%d: %d candidates to %d epochs", ri + 1, len(rungs), len(active), rung)
            for j, cid in enumerate(active, 1):
                rec = recs[cid]
                self._train_candidate(rec, rung, f"Rung {ri + 1}/{len(rungs)} | candidate {j}/{len(active)}")
                if not rec["diverged"]:
                    rec["rung_scores"][str(rung)] = criterion(rec["history"][:rung], w)
                    log.info("  C%d after %d epochs: mean val Dice of the last %d epochs = %.4f", cid, rung, w,
                             rec["rung_scores"][str(rung)])
                self._save_candidates(list(recs.values()))
            alive = [recs[c] for c in active if not recs[c]["diverged"]]
            scored = [{"candidate_id": c["candidate_id"], "score": c["rung_scores"][str(rung)], "params": c["params"]}
                      for c in alive]
            nxt = promote(scored, sizes[ri + 1]) if ri + 1 < len(rungs) else []
            sh_rungs.append({"rung": ri + 1, "epochs": rung, "trained": active,
                             "diverged": [c for c in active if recs[c]["diverged"]],
                             "scores": {str(c["candidate_id"]): c["score"] for c in scored}, "promoted": nxt})
            if nxt:
                log.info("Promoted to %d epochs: %s", rungs[ri + 1], ", ".join(f"C{c}" for c in nxt))
                for c in active:              # eliminated runs: their histories are saved, the weights are not needed
                    if c not in nxt:
                        shutil.rmtree(self.exp.checkpoints / "candidates" / f"cand_{c:02d}", ignore_errors=True)
            active = nxt
        results = [recs[c["candidate_id"]] for c in cands]
        for r in results:
            r["rung_reached"] = sum(1 for x in sh_rungs if r["candidate_id"] in x["trained"])
        self._save_candidates(results)
        save_json(self.path("successive_halving.json"), {"sizes": sizes, "rung_epochs": rungs, "window": w,
                                                         "keep_fraction": float(tcfg["keep_fraction"]),
                                                         "rungs": sh_rungs})
        log.info("Successive halving finished in %s", format_seconds(time.time() - t_all))
        return {"n_candidates": len(cands), "rung_sizes": sizes,
                "n_diverged": sum(1 for r in results if r["diverged"])}

    def _train_candidate(self, rec: Dict[str, Any], until: int, prefix: str) -> None:
        cid = rec["candidate_id"]
        ckpt = self.exp.checkpoints / "candidates" / f"cand_{cid:02d}"
        div_file = ckpt / "diverged.json"
        if rec["diverged"] or len(rec["history"]) >= until:
            return
        if div_file.exists():
            rec.update(load_json(div_file))
            log.warning("Candidate %d previously diverged: %s", cid, rec.get("divergence"))
            return
        log.info("%s | C%d %s | %s | %.3fM params | ZiCo %.2f", prefix, cid, chromosome_to_str(rec["chromosome"]),
                 rec["blocks"], rec["params"] / 1e6, rec["zico"])
        model = self.init_model({"type": "qenas", "chromosome": rec["chromosome"]})
        assert count_parameters(model) == rec["params"], "parameter count mismatch between search and training"
        tcfg = self.cfg["candidate_training"]
        trainer = Trainer(model=model, data=self.data, tcfg=tcfg, alpha=float(tcfg["loss_alpha"]), device=self.device,
                          ckpt_dir=ckpt, seed=self.seed, fingerprint=self.fp,
                          tag=f"candidate_{cid}_{chromosome_key(rec['chromosome'])}", progress_prefix=prefix,
                          history_csv=self.exp.dir / "candidate_histories" / f"candidate_{cid:02d}.csv")
        try:
            rec["history"] = trainer.fit(until=until)["history"]
        except NumericalError as exc:
            log.error("Candidate %d diverged: %s — it is eliminated (gate G1)", cid, exc)
            info = {"diverged": True, "divergence": str(exc), "history": trainer.history}
            save_json(div_file, info)
            rec.update(info)

    def _save_candidates(self, results: List[Dict[str, Any]]) -> None:
        save_json(self.path("candidate_results.json"), results)
        rows = []
        for r in results:
            for h in r.get("history") or []:
                rows.append({"candidate_id": r["candidate_id"], "chromosome": r["chromosome"], "blocks": r["blocks"],
                             **{k: r[k] for k in OBJ_COLS}, **h})
        save_csv(self.path("candidate_training.csv"), rows)

    # ================================================================== 5. selection
    def stage_selection(self) -> Dict[str, Any]:
        results = load_json(self.path("candidate_results.json"))
        sc = self.cfg["selection"]
        w = int(sc["criterion_window"])
        E = int(self.cfg["candidate_training"]["epochs"])
        sel = select_by_learning_capability(results, self.data.val_trivial_dice, float(sc["significance"]), w, E)
        rule = self.method["selection"]
        sel["selection_rule"] = rule
        if rule != "learning_capability":
            # Selection ablations: same candidates and the same learning-capability analysis (kept for reporting),
            # but the final architecture is chosen by another rule from all 16 candidates.
            sel["learning_capability_choice"] = sel["selected_id"]
            pool = [a for a in sel["analysis"] if not a["diverged"]]
            if rule == "random":
                ds = self.cfg["data"]["dataset"]
                rng = np.random.default_rng(derive_seed(self.seed, f"random-candidate-{ds}"))
                pick = pool[int(rng.integers(len(pool)))]
                why = f"uniformly random candidate (rng seed derived from seed {self.seed} and dataset {ds})"
            elif rule == "max_zico":
                pick = sorted(pool, key=lambda a: (-a["zico"], a["params"]))[0]
                why = "highest ZiCo among the candidates (zero-shot rule, no training used)"
            else:
                raise ValueError(f"unknown selection rule {rule}")
            sel["selected_id"] = pick["candidate_id"]
            for a in sel["analysis"]:
                a["selected"] = a["candidate_id"] == pick["candidate_id"]
            sel["reason"] = (f"Candidate {pick['candidate_id']} selected: {why}. Learning-capability selection would "
                             f"have chosen candidate {sel['learning_capability_choice']}.")
        save_json(self.path("selection.json"), sel)
        rows = []
        for a in sorted(sel["analysis"], key=lambda a: a["candidate_id"]):
            g = a["gates"]
            rows.append({
                "Candidate": f"C{a['candidate_id']}", "Chromosome": chromosome_to_str(a["chromosome"]),
                "Blocks": a["blocks"], "Params": a["params"], "ZiCo": a["zico"],
                **{f"Score {e} ep": a["rung_scores"].get(str(e)) for e in self.cfg["candidate_training"]["rungs"]},
                "Epochs trained": a.get("epochs_trained"), "ΔDice": a.get("delta_dice"), "ΔLoss": a.get("delta_loss"),
                "Finalist": a["finalist"],
                "Gates": ("–" if not a["finalist"] else "all passed" if all(g.get(k) for k in g) else
                          "failed " + ", ".join(k.split("_")[0] for k in sorted(g) if not g.get(k))),
                "Eligible": a.get("eligible"), "Capability rank": a.get("capability_rank"), "Selected": a["selected"],
            })
        save_csv(self.path("candidate_comparison.csv"), rows)
        save_json(self.path("candidate_comparison.json"), rows)
        cols = ["Candidate", "Params", "ZiCo"] + [f"Score {e} ep" for e in self.cfg["candidate_training"]["rungs"]] + \
               ["Epochs trained", "Finalist", "Gates", "Selected"]
        save_markdown_table(self.path("candidate_comparison.md"), rows, cols,
                            title="Successive halving and learning-capability selection (validation set)")
        f = self.exp.figures
        results_by_id = {r["candidate_id"]: r for r in results}
        cl = [results_by_id[k] for k in sorted(results_by_id)]
        plots.plot_candidate_curves(cl, "val_dice", "Validation Dice", "Candidates — validation Dice",
                                    f / "candidates_val_dice", sel["selected_id"])
        plots.plot_candidate_curves(cl, "val_loss", "Validation loss (α·BCE + β·soft-mIoU)",
                                    "Candidates — validation loss", f / "candidates_val_loss", sel["selected_id"])
        plots.plot_successive_halving(load_json(self.path("successive_halving.json")), sel["analysis"],
                                      f / "successive_halving")
        first = str(self.cfg["candidate_training"]["rungs"][0])
        plots.plot_scatter_vs_dice(sel["analysis"], "params", "Trainable parameters (millions)",
                                   "Complexity vs early validation quality", f / "params_vs_val_dice", first)
        plots.plot_scatter_vs_dice(sel["analysis"], "zico", "ZiCo", "ZiCo vs early validation quality",
                                   f / "zico_vs_val_dice", first)
        plots.plot_capability_ranking(sel, w, f / "learning_capability_ranking")
        log.info(sel["reason"])
        return {"selected_candidate": sel["selected_id"], "rule": rule, "relaxed_gates": sel["relaxed_gates"]}

    def selected(self) -> Dict[str, Any]:
        """The architecture trained by loss_weights/final/test: the selected QENAS candidate or the baseline."""
        if self.method["kind"] == "baseline":
            return {"arch": {"type": "baseline", "network": self.method["network"]}, "label": self.method["label"],
                    "history": None, "diverged": False}
        if self.method["kind"] == "random_arch":
            # One architecture drawn uniformly from the 5^8 search space; deterministic in (seed, dataset).
            from .models.chromosome import CHROMOSOME_LENGTH, N_BLOCK_TYPES
            ds = self.cfg["data"]["dataset"]
            rng = np.random.default_rng(derive_seed(self.seed, f"random-architecture-{ds}"))
            ch = [int(g) for g in rng.integers(0, N_BLOCK_TYPES, CHROMOSOME_LENGTH)]
            return {"arch": {"type": "qenas", "chromosome": ch}, "label": chromosome_to_str(ch), "history": None,
                    "diverged": False, "chromosome": ch,
                    "reason": f"uniformly random architecture (rng seed derived from seed {self.seed} and dataset {ds})"}
        sel = load_json(self.path("selection.json"))
        results = load_json(self.path("candidate_results.json"))
        rec = next(r for r in results if r["candidate_id"] == sel["selected_id"])
        ana = next(a for a in sel["analysis"] if a["candidate_id"] == sel["selected_id"])
        return {**rec, "arch": {"type": "qenas", "chromosome": rec["chromosome"]},
                "label": chromosome_to_str(rec["chromosome"]), "analysis": ana, "reason": sel["reason"]}

    def arch_key(self, best: Dict[str, Any]) -> str:
        a = best["arch"]
        return chromosome_key(a["chromosome"]) if a["type"] == "qenas" else a["network"]

    # ================================================================== 6. loss weights
    def stage_loss_weights(self) -> Dict[str, Any]:
        lw = self.cfg["loss_weights"]
        tcfg = self.cfg["candidate_training"]
        cand_alpha = float(tcfg["loss_alpha"])
        best = self.selected()
        w = int(lw["criterion_window"])
        rows = []
        alphas = sorted({round(float(a), 6) for a in lw["alphas"]})
        for i, a in enumerate(alphas, 1):
            log.info("=" * 100)
            log.info("Loss-weight run %d/%d | alpha (BCE) = %.3f, beta (soft-mIoU) = %.3f | architecture %s",
                     i, len(alphas), a, 1 - a, best["label"])
            hist = best.get("history") or []
            if abs(a - cand_alpha) < 1e-12 and len(hist) == int(tcfg["epochs"]) and not best.get("diverged"):
                log.info("  identical configuration to the candidate run (same seed, data order, alpha, epochs) — "
                         "reusing its %d-epoch history", len(hist))
                status = "reused candidate run"
            else:
                model = self.init_model(best["arch"])
                tr = Trainer(model=model, data=self.data, tcfg=tcfg, alpha=a, device=self.device,
                             ckpt_dir=self.exp.checkpoints / "loss_weights" / f"alpha_{a:.3f}", seed=self.seed,
                             fingerprint=self.fp, tag=f"alpha_{a:.6f}_{self.arch_key(best)}",
                             progress_prefix=f"Loss-weight run {i}/{len(alphas)} (alpha={a:.2f})",
                             history_csv=self.exp.dir / "loss_weight_histories" / f"alpha_{a:.3f}.csv")
                try:
                    hist = tr.fit()["history"]
                    status = "trained"
                except NumericalError as exc:
                    log.error("alpha=%.3f diverged: %s", a, exc)
                    rows.append({"alpha": a, "beta": 1 - a, "criterion": None, "status": f"diverged: {exc}"})
                    continue
            crit = criterion(hist, w)
            rows.append({"alpha": a, "beta": round(1 - a, 6), "criterion": crit,
                         "best_val_dice": max(h["val_dice"] for h in hist), "final_val_dice": hist[-1]["val_dice"],
                         "status": status})
            log.info("  alpha=%.3f -> mean val Dice over last %d epochs = %.4f", a, w, crit)
        ok = [r for r in rows if r["criterion"] is not None]
        if not ok:
            raise RuntimeError("All loss-weight runs diverged")
        top = max(r["criterion"] for r in ok)
        chosen = sorted([r for r in ok if r["criterion"] == top], key=lambda r: (abs(r["alpha"] - 0.5), r["alpha"]))[0]
        for r in rows:
            r["chosen"] = r is chosen
        save_csv(self.path("loss_weights.csv"), rows)
        save_json(self.path("loss_weights.json"), {"alpha": chosen["alpha"], "beta": chosen["beta"],
                                                    "criterion": f"mean validation Dice over last {w} epochs",
                                                    "runs": rows})
        plots.plot_loss_weight_search(rows, chosen["alpha"], self.exp.figures / "loss_weight_selection")
        log.info("Selected loss weights: alpha (BCE) = %.3f, beta (soft-mIoU) = %.3f (validation criterion %.4f)",
                 chosen["alpha"], chosen["beta"], chosen["criterion"])
        return {"alpha": chosen["alpha"], "beta": chosen["beta"]}

    def final_alpha(self) -> float:
        """BCE weight of the final training: chosen on validation (grid) or the neutral 0.5 (fixed)."""
        if self.method.get("loss_weights") == "grid":
            return float(load_json(self.path("loss_weights.json"))["alpha"])
        return float(self.cfg["candidate_training"]["loss_alpha"])

    # ================================================================== 7. final training
    def stage_final(self) -> Dict[str, Any]:
        best = self.selected()
        alpha = self.final_alpha()
        fcfg = self.cfg["final_training"]
        log.info("Final training of %s for up to %d epochs | alpha=%.3f beta=%.3f (%s) | AdamW lr=%.1e wd=%.1e | "
                 "warm-up + poly schedule | early stopping on overfitting only", best["label"], fcfg["epochs"],
                 alpha, 1 - alpha, "chosen on validation" if self.method.get("loss_weights") == "grid" else "fixed",
                 fcfg["lr"], fcfg["weight_decay"])
        model = self.init_model(best["arch"])
        ckpt = self.exp.checkpoints / "final"
        tr = Trainer(model=model, data=self.data, tcfg=fcfg, alpha=alpha, device=self.device, ckpt_dir=ckpt,
                     seed=self.seed, fingerprint=self.fp, tag=f"final_{self.arch_key(best)}",
                     progress_prefix="Final training", early_stopping=fcfg["early_stopping"],
                     history_csv=self.path("final_training.csv"))
        out = tr.fit()
        state = load_best_weights(model, ckpt, self.device)
        loss_fn = BCESoftMIoULoss(alpha)
        bs = int(fcfg["batch_size"])
        train_m = evaluate(model, self.data.train_eval_loader(bs), loss_fn, self.device)
        val_m = evaluate(model, self.data.val_loader(bs), loss_fn, self.device)
        res = {"best_epoch": state["epoch"], "epochs_run": out["epochs_run"], "stopped_early": out["stopped_early"],
               "stop_reason": out["stop_reason"], "alpha": alpha, "beta": 1 - alpha,
               "train_metrics_best_checkpoint": train_m, "val_metrics_best_checkpoint": val_m}
        save_json(self.path("final_validation.json"), res)
        log.info("Best checkpoint (epoch %d) restored | Train Loss %.4f Dice %.4f | Val Loss %.4f Dice %.4f IoU %.4f "
                 "mIoU %.4f Sensitivity %.4f HD95 %.2f px", state["epoch"], train_m["loss"], train_m["dice"],
                 val_m["loss"], val_m["dice"], val_m["iou"], val_m["miou"], val_m["sensitivity"], val_m["hd95"])
        plots.plot_final_curves(out["history"], state["epoch"], self.exp.figures)
        return {"best_epoch": state["epoch"], "best_val_dice": val_m["dice"]}

    # ================================================================== 8. test (once)
    def stage_test(self) -> Dict[str, Any]:
        if self.exp.state.get("test_evaluated"):
            log.info("Test set already evaluated once for this experiment — not re-evaluating.")
            return {"skipped": True}
        for s in self.stages[:self.stages.index("test")]:
            if not self.exp.is_done(s):
                raise RuntimeError(f"Test evaluation requested before stage '{s}' finished — refusing (test isolation).")
        best = self.selected()
        fv = load_json(self.path("final_validation.json"))
        fcfg = self.cfg["final_training"]
        model = self.build(best["arch"])
        state = load_best_weights(model, self.exp.checkpoints / "final", self.device)
        model.to(self.device)
        log.info("Evaluating the held-out TEST split exactly once | best validation checkpoint (epoch %d)",
                 state["epoch"])
        test_m, rows, preds = evaluate(model, self.data.load_test(int(fcfg["batch_size"])),
                                       BCESoftMIoULoss(float(fv["alpha"])), self.device, keep=True)
        self.exp.state["test_evaluated"] = True
        self.exp.state["test_evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.exp.save_state()
        ids = [s.sample_id for s in self.splits["test"]]
        save_csv(self.path("test_per_image.csv"), [{"sample_id": i, **r} for i, r in zip(ids, rows)])
        np.savez_compressed(self.path("test_predictions.npz"), sample_ids=np.array(ids), masks=np.stack(preds))
        h, w = self.cfg["data"]["image_size"]
        cin = int(self.cfg["data"]["in_channels"])
        latency = inference_time_ms(model, [cin, h, w], self.device)
        cost = cost_summary(model, [h, w], cin, int(fcfg["batch_size"]))
        final = {
            "dataset": self.cfg["data"]["dataset"], "method": self.method_name, "method_label": self.method["label"], "seed": self.seed,
            "architecture": best["arch"], "architecture_label": best["label"],
            "blocks": decode(best["arch"]["chromosome"]) if best["arch"]["type"] == "qenas" else None,
            **cost, "inference_ms_per_image": latency, "inference_device": str(self.device),
            "zico": best.get("zico"), "alpha": fv["alpha"], "beta": fv["beta"],
            "best_epoch": fv["best_epoch"], "epochs_run": fv["epochs_run"],
            "train": fv["train_metrics_best_checkpoint"], "validation": fv["val_metrics_best_checkpoint"],
            "test": test_m, "n_test": len(ids),
        }
        save_json(self.path("final_metrics.json"), final)
        write_final_result(self.exp.dir, final)
        log.info("TEST | Dice (DSC) %.4f | mIoU %.4f | IoU %.4f | Sensitivity %.4f | HD95 %.2f px | Params %.3f M | "
                 "FLOPs %.2f G (batch of %d; %.2f G per image) | %.2f ms/image", test_m["dice"], test_m["miou"],
                 test_m["iou"], test_m["sensitivity"], test_m["hd95"], cost["params_m"], cost["gflops_mixed_ggnas"],
                 cost["flops_batch_size"], cost["gflops_per_image"], latency)
        return {"test_dice": test_m["dice"], "test_miou": test_m["miou"], "test_hd95": test_m["hd95"]}

    # ================================================================== 9. report
    def stage_report(self) -> Dict[str, Any]:
        cfg = self.cfg
        final = load_json(self.path("final_metrics.json"))
        best = self.selected()
        model = self.build(best["arch"])
        h, w = cfg["data"]["image_size"]
        bs = int(cfg["final_training"]["batch_size"])
        if isinstance(model, QENASNet):
            summ = model_summary(model, [h, w], int(cfg["data"]["in_channels"]), bs)
            plot_architecture(summ, self.exp.figures / "selected_architecture",
                              title=f"QENAS architecture selected on {cfg['data']['dataset']}")
            text = summary_text(summ)
        else:
            summ = cost_summary(model, [h, w], int(cfg["data"]["in_channels"]), bs)
            text = (f"{self.method['label']}: {summ['total_params']:,d} parameters ({summ['params_m']:.3f} M) | "
                    f"FLOPs {summ['gflops_per_image']:.3f} G per image, {summ['gflops_mixed_ggnas']:.2f} G per batch "
                    f"of {bs}")
        save_json(self.path("model_summary.json"), summ)
        self.path("model_summary.txt").write_text(text + "\n\n" + repr(model) + "\n", encoding="utf-8")
        searched = self.method["kind"] not in ("baseline", "random_arch")
        if searched:
            gens = load_json(self.path("nsga2_generations.json"))
            save_markdown_table(self.path("nsga2_generations.md"), [
                {"Generation": g["generation"], "Population": g["population"], "Pareto size": g["pareto_size"],
                 "Hypervolume": g["hypervolume"], "Best params": g["best_params"], "Best ZiCo": g["best_zico"],
                 "New evaluations": g["new_evaluations"]} for g in gens],
                ["Generation", "Population", "Pareto size", "Hypervolume", "Best params", "Best ZiCo",
                 "New evaluations"], title="NSGA-II population per generation", float_fmt="{:.4f}")
            front = load_json(self.path("pareto_front.json"))
            save_markdown_table(self.path("pareto_front.md"), [
                {"#": r["front_index"], "Chromosome": chromosome_to_str(r["chromosome"]), "Blocks": r["blocks"],
                 "Params": r["params"], "ZiCo": r["zico"]} for r in front],
                ["#", "Chromosome", "Blocks", "Params", "ZiCo"],
                title="Final Pareto front (non-dominated in the optimised objectives)", float_fmt="{:.3f}")
        repro = {
            "fingerprint": self.fp, "config": cfg, "environment": environment_info(self.device),
            "split_summary": load_json(self.path("split.json"))["summary"],
            "selected_architecture": {"architecture": best["arch"], "params": final["total_params"]},
            "loss_coefficients": {"alpha_bce": final["alpha"], "beta_soft_miou": final["beta"],
                                  "rule": self.method.get("loss_weights")},
            "stages": self.exp.state["stages"],
        }
        if self.path("search_summary.json").exists():
            repro["zico_batches_sha256"] = load_json(self.path("search_summary.json")).get("zico_batches_sha256")
        save_json(self.path("reproducibility.json"), repro)
        text = research_summary(cfg, self.method, best, final, load_json(self.path("final_validation.json")),
                                load_json(self.path("search_summary.json")) if searched else None,
                                load_json(self.path("successive_halving.json")) if searched else None)
        self.path("summary.txt").write_text(text, encoding="utf-8")
        return {}


def final_result_row(final: Dict[str, Any]) -> Dict[str, Any]:
    """The final test table: the Mixed-GGNAS metrics (Dice, mIoU, Params, FLOPs) + HD95 and sensitivity."""
    t = final["test"]
    return {"Dataset": final["dataset"], "Method": final["method_label"], "Seed": final["seed"],
            "Dice / DSC (%)": 100 * t["dice"], "mIoU (%)": 100 * t["miou"],
            "Params (M)": final["params_m"], f"FLOPs (G, batch {final['flops_batch_size']})": final["gflops_mixed_ggnas"],
            "HD95 (px)": t["hd95"], "Sensitivity (%)": 100 * t["sensitivity"], "IoU (%)": 100 * t["iou"],
            "GFLOPs / image": final["gflops_per_image"], "ms / image": final["inference_ms_per_image"]}


def write_final_result(exp_dir: Path, final: Dict[str, Any]) -> None:
    row = final_result_row(final)
    save_csv(exp_dir / "final_result.csv", [row])
    save_json(exp_dir / "final_result.json", [row])
    save_markdown_table(exp_dir / "final_result.md", [row], list(row), title="Final result (test set, evaluated once)",
                        float_fmt="{:.2f}")


def research_summary(cfg: Dict[str, Any], method: Dict[str, Any], best: Dict[str, Any], final: Dict[str, Any],
                     fv: Dict[str, Any], search: Optional[Dict[str, Any]], sh: Optional[Dict[str, Any]]) -> str:
    v = fv["val_metrics_best_checkpoint"]
    t = final["test"]
    lines = ["================ QENAS RESULT ================", "",
             f"Dataset: {cfg['data']['dataset']}",
             f"Method: {method['label']} — {method['doc']}", f"Seed: {cfg['seed']}", ""]
    if search is not None:
        a = best["analysis"]
        lines += [
            f"Search (training-free {'random sampling' if search.get('sampler') == 'random' else 'NSGA-II'}; "
            f"optimised objectives: {', '.join(search['objectives'])}):",
            f"Population: {cfg['search']['population_size']}   Generations: {cfg['search']['generations']}",
            f"Unique architectures evaluated: {search['unique_evaluations']}",
            f"Hypervolume: {search['hypervolume_generation_0']:.4f} (generation 0) -> {search['hypervolume_final']:.4f} (final)",
            f"Final Pareto front size: {search['final_front_size']}",
            f"Successive halving: {' -> '.join(map(str, sh['sizes']))} candidates trained to "
            f"{' -> '.join(map(str, sh['rung_epochs']))} epochs", "",
            "Selected architecture:",
            f"Chromosome: {best['label']}",
            f"Block sequence: {decode(best['arch']['chromosome'])}",
            f"ZiCo: {best['zico']:.3f}", "",
            "Candidate scores (mean val Dice of the last epochs): " +
            ", ".join(f"{e} ep {s:.4f}" for e, s in sorted(a["rung_scores"].items(), key=lambda kv: int(kv[0]))),
            f"Selection: {best['reason']}", ""]
    elif method["kind"] == "random_arch":
        lines += [f"Architecture: {best['label']} — {decode(best['arch']['chromosome'])} ({best['reason']}; no "
                  f"search)", ""]
    else:
        lines += [f"Architecture: {best['label']} (fixed baseline, no search)", ""]
    lines += [
        "Loss coefficients:",
        f"alpha (BCE): {final['alpha']:.3f}   beta (soft-mIoU): {final['beta']:.3f}"
        + ("" if method.get("loss_weights") == "grid" else "   (fixed, no loss-coefficient search)"), "",
        "Final training:",
        f"Epochs run: {fv['epochs_run']} (best epoch {fv['best_epoch']})" + (f" — {fv['stop_reason']}" if fv["stopped_early"] else ""),
        f"Best validation Dice: {v['dice']:.4f}   mIoU: {v['miou']:.4f}   HD95: {v['hd95']:.2f} px", "",
        f"Final test (n = {final['n_test']}, evaluated once):",
        f"Dice (DSC):        {100 * t['dice']:.2f} %",
        f"mIoU (class mean): {100 * t['miou']:.2f} %",
        f"Params:            {final['params_m']:.3f} M",
        f"FLOPs:             {final['gflops_mixed_ggnas']:.2f} G per batch of {final['flops_batch_size']} "
        f"(Mixed-GGNAS convention; {final['gflops_per_image']:.2f} G per image)",
        f"HD95:              {t['hd95']:.2f} px" + (f"  ({t['hd95_empty_pred']} empty predictions scored as the image diagonal)"
                                                    if t.get("hd95_empty_pred") else ""),
        f"Sensitivity:       {100 * t['sensitivity']:.2f} %",
        f"IoU (foreground):  {100 * t['iou']:.2f} %",
        f"Inference:         {final['inference_ms_per_image']:.2f} ms per image on {final['inference_device']}",
        f"Test loss:         {t['loss']:.4f}", "",
        "================================================", "",
    ]
    return "\n".join(lines)


def study_config(dataset: str, method: str, seed: int, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Configuration of one study experiment (used by study.py and its tests)."""
    from .config import build_config
    ov = {"method": method, "seed": seed}
    ov.update(overrides or {})
    return build_config(dataset, overrides=ov)
