"""End-to-end QENAS pipeline.

    split -> search (NSGA-II, training-free) -> pareto (representative subset)
          -> candidates (E-epoch supervised runs) -> selection (learning capability)
          -> loss_weights (alpha on validation) -> final (ordinary supervised training)
          -> test (once, best-validation checkpoint) -> report

Every stage is idempotent and resumable; completed stages are skipped on re-run.
The test split is first materialised in ``stage_test`` and nowhere else.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import config_fingerprint
from .datasets.dataset import DataBundle
from .datasets.discovery import DatasetError, Sample, discover
from .datasets.splits import check_no_leakage, make_splits, split_summary
from .experiment import Experiment
from .models.chromosome import SEARCH_SPACE_SIZE, chromosome_key, chromosome_to_str, decode, decode_short
from .models.network import build_model, count_parameters
from .models.summary import model_summary, summary_text
from .search.nsga2_search import NSGA2Search
from .search.pareto import select_pareto_candidates
from .training.losses import BCESoftMIoULoss
from .training.selection import select_by_learning_capability
from .training.trainer import NumericalError, Trainer, amp_enabled, evaluate, load_best_weights
from .utils.env import environment_info
from .utils.io import load_json, save_csv, save_json, save_markdown_table
from .utils.logging_utils import format_seconds, get_logger
from .visualization import plots
from .visualization.architecture import plot_architecture

log = get_logger()

STAGES = ["split", "search", "pareto", "candidates", "selection", "loss_weights", "final", "test", "report"]


class Pipeline:
    def __init__(self, cfg: Dict[str, Any], exp: Experiment, device: torch.device):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.fp = exp.fingerprint
        self.seed = int(cfg["seed"])
        self._data: Optional[DataBundle] = None
        self._splits: Optional[Dict[str, List[Sample]]] = None

    # ================================================================== helpers
    def path(self, name: str) -> Path:
        return self.exp.dir / name

    @property
    def init_seed(self) -> int:
        sf = self.cfg["search"]["synflow"]
        return int(sf["init_seed"] if sf["init_seed"] is not None else self.seed)

    def init_model(self, chromosome: List[int]) -> torch.nn.Module:
        """Same seeded initialisation as the SynFlow evaluation: the scored network is the trained network."""
        devs = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(self.init_seed)
            return build_model(chromosome, self.cfg)

    @property
    def splits(self) -> Dict[str, List[Sample]]:
        if self._splits is None:
            d = load_json(self.path("split.json"))
            self._splits = {k: [Sample.from_dict(s) for s in v] for k, v in d["splits"].items()}
            for lst in self._splits.values():
                for s in lst:
                    for p in [s.image] + s.masks:
                        if not Path(p).exists():
                            raise DatasetError(f"File listed in split.json no longer exists: {p}")
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

    # ================================================================== driver
    def run(self, stop_after: Optional[str] = None) -> None:
        if stop_after is not None and stop_after not in STAGES:
            raise ValueError(f"--stop-after must be one of {STAGES}")
        for stage in STAGES:
            if self.exp.is_done(stage):
                log.info("Stage '%s' already completed — skipping", stage)
            else:
                log.info("#" * 100)
                log.info("STAGE: %s", stage.upper())
                t0 = time.time()
                info = getattr(self, f"stage_{stage}")() or {}
                info["seconds"] = round(time.time() - t0, 1)
                self.exp.mark_done(stage, info)
                log.info("Stage '%s' finished in %s", stage, format_seconds(info["seconds"]))
            if stop_after == stage:
                log.info("Stopping after stage '%s' as requested (resume later with the same command).", stage)
                return
        self.exp.mark_completed()
        print(self.path("summary.txt").read_text(encoding="utf-8"))

    # ================================================================== 1. split
    def stage_split(self) -> Dict[str, Any]:
        disc = discover(self.cfg)
        splits = make_splits(disc, self.cfg)
        summ = split_summary(splits)
        save_json(self.path("dataset_report.json"), disc.report)
        save_json(self.path("split.json"), {
            "dataset": disc.dataset, "seed": self.seed, "summary": summ,
            "splits": {k: [s.to_dict() for s in v] for k, v in splits.items()},
        })
        self._splits = splits
        ds = self.cfg["data"]["dataset"]
        log.info("Dataset structure: %s", disc.report["layout"])
        log.info("\n%s\nTrain: %d\nValidation: %d\nTest: %d", ds, summ["train"]["n"], summ["val"]["n"],
                 summ["test"]["n"])
        return {"train": summ["train"]["n"], "val": summ["val"]["n"], "test": summ["test"]["n"]}

    # ================================================================== 2. search
    def stage_search(self) -> Dict[str, Any]:
        s = self.cfg["search"]
        log.info("Training-free NSGA-II | search space 5^8 = %s | population %d | generations %d | "
                 "crossover p=%.2f | mutation p_var=%.3f | objectives: f1=#params, f2=1/(1+ln(1+SynFlow))",
                 f"{SEARCH_SPACE_SIZE:,}", s["population_size"], s["generations"], s["crossover_prob"],
                 s["mutation_prob_var"])
        search = NSGA2Search(self.cfg, self.exp.dir, self.device, self.fp)
        every = max(1, int(s["plot_every"]))

        def on_generation(srch: NSGA2Search) -> None:
            g = srch.generations_done - 1
            if g == 0 or g % every == 0:
                rows = [r for r in srch.population_history if r["generation"] == g]
                arch = [r for r in srch.archive.values() if r["first_generation"] <= g]
                name = "pareto_initial_population" if g == 0 else f"pareto_generation_{g:03d}"
                plots.plot_pareto_population(rows, arch, g, srch.n_generations, self.exp.figures / name)

        res = search.run(resume=True, on_generation=on_generation)
        hist = res["population_history"]
        last_gen = max(r["generation"] for r in hist)
        plots.plot_pareto_population([r for r in hist if r["generation"] == last_gen], res["archive"], last_gen,
                                     search.n_generations, self.exp.figures / "pareto_front_final",
                                     title=f"Final population and Pareto front (generation {last_gen})")
        plots.plot_pareto_evolution(hist, search.n_generations, self.exp.figures / "pareto_front_evolution")
        info = {"unique_evaluations": len(res["archive"]), "final_front_size": len(res["pareto_front"]),
                "cache_hits": res["cache_hits"], "cache_misses": res["cache_misses"]}
        save_json(self.path("search_summary.json"), info)
        log.info("Search complete: %d unique architectures evaluated, final Pareto front size %d",
                 info["unique_evaluations"], info["final_front_size"])
        return info

    # ================================================================== 3. pareto subset
    def stage_pareto(self) -> Dict[str, Any]:
        front = load_json(self.path("pareto_front.json"))
        k = int(self.cfg["pareto"]["max_candidates"])
        cands = select_pareto_candidates(front, k)
        for i, c in enumerate(cands, 1):
            c["candidate_id"] = i
        save_json(self.path("pareto_candidates.json"), cands)
        save_csv(self.path("pareto_candidates.csv"), cands, ["candidate_id", "chromosome", "blocks", "params",
                                                             "synflow_raw", "synflow_log", "f2_synflow",
                                                             "selection_reason"])
        all_rows = _read_archive(self.path("search_results.csv"))
        plots.plot_front_log_synflow(front, all_rows, self.exp.figures / "pareto_front_log_synflow", cands)
        hist = _read_history(self.path("search_population_history.csv"))
        last_gen = max(r["generation"] for r in hist)
        plots.plot_pareto_population([r for r in hist if r["generation"] == last_gen], all_rows, last_gen,
                                     int(self.cfg["search"]["generations"]),
                                     self.exp.figures / "pareto_front_final_with_candidates", highlight=cands,
                                     title="Final Pareto front and candidates selected for supervised evaluation",
                                     candidate_epochs=int(self.cfg["candidate_training"]["epochs"]))
        log.info("Final Pareto front: %d architectures; %d selected for %d-epoch evaluation (%s)", len(front),
                 len(cands), self.cfg["candidate_training"]["epochs"],
                 "entire front" if len(front) <= k else "extremes + farthest-point sampling")
        for c in cands:
            log.info("  C%d %s %s | %.3fM params | log-SynFlow %.2f | %s", c["candidate_id"],
                     chromosome_to_str(c["chromosome"]), c["blocks"], c["params"] / 1e6, c["synflow_log"],
                     c["selection_reason"])
        return {"front_size": len(front), "n_candidates": len(cands)}

    # ================================================================== 4. candidate training
    def stage_candidates(self) -> Dict[str, Any]:
        cands = load_json(self.path("pareto_candidates.json"))
        tcfg = self.cfg["candidate_training"]
        data = self.data
        results = []
        n = len(cands)
        t_all = time.time()
        for i, c in enumerate(cands, 1):
            cid = c["candidate_id"]
            ckpt = self.exp.checkpoints / "candidates" / f"cand_{cid:02d}"
            div_file = ckpt / "diverged.json"
            log.info("=" * 100)
            log.info("Candidate %d/%d | C%d %s | %s | %.3fM params | log-SynFlow %.2f", i, n, cid,
                     chromosome_to_str(c["chromosome"]), c["blocks"], c["params"] / 1e6, c["synflow_log"])
            rec = dict(c)
            if div_file.exists():
                rec.update(load_json(div_file))
                results.append(rec)
                log.warning("Candidate %d previously diverged: %s", cid, rec.get("divergence"))
                continue
            model = self.init_model(c["chromosome"])
            assert count_parameters(model) == c["params"], "parameter count mismatch between search and training"
            trainer = Trainer(model=model, data=data, tcfg=tcfg, alpha=float(tcfg["loss_alpha"]), device=self.device,
                              ckpt_dir=ckpt, seed=self.seed, fingerprint=self.fp,
                              tag=f"candidate_{cid}_{chromosome_key(c['chromosome'])}",
                              progress_prefix=f"Candidate {i}/{n}", early_stopping=None,
                              history_csv=self.exp.dir / "candidate_histories" / f"candidate_{cid:02d}.csv")
            try:
                out = trainer.fit()
                rec.update({"diverged": False, "history": out["history"]})
            except NumericalError as exc:
                log.error("Candidate %d diverged: %s — it is disqualified from selection (gate G1)", cid, exc)
                info = {"diverged": True, "divergence": str(exc), "history": trainer.history}
                save_json(div_file, info)
                rec.update(info)
            results.append(rec)
            self._write_candidate_long_csv(results)
            done = i
            el = time.time() - t_all
            log.info("Candidates completed: %d/%d (%.0f%%) | elapsed %s", done, n, 100 * done / n, format_seconds(el))
            del model, trainer
        save_json(self.path("candidate_results.json"), results)
        self._write_candidate_long_csv(results)
        return {"n_candidates": n, "n_diverged": sum(1 for r in results if r.get("diverged"))}

    def _write_candidate_long_csv(self, results: List[Dict[str, Any]]) -> None:
        rows = []
        for r in results:
            for h in r.get("history") or []:
                rows.append({"candidate_id": r["candidate_id"], "chromosome": r["chromosome"], "blocks": r["blocks"],
                             "params": r["params"], "synflow_log": r["synflow_log"], "f2_synflow": r["f2_synflow"],
                             **h})
        save_csv(self.path("candidate_training.csv"), rows)

    # ================================================================== 5. selection
    def stage_selection(self) -> Dict[str, Any]:
        results = load_json(self.path("candidate_results.json"))
        sc = self.cfg["selection"]
        sel = select_by_learning_capability(results, self.data.val_trivial_dice, float(sc["significance"]),
                                            int(sc["noise_window"]))
        save_json(self.path("selection.json"), sel)
        E = int(self.cfg["candidate_training"]["epochs"])
        rows = []
        for a in sorted(sel["analysis"], key=lambda a: a["candidate_id"]):
            g = a["gates"]
            rows.append({
                "Candidate": f"C{a['candidate_id']}", "Chromosome": chromosome_to_str(a["chromosome"]),
                "Blocks": a["blocks"], "Params": a["params"], "SynFlow (ln(1+S))": a["synflow_log"],
                "SynFlow objective f2": a["f2_synflow"],
                "Epoch-1 Dice": a.get("epoch1_val_dice"), f"Epoch-{E} Dice": a.get("final_val_dice"),
                "ΔDice": a.get("delta_dice"), "Epoch-1 Loss": a.get("epoch1_val_loss"),
                f"Epoch-{E} Loss": a.get("final_val_loss"), "ΔLoss": a.get("delta_loss"),
                f"Epoch-{E} IoU": a.get("final_val_iou"), "Dice trend tau": a.get("dice_trend_tau"),
                "Dice trend p": a.get("dice_trend_p"),
                "Gates": ("all passed" if all(g.get(k) for k in g) else
                          "failed " + ", ".join(k.split("_")[0] for k in sorted(g) if not g.get(k))),
                "Eligible": a["eligible"], "Capability rank": a["capability_rank"], "Selected": a["selected"],
            })
        save_csv(self.path("candidate_comparison.csv"), rows)
        save_json(self.path("candidate_comparison.json"), rows)
        cols = ["Candidate", "Params", "SynFlow (ln(1+S))", "Epoch-1 Dice", f"Epoch-{E} Dice", "ΔDice",
                "Epoch-1 Loss", f"Epoch-{E} Loss", f"Epoch-{E} IoU", "Gates", "Selected"]
        save_markdown_table(self.path("candidate_comparison.md"), rows, cols,
                            title=f"{E}-epoch learning-capability comparison (validation set)")
        # figures
        results_by_id = {r["candidate_id"]: r for r in results}
        cl = [results_by_id[k] for k in sorted(results_by_id)]
        f = self.exp.figures
        plots.plot_candidate_curves(cl, "val_dice", "Validation Dice", "Pareto candidates — validation Dice",
                                    f / "candidates_val_dice", sel["selected_id"])
        plots.plot_candidate_curves(cl, "val_loss", "Validation loss (α·BCE + β·soft-mIoU)",
                                    "Pareto candidates — validation loss", f / "candidates_val_loss",
                                    sel["selected_id"])
        plots.plot_delta_dice(sel["analysis"], f / "candidates_delta_dice")
        plots.plot_scatter_vs_dice(sel["analysis"], "params", "Trainable parameters (millions)",
                                   "Complexity vs segmentation quality", f / "params_vs_val_dice")
        plots.plot_scatter_vs_dice(sel["analysis"], "synflow_log", "log-SynFlow ln(1+S)",
                                   "SynFlow vs segmentation quality", f / "synflow_vs_val_dice")
        plots.plot_capability_ranking(sel, f / "learning_capability_ranking")
        log.info(sel["reason"])
        return {"selected_candidate": sel["selected_id"], "relaxed_gates": sel["relaxed_gates"]}

    def selected(self) -> Dict[str, Any]:
        sel = load_json(self.path("selection.json"))
        results = load_json(self.path("candidate_results.json"))
        rec = next(r for r in results if r["candidate_id"] == sel["selected_id"])
        ana = next(a for a in sel["analysis"] if a["candidate_id"] == sel["selected_id"])
        return {**rec, "analysis": ana, "reason": sel["reason"]}

    # ================================================================== 6. loss weights
    def stage_loss_weights(self) -> Dict[str, Any]:
        lw = self.cfg["loss_weights"]
        ctc = self.cfg["candidate_training"]
        cand_alpha = float(ctc["loss_alpha"])
        best = self.selected()
        if not lw["enabled"]:
            save_json(self.path("loss_weights.json"), {"enabled": False, "alpha": cand_alpha, "beta": 1 - cand_alpha})
            return {"alpha": cand_alpha}
        tcfg = dict(ctc)
        tcfg["epochs"] = int(lw["epochs"])
        w = int(lw["criterion_window"])
        rows = []
        alphas = sorted({round(float(a), 6) for a in lw["alphas"]})
        for i, a in enumerate(alphas, 1):
            log.info("=" * 100)
            log.info("Loss-weight run %d/%d | alpha (BCE) = %.3f, beta (soft-mIoU) = %.3f | architecture %s",
                     i, len(alphas), a, 1 - a, chromosome_to_str(best["chromosome"]))
            reuse = (lw["reuse_candidate_run"] and abs(a - cand_alpha) < 1e-12 and int(lw["epochs"]) == int(ctc["epochs"])
                     and not best.get("diverged"))
            if reuse:
                hist = best["history"]
                log.info("  identical configuration to the candidate run (same seed, data order, alpha, epochs) — "
                         "reusing its %d-epoch history", len(hist))
                status = "reused candidate run"
            else:
                model = self.init_model(best["chromosome"])
                tr = Trainer(model=model, data=self.data, tcfg=tcfg, alpha=a, device=self.device,
                             ckpt_dir=self.exp.checkpoints / "loss_weights" / f"alpha_{a:.3f}", seed=self.seed,
                             fingerprint=self.fp, tag=f"alpha_{a:.6f}",
                             progress_prefix=f"Loss-weight run {i}/{len(alphas)} (alpha={a:.2f})",
                             history_csv=self.exp.dir / "loss_weight_histories" / f"alpha_{a:.3f}.csv")
                try:
                    hist = tr.fit()["history"]
                    status = "trained"
                except NumericalError as exc:
                    log.error("alpha=%.3f diverged: %s", a, exc)
                    rows.append({"alpha": a, "beta": 1 - a, "criterion": None, "status": f"diverged: {exc}"})
                    continue
            crit = float(np.mean([h["val_dice"] for h in hist[-w:]]))
            rows.append({"alpha": a, "beta": round(1 - a, 6), "criterion": crit, "best_val_dice": max(h["val_dice"] for h in hist),
                         "final_val_dice": hist[-1]["val_dice"], "status": status})
            log.info("  alpha=%.3f -> mean val Dice over last %d epochs = %.4f", a, w, crit)
        ok = [r for r in rows if r["criterion"] is not None]
        if not ok:
            raise RuntimeError("All loss-weight runs diverged")
        top = max(r["criterion"] for r in ok)
        chosen = sorted([r for r in ok if r["criterion"] == top], key=lambda r: (abs(r["alpha"] - 0.5), r["alpha"]))[0]
        for r in rows:
            r["chosen"] = r is chosen
        save_csv(self.path("loss_weights.csv"), rows)
        save_json(self.path("loss_weights.json"), {"enabled": True, "alpha": chosen["alpha"], "beta": chosen["beta"],
                                                    "criterion": f"mean validation Dice over last {w} epochs",
                                                    "runs": rows})
        plots.plot_loss_weight_search(rows, chosen["alpha"], self.exp.figures / "loss_weight_selection")
        log.info("Selected loss weights: alpha (BCE) = %.3f, beta (soft-mIoU) = %.3f (validation criterion %.4f)",
                 chosen["alpha"], chosen["beta"], chosen["criterion"])
        return {"alpha": chosen["alpha"], "beta": chosen["beta"]}

    # ================================================================== 7. final training
    def stage_final(self) -> Dict[str, Any]:
        best = self.selected()
        lw = load_json(self.path("loss_weights.json"))
        alpha = float(lw["alpha"])
        fcfg = self.cfg["final_training"]
        log.info("Final training of %s (%s) for up to %d epochs | alpha=%.3f beta=%.3f | %s lr=%.1e wd=%.1e | "
                 "scheduler=%s | no scale selection, no scale collapse", chromosome_to_str(best["chromosome"]),
                 best["blocks"], fcfg["epochs"], alpha, 1 - alpha, fcfg["optimizer"], fcfg["lr"], fcfg["weight_decay"],
                 fcfg["scheduler"])
        model = self.init_model(best["chromosome"])
        ckpt = self.exp.checkpoints / "final"
        tr = Trainer(model=model, data=self.data, tcfg=fcfg, alpha=alpha, device=self.device, ckpt_dir=ckpt,
                     seed=self.seed, fingerprint=self.fp, tag=f"final_{chromosome_key(best['chromosome'])}",
                     progress_prefix="Final training", early_stopping=fcfg["early_stopping"],
                     history_csv=self.path("final_training.csv"))
        out = tr.fit()
        state = load_best_weights(model, ckpt, self.device)
        loss_fn = BCESoftMIoULoss(alpha)
        amp = amp_enabled(fcfg.get("amp", "auto"), self.device)
        bs = int(fcfg["batch_size"])
        train_m = evaluate(model, self.data.train_eval_loader(bs), loss_fn, self.device, amp)
        val_m = evaluate(model, self.data.val_loader(bs), loss_fn, self.device, amp)
        res = {"best_epoch": state["epoch"], "epochs_run": out["epochs_run"], "stopped_early": out["stopped_early"],
               "stop_reason": out["stop_reason"], "alpha": alpha, "beta": 1 - alpha,
               "train_metrics_best_checkpoint": train_m, "val_metrics_best_checkpoint": val_m}
        save_json(self.path("final_validation.json"), res)
        log.info("Best checkpoint (epoch %d) restored | Train Loss %.4f Dice %.4f IoU %.4f | Val Loss %.4f Dice %.4f "
                 "IoU %.4f", state["epoch"], train_m["loss"], train_m["dice"], train_m["iou"], val_m["loss"],
                 val_m["dice"], val_m["iou"])
        plots.plot_final_curves(out["history"], state["epoch"], self.exp.figures)
        return {"best_epoch": state["epoch"], "best_val_dice": val_m["dice"]}

    # ================================================================== 8. test (once)
    def stage_test(self) -> Dict[str, Any]:
        if self.exp.state.get("test_evaluated"):
            log.info("Test set already evaluated once for this experiment — not re-evaluating.")
            return {"skipped": True}
        for s in ("split", "search", "pareto", "candidates", "selection", "loss_weights", "final"):
            if not self.exp.is_done(s):
                raise RuntimeError(f"Test evaluation requested before stage '{s}' finished — refusing (test isolation).")
        best = self.selected()
        fv = load_json(self.path("final_validation.json"))
        model = build_model(best["chromosome"], self.cfg)
        state = load_best_weights(model, self.exp.checkpoints / "final", self.device)
        model.to(self.device)
        fcfg = self.cfg["final_training"]
        loss_fn = BCESoftMIoULoss(float(fv["alpha"]))
        log.info("Evaluating the held-out TEST split exactly once (best validation checkpoint, epoch %d)", state["epoch"])
        test_loader = self.data.load_test(int(fcfg["batch_size"]))
        test_m = evaluate(model, test_loader, loss_fn, self.device, amp_enabled(fcfg.get("amp", "auto"), self.device))
        self.exp.state["test_evaluated"] = True
        self.exp.state["test_evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.exp.save_state()
        summ = model_summary(model, self.cfg["data"]["image_size"], int(self.cfg["data"]["in_channels"]))
        final = {
            "dataset": self.cfg["data"]["dataset"], "chromosome": best["chromosome"], "blocks": decode(best["chromosome"]),
            "params": summ["total_params"], "gmacs": summ["gmacs"], "synflow_log": best["synflow_log"],
            "synflow_objective": best["f2_synflow"], "alpha": fv["alpha"], "beta": fv["beta"],
            "best_epoch": fv["best_epoch"], "epochs_run": fv["epochs_run"],
            "train": fv["train_metrics_best_checkpoint"], "validation": fv["val_metrics_best_checkpoint"],
            "test": test_m, "n_test": len(self.splits["test"]),
        }
        save_json(self.path("final_metrics.json"), final)
        row = {"Dataset": final["dataset"], "Params": final["params"], "Test Loss": test_m["loss"],
               "Test Dice": test_m["dice"], "Test IoU": test_m["iou"], "Test mIoU": test_m["miou"]}
        save_csv(self.path("final_result.csv"), [row])
        save_json(self.path("final_result.json"), [row])
        save_markdown_table(self.path("final_result.md"), [row], list(row), title="Final result (test set, evaluated once)")
        log.info("TEST | Loss %.4f | Dice %.4f | IoU %.4f | mIoU %.4f", test_m["loss"], test_m["dice"], test_m["iou"],
                 test_m["miou"])
        return {"test_dice": test_m["dice"], "test_iou": test_m["iou"]}

    # ================================================================== 9. report
    def stage_report(self) -> Dict[str, Any]:
        cfg = self.cfg
        final = load_json(self.path("final_metrics.json"))
        best = self.selected()
        model = build_model(best["chromosome"], cfg)
        summ = model_summary(model, cfg["data"]["image_size"], int(cfg["data"]["in_channels"]))
        save_json(self.path("model_summary.json"), summ)
        self.path("model_summary.txt").write_text(summary_text(summ) + "\n\n" + repr(model) + "\n", encoding="utf-8")
        plot_architecture(summ, self.exp.figures / "selected_architecture",
                          title=f"QENAS architecture selected on {cfg['data']['dataset']}")
        # tables
        gens = load_json(self.path("nsga2_generations.json"))
        save_markdown_table(self.path("nsga2_generations.md"), [
            {"Generation": g["generation"], "Population": g["population"], "Pareto Size": g["pareto_size"],
             "Best Params": g["best_params"], "Best SynFlow (f2)": g["best_synflow_objective"],
             "Best ln(1+SynFlow)": g["best_log_synflow"], "New evaluations": g["new_evaluations"]} for g in gens],
            ["Generation", "Population", "Pareto Size", "Best Params", "Best SynFlow (f2)", "Best ln(1+SynFlow)",
             "New evaluations"], title="NSGA-II population per generation", float_fmt="{:.5f}")
        front = load_json(self.path("pareto_front.json"))
        save_markdown_table(self.path("pareto_front.md"), [
            {"Rank": r["front_index"], "Chromosome": chromosome_to_str(r["chromosome"]), "Blocks": r["blocks"],
             "Params": r["params"], "SynFlow (f2)": r["f2_synflow"], "ln(1+SynFlow)": r["synflow_log"]} for r in front],
            ["Rank", "Chromosome", "Blocks", "Params", "SynFlow (f2)", "ln(1+SynFlow)"],
            title="Final Pareto front (non-dominated, sorted by parameters)", float_fmt="{:.5f}")
        repro = {
            "fingerprint": self.fp, "config": cfg, "environment": environment_info(self.device),
            "split_summary": load_json(self.path("split.json"))["summary"],
            "selected_architecture": {"chromosome": best["chromosome"], "blocks": decode(best["chromosome"]),
                                      "params": summ["total_params"]},
            "loss_coefficients": {"alpha_bce": final["alpha"], "beta_soft_miou": final["beta"]},
            "stages": self.exp.state["stages"],
        }
        save_json(self.path("reproducibility.json"), repro)
        text = research_summary(cfg, best, final, load_json(self.path("final_validation.json")),
                                load_json(self.path("search_summary.json")), len(load_json(self.path("pareto_candidates.json"))))
        self.path("summary.txt").write_text(text, encoding="utf-8")
        return {}


def research_summary(cfg: Dict[str, Any], best: Dict[str, Any], final: Dict[str, Any], fv: Dict[str, Any],
                     search: Dict[str, Any], n_cands: int) -> str:
    a = best["analysis"]
    E = int(cfg["candidate_training"]["epochs"])
    v = fv["val_metrics_best_checkpoint"]
    t = final["test"]
    return "\n".join([
        "================ QENAS RESULT ================", "",
        f"Dataset: {cfg['data']['dataset']}", "",
        "Search:",
        f"Population: {cfg['search']['population_size']}",
        f"Generations: {cfg['search']['generations']}",
        f"Unique architectures evaluated (training-free): {search['unique_evaluations']}",
        f"Final Pareto front size: {search['final_front_size']}",
        f"Final Pareto candidates: {n_cands}", "",
        "Selected architecture:",
        f"Chromosome: {chromosome_to_str(best['chromosome'])}",
        f"Block sequence: {decode(best['chromosome'])}", "",
        f"Parameters: {final['params']:,} ({final['params'] / 1e6:.3f} M)",
        f"SynFlow objective: {best['f2_synflow']:.6f}  (ln(1+SynFlow) = {best['synflow_log']:.3f})", "",
        f"{E}-epoch capability:",
        f"Epoch-1 Dice: {a['epoch1_val_dice']:.4f}",
        f"Epoch-{E} Dice: {a['final_val_dice']:.4f}",
        f"ΔDice: {a['delta_dice']:+.4f}",
        f"Selection: {best['reason']}", "",
        "Loss coefficients:",
        f"alpha (BCE): {final['alpha']:.3f}   beta (soft-mIoU): {final['beta']:.3f}", "",
        "Final training:",
        f"Epochs run: {fv['epochs_run']} (best epoch {fv['best_epoch']})" + (f" — {fv['stop_reason']}" if fv["stopped_early"] else ""),
        f"Best validation Dice: {v['dice']:.4f}",
        f"Best validation IoU: {v['iou']:.4f}", "",
        "Final test:",
        f"Test Loss: {t['loss']:.4f}",
        f"Test Dice: {t['dice']:.4f}",
        f"Test IoU: {t['iou']:.4f}",
        f"(Test mIoU, class-mean as in Mixed-GGNAS: {t['miou']:.4f}; n_test = {final['n_test']})", "",
        "================================================", "",
    ])


def _read_archive(path: Path) -> List[Dict[str, Any]]:
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        return [{"params": int(r["params"]), "f2_synflow": float(r["f2_synflow"]),
                 "synflow_log": float(r["synflow_log"])} for r in csv.DictReader(fh)]


def _read_history(path: Path) -> List[Dict[str, Any]]:
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        return [{"generation": int(r["generation"]), "params": int(r["params"]), "f2_synflow": float(r["f2_synflow"]),
                 "synflow_log": float(r["synflow_log"]), "nondominated": r["nondominated"] == "True"}
                for r in csv.DictReader(fh)]
