"""The QENAS pipeline - one fixed sequence of stages:

    split -> search (NSGA-II, training-free: params, SynFlow, ZiCo) -> pareto (representative subset)
          -> candidates (20-epoch supervised runs) -> selection (learning capability)
          -> loss_weights (alpha on validation) -> final (supervised training with validation;
             best checkpoint and flip-TTA decision fixed on validation) -> test (once) -> report

Every stage is idempotent and resumable; completed stages are skipped on re-run.
The test split is first materialised in ``stage_test`` and nowhere else.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .datasets.dataset import DataBundle
from .datasets.discovery import DatasetError, Sample, discover, rebase_path, relative_to_root
from .datasets.splits import check_no_leakage, make_splits, split_summary
from .experiment import Experiment
from .models.chromosome import SEARCH_SPACE_SIZE, chromosome_key, chromosome_to_str, decode
from .models.network import build_model, count_parameters
from .models.summary import model_summary, summary_text
from .search.nsga2_search import NSGA2Search
from .search.objectives import ZICO_BATCHES, ZICO_LOSS_ALPHA, zico_batches_digest
from .search.pareto import select_pareto_candidates
from .training.losses import BCESoftMIoULoss
from .training.selection import select_by_learning_capability
from .training.trainer import NumericalError, Trainer, evaluate, load_best_weights
from .utils.env import environment_info
from .utils.io import load_json, save_csv, save_json, save_markdown_table
from .utils.logging_utils import format_seconds, get_logger
from .visualization import plots
from .visualization.architecture import plot_architecture

log = get_logger()

STAGES = ["split", "search", "pareto", "candidates", "selection", "loss_weights", "final", "test", "report"]
OBJ_COLS = ["params", "synflow_raw", "synflow_log", "zico", "f1_params", "f2_synflow", "f3_zico"]


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

    def init_model(self, chromosome: List[int]) -> torch.nn.Module:
        """Same seeded initialisation as the zero-cost evaluation: the scored network is the trained network."""
        devs = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(self.seed)
            return build_model(chromosome, self.cfg)

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
        root = self.cfg["data"]["root"]

        def portable(s: Sample) -> Dict[str, Any]:
            d = s.to_dict()
            d["image"] = relative_to_root(s.image, root)
            d["masks"] = [relative_to_root(m, root) for m in s.masks]
            return d

        save_json(self.path("split.json"), {
            "dataset": disc.dataset, "seed": self.seed, "summary": summ, "data_root": root,
            "paths": "relative to data_root (re-based onto --data-root when resumed elsewhere)",
            "splits": {k: [portable(s) for s in v] for k, v in splits.items()},
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
        bs = int(self.cfg["candidate_training"]["batch_size"])
        log.info("Training-free NSGA-II | search space 5^8 = %s | population %d | generations %d | "
                 "crossover p=%.2f | mutation p_var=%.3f | objectives: f1 = #params, f2 = 1/(1+ln(1+SynFlow)), "
                 "f3 = 1/(1+ZiCo)", f"{SEARCH_SPACE_SIZE:,}", s["population_size"], s["generations"],
                 s["crossover_prob"], s["mutation_prob_var"])
        zb = self.data.zico_batches(bs, ZICO_BATCHES, self.seed)
        digest = zico_batches_digest(zb)
        log.info("ZiCo data: the first %d augmented training batches of epoch 0 (batch size %d, sha256 %s); "
                 "loss alpha=%.2f; validation and test images are not used by the search", ZICO_BATCHES, bs, digest,
                 ZICO_LOSS_ALPHA)
        search = NSGA2Search(self.cfg, self.exp.dir, self.device, self.fp, zb)
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
                "hypervolume_generation_0": table[0]["hypervolume"], "hypervolume_final": table[-1]["hypervolume"],
                "zico_batches_sha256": digest}
        save_json(self.path("search_summary.json"), info)
        log.info("Search complete: %d unique architectures evaluated, final Pareto front size %d, hypervolume "
                 "%.4f (generation 0) -> %.4f (generation %d)", info["unique_evaluations"], info["final_front_size"],
                 info["hypervolume_generation_0"], info["hypervolume_final"], last_gen)
        return {k: v for k, v in info.items() if k != "zico_batches_sha256"}

    # ================================================================== 3. pareto subset
    def stage_pareto(self) -> Dict[str, Any]:
        front = load_json(self.path("pareto_front.json"))
        k = int(self.cfg["pareto"]["max_candidates"])
        cands = select_pareto_candidates(front, k)
        for i, c in enumerate(cands, 1):
            c["candidate_id"] = i
        save_json(self.path("pareto_candidates.json"), cands)
        save_csv(self.path("pareto_candidates.csv"), cands,
                 ["candidate_id", "chromosome", "blocks", *OBJ_COLS, "selection_reason"])
        archive = load_json(self.path("search_results.json"))
        hist = load_json(self.path("search_population_history.json"))
        last_gen = max(r["generation"] for r in hist)
        plots.plot_front_3d(front, archive, self.exp.figures / "pareto_front_3d", cands)
        plots.plot_pareto_population([r for r in hist if r["generation"] == last_gen], archive, last_gen,
                                     int(self.cfg["search"]["generations"]),
                                     self.exp.figures / "pareto_front_final_with_candidates", highlight=cands,
                                     title="Final Pareto front and candidates selected for supervised evaluation",
                                     candidate_epochs=int(self.cfg["candidate_training"]["epochs"]))
        log.info("Final Pareto front: %d architectures; %d selected for %d-epoch evaluation (%s)", len(front),
                 len(cands), self.cfg["candidate_training"]["epochs"],
                 "entire front" if len(front) <= k else "extremes + farthest-point sampling")
        for c in cands:
            log.info("  C%d %s %s | %.3fM params | ln(1+SynFlow) %.2f | ZiCo %.2f | %s", c["candidate_id"],
                     chromosome_to_str(c["chromosome"]), c["blocks"], c["params"] / 1e6, c["synflow_log"], c["zico"],
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
            log.info("Candidate %d/%d | C%d %s | %s | %.3fM params | ln(1+SynFlow) %.2f | ZiCo %.2f", i, n, cid,
                     chromosome_to_str(c["chromosome"]), c["blocks"], c["params"] / 1e6, c["synflow_log"], c["zico"])
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
                              progress_prefix=f"Candidate {i}/{n}",
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
            log.info("Candidates completed: %d/%d (%.0f%%) | elapsed %s", i, n, 100 * i / n,
                     format_seconds(time.time() - t_all))
            del model, trainer
        save_json(self.path("candidate_results.json"), results)
        self._write_candidate_long_csv(results)
        return {"n_candidates": n, "n_diverged": sum(1 for r in results if r.get("diverged"))}

    def _write_candidate_long_csv(self, results: List[Dict[str, Any]]) -> None:
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
        sel = select_by_learning_capability(results, self.data.val_trivial_dice, float(sc["significance"]),
                                            int(sc["noise_window"]))
        save_json(self.path("selection.json"), sel)
        E = int(self.cfg["candidate_training"]["epochs"])
        rows = []
        for a in sorted(sel["analysis"], key=lambda a: a["candidate_id"]):
            g = a["gates"]
            rows.append({
                "Candidate": f"C{a['candidate_id']}", "Chromosome": chromosome_to_str(a["chromosome"]),
                "Blocks": a["blocks"], "Params": a["params"], "ln(1+SynFlow)": a["synflow_log"], "ZiCo": a["zico"],
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
        cols = ["Candidate", "Params", "ln(1+SynFlow)", "ZiCo", "Epoch-1 Dice", f"Epoch-{E} Dice", "ΔDice",
                "Epoch-1 Loss", f"Epoch-{E} Loss", f"Epoch-{E} IoU", "Gates", "Selected"]
        save_markdown_table(self.path("candidate_comparison.md"), rows, cols,
                            title=f"{E}-epoch learning-capability comparison (validation set)")
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
        plots.plot_scatter_vs_dice(sel["analysis"], "synflow_log", "ln(1 + SynFlow)",
                                   "SynFlow vs segmentation quality", f / "synflow_vs_val_dice")
        plots.plot_scatter_vs_dice(sel["analysis"], "zico", "ZiCo", "ZiCo vs segmentation quality",
                                   f / "zico_vs_val_dice")
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
        tcfg = self.cfg["candidate_training"]
        cand_alpha = float(tcfg["loss_alpha"])
        best = self.selected()
        w = int(lw["criterion_window"])
        rows = []
        alphas = sorted({round(float(a), 6) for a in lw["alphas"]})
        for i, a in enumerate(alphas, 1):
            log.info("=" * 100)
            log.info("Loss-weight run %d/%d | alpha (BCE) = %.3f, beta (soft-mIoU) = %.3f | architecture %s",
                     i, len(alphas), a, 1 - a, chromosome_to_str(best["chromosome"]))
            if abs(a - cand_alpha) < 1e-12 and not best.get("diverged"):
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

    # ================================================================== 7. final training
    def stage_final(self) -> Dict[str, Any]:
        best = self.selected()
        alpha = float(load_json(self.path("loss_weights.json"))["alpha"])
        fcfg = self.cfg["final_training"]
        log.info("Final training of %s (%s) for up to %d epochs | alpha=%.3f beta=%.3f | AdamW lr=%.1e wd=%.1e | "
                 "warm-up + poly schedule | early stopping on overfitting only", chromosome_to_str(best["chromosome"]),
                 best["blocks"], fcfg["epochs"], alpha, 1 - alpha, fcfg["lr"], fcfg["weight_decay"])
        model = self.init_model(best["chromosome"])
        ckpt = self.exp.checkpoints / "final"
        tr = Trainer(model=model, data=self.data, tcfg=fcfg, alpha=alpha, device=self.device, ckpt_dir=ckpt,
                     seed=self.seed, fingerprint=self.fp, tag=f"final_{chromosome_key(best['chromosome'])}",
                     progress_prefix="Final training", early_stopping=fcfg["early_stopping"],
                     history_csv=self.path("final_training.csv"))
        out = tr.fit()
        state = load_best_weights(model, ckpt, self.device)
        loss_fn = BCESoftMIoULoss(alpha)
        bs = int(fcfg["batch_size"])
        train_m = evaluate(model, self.data.train_eval_loader(bs), loss_fn, self.device)
        val_m = evaluate(model, self.data.val_loader(bs), loss_fn, self.device)
        # Flip test-time augmentation is used for the test iff it raises the validation Dice of this checkpoint.
        val_tta = evaluate(model, self.data.val_loader(bs), loss_fn, self.device, tta=True)
        use_tta = val_tta["dice"] > val_m["dice"]
        res = {"best_epoch": state["epoch"], "epochs_run": out["epochs_run"], "stopped_early": out["stopped_early"],
               "stop_reason": out["stop_reason"], "alpha": alpha, "beta": 1 - alpha,
               "train_metrics_best_checkpoint": train_m, "val_metrics_best_checkpoint": val_m,
               "val_metrics_best_checkpoint_hflip_tta": val_tta,
               "tta": {"use_tta": use_tta, "val_dice_no_tta": val_m["dice"], "val_dice_hflip_tta": val_tta["dice"]}}
        save_json(self.path("final_validation.json"), res)
        log.info("Best checkpoint (epoch %d) restored | Train Loss %.4f Dice %.4f | Val Loss %.4f Dice %.4f IoU %.4f "
                 "mIoU %.4f Sensitivity %.4f HD95 %.2f px", state["epoch"], train_m["loss"], train_m["dice"],
                 val_m["loss"], val_m["dice"], val_m["iou"], val_m["miou"], val_m["sensitivity"], val_m["hd95"])
        log.info("Flip TTA on validation: Dice %.4f (without %.4f) -> TTA %s for the test", val_tta["dice"],
                 val_m["dice"], "ON" if use_tta else "OFF")
        plots.plot_final_curves(out["history"], state["epoch"], self.exp.figures)
        return {"best_epoch": state["epoch"], "best_val_dice": val_m["dice"], "tta": use_tta}

    # ================================================================== 8. test (once)
    def stage_test(self) -> Dict[str, Any]:
        if self.exp.state.get("test_evaluated"):
            log.info("Test set already evaluated once for this experiment — not re-evaluating.")
            return {"skipped": True}
        for s in STAGES[:STAGES.index("test")]:
            if not self.exp.is_done(s):
                raise RuntimeError(f"Test evaluation requested before stage '{s}' finished — refusing (test isolation).")
        best = self.selected()
        fv = load_json(self.path("final_validation.json"))
        fcfg = self.cfg["final_training"]
        model = build_model(best["chromosome"], self.cfg)
        state = load_best_weights(model, self.exp.checkpoints / "final", self.device)
        model.to(self.device)
        use_tta = bool(fv["tta"]["use_tta"])
        log.info("Evaluating the held-out TEST split exactly once | best validation checkpoint (epoch %d) | flip TTA: %s",
                 state["epoch"], "on" if use_tta else "off")
        test_m = evaluate(model, self.data.load_test(int(fcfg["batch_size"])), BCESoftMIoULoss(float(fv["alpha"])),
                          self.device, tta=use_tta)
        self.exp.state["test_evaluated"] = True
        self.exp.state["test_evaluated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.exp.save_state()
        summ = model_summary(model, self.cfg["data"]["image_size"], int(self.cfg["data"]["in_channels"]),
                             int(fcfg["batch_size"]))
        final = {
            "dataset": self.cfg["data"]["dataset"], "chromosome": best["chromosome"], "blocks": decode(best["chromosome"]),
            "params": summ["total_params"], "params_m": summ["params_m"],
            "gflops_mixed_ggnas": summ["gflops_mixed_ggnas"], "gflops_per_image": summ["gflops_per_image"],
            "flops_batch_size": summ["flops_batch_size"],
            "synflow_log": best["synflow_log"], "zico": best["zico"], "alpha": fv["alpha"], "beta": fv["beta"],
            "best_epoch": fv["best_epoch"], "epochs_run": fv["epochs_run"],
            "train": fv["train_metrics_best_checkpoint"], "validation": fv["val_metrics_best_checkpoint"],
            "test": test_m, "n_test": len(self.splits["test"]), "test_tta": use_tta,
        }
        save_json(self.path("final_metrics.json"), final)
        row = final_result_row(final)
        save_csv(self.path("final_result.csv"), [row])
        save_json(self.path("final_result.json"), [row])
        save_markdown_table(self.path("final_result.md"), [row], list(row),
                            title="Final result (test set, evaluated once)", float_fmt="{:.2f}")
        log.info("TEST | Dice (DSC) %.4f | mIoU %.4f | IoU %.4f | Sensitivity %.4f | HD95 %.2f px | Params %.3f M | "
                 "FLOPs %.2f G (batch of %d; %.2f G per image)", test_m["dice"], test_m["miou"], test_m["iou"],
                 test_m["sensitivity"], test_m["hd95"], summ["params_m"], summ["gflops_mixed_ggnas"],
                 summ["flops_batch_size"], summ["gflops_per_image"])
        return {"test_dice": test_m["dice"], "test_miou": test_m["miou"], "test_hd95": test_m["hd95"]}

    # ================================================================== 9. report
    def stage_report(self) -> Dict[str, Any]:
        cfg = self.cfg
        final = load_json(self.path("final_metrics.json"))
        best = self.selected()
        model = build_model(best["chromosome"], cfg)
        summ = model_summary(model, cfg["data"]["image_size"], int(cfg["data"]["in_channels"]),
                             int(cfg["final_training"]["batch_size"]))
        save_json(self.path("model_summary.json"), summ)
        self.path("model_summary.txt").write_text(summary_text(summ) + "\n\n" + repr(model) + "\n", encoding="utf-8")
        plot_architecture(summ, self.exp.figures / "selected_architecture",
                          title=f"QENAS architecture selected on {cfg['data']['dataset']}")
        gens = load_json(self.path("nsga2_generations.json"))
        save_markdown_table(self.path("nsga2_generations.md"), [
            {"Generation": g["generation"], "Population": g["population"], "Pareto size": g["pareto_size"],
             "Hypervolume": g["hypervolume"], "Best params": g["best_params"],
             "Best ln(1+SynFlow)": g["best_log_synflow"], "Best ZiCo": g["best_zico"],
             "New evaluations": g["new_evaluations"]} for g in gens],
            ["Generation", "Population", "Pareto size", "Hypervolume", "Best params", "Best ln(1+SynFlow)", "Best ZiCo",
             "New evaluations"], title="NSGA-II population per generation", float_fmt="{:.4f}")
        front = load_json(self.path("pareto_front.json"))
        save_markdown_table(self.path("pareto_front.md"), [
            {"#": r["front_index"], "Chromosome": chromosome_to_str(r["chromosome"]), "Blocks": r["blocks"],
             "Params": r["params"], "ln(1+SynFlow)": r["synflow_log"], "ZiCo": r["zico"]} for r in front],
            ["#", "Chromosome", "Blocks", "Params", "ln(1+SynFlow)", "ZiCo"],
            title="Final Pareto front (non-dominated in parameters, SynFlow and ZiCo)", float_fmt="{:.3f}")
        repro = {
            "fingerprint": self.fp, "config": cfg, "environment": environment_info(self.device),
            "split_summary": load_json(self.path("split.json"))["summary"],
            "zico_batches_sha256": load_json(self.path("search_summary.json"))["zico_batches_sha256"],
            "selected_architecture": {"chromosome": best["chromosome"], "blocks": decode(best["chromosome"]),
                                      "params": summ["total_params"]},
            "loss_coefficients": {"alpha_bce": final["alpha"], "beta_soft_miou": final["beta"]},
            "test_tta": final["test_tta"], "stages": self.exp.state["stages"],
        }
        save_json(self.path("reproducibility.json"), repro)
        text = research_summary(cfg, best, final, load_json(self.path("final_validation.json")),
                                load_json(self.path("search_summary.json")), len(load_json(self.path("pareto_candidates.json"))))
        self.path("summary.txt").write_text(text, encoding="utf-8")
        return {}


def final_result_row(final: Dict[str, Any]) -> Dict[str, Any]:
    """The final test table: the Mixed-GGNAS metrics (Dice, mIoU, Params, FLOPs) + HD95 and sensitivity."""
    t = final["test"]
    return {"Dataset": final["dataset"], "Dice / DSC (%)": 100 * t["dice"], "mIoU (%)": 100 * t["miou"],
            "Params (M)": final["params_m"], f"FLOPs (G, batch {final['flops_batch_size']})": final["gflops_mixed_ggnas"],
            "HD95 (px)": t["hd95"], "Sensitivity (%)": 100 * t["sensitivity"], "IoU (%)": 100 * t["iou"],
            "GFLOPs / image": final["gflops_per_image"], "Flip TTA": "yes" if final["test_tta"] else "no"}


def research_summary(cfg: Dict[str, Any], best: Dict[str, Any], final: Dict[str, Any], fv: Dict[str, Any],
                     search: Dict[str, Any], n_cands: int) -> str:
    a = best["analysis"]
    E = int(cfg["candidate_training"]["epochs"])
    v = fv["val_metrics_best_checkpoint"]
    t = final["test"]
    return "\n".join([
        "================ QENAS RESULT ================", "",
        f"Dataset: {cfg['data']['dataset']}", "",
        "Search (training-free NSGA-II; objectives: parameters, SynFlow, ZiCo):",
        f"Population: {cfg['search']['population_size']}",
        f"Generations: {cfg['search']['generations']}",
        f"Unique architectures evaluated: {search['unique_evaluations']}",
        f"Hypervolume: {search['hypervolume_generation_0']:.4f} (generation 0) -> {search['hypervolume_final']:.4f} (final)",
        f"Final Pareto front size: {search['final_front_size']}",
        f"Pareto candidates trained: {n_cands}", "",
        "Selected architecture:",
        f"Chromosome: {chromosome_to_str(best['chromosome'])}",
        f"Block sequence: {decode(best['chromosome'])}",
        f"ln(1+SynFlow): {best['synflow_log']:.3f}   ZiCo: {best['zico']:.3f}", "",
        f"{E}-epoch capability:",
        f"Epoch-1 Dice: {a['epoch1_val_dice']:.4f}",
        f"Epoch-{E} Dice: {a['final_val_dice']:.4f}",
        f"ΔDice: {a['delta_dice']:+.4f}",
        f"Selection: {best['reason']}", "",
        "Loss coefficients:",
        f"alpha (BCE): {final['alpha']:.3f}   beta (soft-mIoU): {final['beta']:.3f}", "",
        "Final training:",
        f"Epochs run: {fv['epochs_run']} (best epoch {fv['best_epoch']})" + (f" — {fv['stop_reason']}" if fv["stopped_early"] else ""),
        f"Best validation Dice: {v['dice']:.4f}   mIoU: {v['miou']:.4f}   HD95: {v['hd95']:.2f} px",
        f"Validation Dice with flip TTA: {fv['tta']['val_dice_hflip_tta']:.4f} -> TTA {'on' if fv['tta']['use_tta'] else 'off'}",
        "",
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
        f"Test loss:         {t['loss']:.4f}", "",
        "================================================", "",
    ])
