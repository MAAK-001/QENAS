"""Regenerate every experiment's figures from its saved results (PNG files are not stored on GitHub).

Uses exactly the plotting calls of the pipeline stages, reading the stage outputs (search history, candidate
histories, selection, loss weights, final training log, model summary). Nothing is recomputed or retrained.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..utils.io import load_json
from ..visualization import plots
from ..visualization.architecture import plot_architecture
from .analysis import _read_csv


def regenerate(results: Path) -> List[str]:
    done = []
    for st in sorted(Path(results).glob("*/*/seed_*/state.json")):
        d = st.parent
        f = d / "figures"
        f.mkdir(exist_ok=True)
        cfg = load_json(d / "config.json")["config"]
        if (d / "search_population_history.json").exists():
            G, every = cfg["search"]["generations"], cfg["search"]["plot_every"]
            hist = load_json(d / "search_population_history.json")
            arch = load_json(d / "search_results.json")
            for g in sorted({r["generation"] for r in hist}):
                if g == 0 or g % every == 0:
                    plots.plot_pareto_population([r for r in hist if r["generation"] == g],
                                                 [r for r in arch if r["first_generation"] <= g], g, G,
                                                 f / ("pareto_initial_population" if g == 0 else f"pareto_generation_{g:03d}"))
            last = max(r["generation"] for r in hist)
            plots.plot_pareto_population([r for r in hist if r["generation"] == last], arch, last, G,
                                         f / "pareto_front_final",
                                         title=f"Final population and Pareto front (generation {last})")
            plots.plot_search_progress(hist, load_json(d / "nsga2_generations.json"), f / "search_progress")
            if (d / "pareto_candidates.json").exists():
                plots.plot_pareto_population([r for r in hist if r["generation"] == last], arch, last, G,
                                             f / "pareto_front_final_with_candidates",
                                             highlight=load_json(d / "pareto_candidates.json"),
                                             title="Final population and the candidates trained by successive halving")
        if (d / "selection.json").exists() and (d / "successive_halving.json").exists():
            res, sel = load_json(d / "candidate_results.json"), load_json(d / "selection.json")
            sh = load_json(d / "successive_halving.json")
            cl = sorted(res, key=lambda r: r["candidate_id"])
            plots.plot_candidate_curves(cl, "val_dice", "Validation Dice", "Candidates — validation Dice",
                                        f / "candidates_val_dice", sel["selected_id"])
            plots.plot_candidate_curves(cl, "val_loss", "Validation loss (α·BCE + β·soft-mIoU)",
                                        "Candidates — validation loss", f / "candidates_val_loss", sel["selected_id"])
            plots.plot_successive_halving(sh, sel["analysis"], f / "successive_halving")
            first = str(sh["rung_epochs"][0])
            for key, label, title, name in (("params", "Trainable parameters (millions)",
                                             "Complexity vs early validation quality", "params_vs_val_dice"),
                                            ("zico", "ZiCo", "ZiCo vs early validation quality", "zico_vs_val_dice")):
                plots.plot_scatter_vs_dice(sel["analysis"], key, label, title, f / name, first)
            plots.plot_capability_ranking(sel, int(sh["window"]), f / "learning_capability_ranking")
        if (d / "loss_weights.json").exists():
            lw = load_json(d / "loss_weights.json")
            plots.plot_loss_weight_search(lw["runs"], lw["alpha"], f / "loss_weight_selection")
        if (d / "final_training.csv").exists() and (d / "final_validation.json").exists():
            h = _read_csv(d / "final_training.csv")
            plots.plot_final_curves(h, int(load_json(d / "final_validation.json")["best_epoch"]), f)
        ms = d / "model_summary.json"
        if ms.exists() and "positions" in load_json(ms):
            plot_architecture(load_json(ms), f / "selected_architecture",
                              title=f"QENAS architecture selected on {cfg['data']['dataset']}")
        done.append(str(d))
    print(f"[figures] regenerated figures of {len(done)} experiments")
    return done
