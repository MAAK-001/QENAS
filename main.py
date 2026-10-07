"""QENAS — Quick Evolutionary NAS.  Command-line entry point.

Examples
    python main.py --dataset BUSI
    python main.py --dataset CVC --population-size 10 --generations 10 --epochs 100
    python main.py --dataset IDRID --device cuda --batch-size 4
    python main.py --dataset BUSI --stop-after search          # search only; re-run later to continue
    python main.py --dataset BUSI --smoke                      # tiny end-to-end pipeline check
    python main.py --dataset BUSI --no-resume                  # force a fresh experiment

An interrupted run is resumed automatically by re-running the same command.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from qenas.config import QENAS_ROOT, SUPPORTED_DATASETS, build_config, config_fingerprint
from qenas.experiment import Experiment, ResumeConflict
from qenas.pipeline import STAGES, Pipeline
from qenas.utils.env import environment_info, resolve_device
from qenas.utils.io import save_json
from qenas.utils.logging_utils import setup_logging
from qenas.utils.seed import seed_everything


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QENAS: training-free NSGA-II (params, SynFlow) + learning-capability "
                                            "selection for medical image segmentation",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", required=True, type=str.upper, choices=SUPPORTED_DATASETS)
    p.add_argument("--config", default=None, help="optional YAML file with configuration overrides")
    p.add_argument("--smoke", action="store_true", help="tiny configuration (configs/smoke.yaml) to verify the pipeline")
    p.add_argument("--data-root", default=None, help="folder containing BUSI*/CVC*/IDRID* (default: ../Datasets)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N")
    g = p.add_argument_group("search")
    g.add_argument("--population-size", type=int, default=None)
    g.add_argument("--generations", type=int, default=None, help="evolutionary generations after generation 0")
    g.add_argument("--crossover-prob", type=float, default=None)
    g.add_argument("--mutation-prob-var", type=float, default=None, help="per-gene mutation probability")
    g.add_argument("--pareto-candidates", type=int, default=None, help="max Pareto candidates for supervised evaluation")
    g = p.add_argument_group("model")
    g.add_argument("--base-channels", type=int, default=None)
    g.add_argument("--scale", type=int, choices=(3, 5, 7), default=None, help="fixed convolution scale of manual blocks")
    g.add_argument("--darts-genotype", default=None, help="auto | darts_cell_busi | darts_cell_cvc | darts_cell_idrid | nasunet ...")
    g.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"), default=None)
    g.add_argument("--idrid-target", default=None, help="OD | MA | HE | EX | SE | LESIONS")
    g = p.add_argument_group("training")
    g.add_argument("--candidate-epochs", type=int, default=None, help="supervised epochs per Pareto candidate (default 20)")
    g.add_argument("--epochs", type=int, default=None, help="final-training epochs (default 100)")
    g.add_argument("--batch-size", type=int, default=None, help="batch size for all supervised stages")
    g.add_argument("--learning-rate", type=float, default=None, help="learning rate for all supervised stages")
    g.add_argument("--weight-decay", type=float, default=None)
    g.add_argument("--optimizer", default=None, choices=("adamw", "adam", "sgd"))
    g.add_argument("--scheduler", default=None, choices=("none", "poly", "cosine"), help="final-training schedule")
    g.add_argument("--grad-clip", type=float, default=None)
    g.add_argument("--amp", default=None, choices=("auto", "true", "false"))
    g.add_argument("--alphas", type=float, nargs="+", default=None, help="BCE weights to evaluate (beta = 1 - alpha)")
    g.add_argument("--no-loss-weight-search", action="store_true")
    g.add_argument("--early-stopping-patience", type=int, default=None)
    g.add_argument("--no-early-stopping", action="store_true")
    g.add_argument("--num-workers", type=int, default=None)
    g = p.add_argument_group("experiment control")
    r = g.add_mutually_exclusive_group()
    r.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="resume the latest unfinished experiment of this dataset (default)")
    r.add_argument("--no-resume", dest="resume", action="store_false", help="always start a new experiment")
    g.add_argument("--experiment-dir", default=None, help="resume this specific experiment directory")
    g.add_argument("--stop-after", default=None, choices=STAGES, help="stop after this stage (resume later)")
    return p.parse_args(argv)


def overrides_from_args(a: argparse.Namespace) -> dict:
    o = {
        "data.root": a.data_root, "seed": a.seed, "device": a.device,
        "search.population_size": a.population_size, "search.generations": a.generations,
        "search.crossover_prob": a.crossover_prob, "search.mutation_prob_var": a.mutation_prob_var,
        "pareto.max_candidates": a.pareto_candidates,
        "model.base_channels": a.base_channels, "model.scale": a.scale, "model.darts_genotype": a.darts_genotype,
        "data.image_size": list(a.image_size) if a.image_size else None,
        "data.idrid_target": a.idrid_target.upper() if a.idrid_target else None,
        "candidate_training.epochs": a.candidate_epochs, "final_training.epochs": a.epochs,
        "final_training.scheduler": a.scheduler, "loss_weights.alphas": a.alphas,
        "data.num_workers": a.num_workers,
    }
    for sect in ("candidate_training", "final_training"):
        o[f"{sect}.batch_size"] = a.batch_size
        o[f"{sect}.lr"] = a.learning_rate
        o[f"{sect}.weight_decay"] = a.weight_decay
        o[f"{sect}.optimizer"] = a.optimizer
        o[f"{sect}.grad_clip"] = a.grad_clip
        o[f"{sect}.amp"] = a.amp
    if a.candidate_epochs is not None:
        o["loss_weights.epochs"] = a.candidate_epochs   # keep alpha runs comparable to candidate runs
    if a.no_loss_weight_search:
        o["loss_weights.enabled"] = False
    if a.no_early_stopping:
        o["final_training.early_stopping.enabled"] = False
    if a.early_stopping_patience is not None:
        o["final_training.early_stopping.patience"] = a.early_stopping_patience
    return o


def main(argv=None) -> int:
    a = parse_args(argv)
    yaml_path = a.config
    if a.smoke:
        if yaml_path:
            raise SystemExit("--smoke cannot be combined with --config")
        yaml_path = str(QENAS_ROOT / "configs" / "smoke.yaml")
    cfg = build_config(a.dataset, yaml_path=yaml_path, overrides=overrides_from_args(a))
    if a.smoke:
        cfg["smoke_test"] = True
    fp = config_fingerprint(cfg)
    setup_logging()
    try:
        if a.experiment_dir:
            exp = Experiment.open_existing(Path(a.experiment_dir), cfg, fp)
        elif a.resume:
            found = Experiment.find_resumable(cfg)
            exp = Experiment.open_existing(found, cfg, fp) if found else Experiment.create(cfg, fp)
        else:
            exp = Experiment.create(cfg, fp)
    except ResumeConflict as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    log = setup_logging(exp.logs / "run.log")
    if exp.completed:
        log.info("Experiment %s is already completed; it will not be modified.", exp.dir)
        s = exp.dir / "summary.txt"
        if s.exists():
            print(s.read_text(encoding="utf-8"))
        return 0
    device = resolve_device(cfg["device"])
    seed_everything(int(cfg["seed"]), deterministic=bool(cfg["deterministic"]))
    env = environment_info(device)
    save_json(exp.dir / "config.json", {"fingerprint": fp, "config": cfg})
    save_json(exp.dir / "environment.json", env)
    log.info("QENAS | dataset %s | experiment %s | device %s | torch %s | python %s | seed %d | fingerprint %s",
             cfg["data"]["dataset"], exp.dir, device, env["torch_version"], env["python_version"], cfg["seed"], fp)
    if device.type == "cpu":
        log.warning("Running on CPU. The training-free search is fast, but supervised training of the Pareto "
                    "candidates and the final model is slow on CPU; a CUDA GPU is strongly recommended.")
    try:
        Pipeline(cfg, exp, device).run(stop_after=a.stop_after)
    except KeyboardInterrupt:
        log.warning("Interrupted. All completed epochs/generations are checkpointed; re-run the same command to resume.")
        return 130
    except Exception:
        log.error("QENAS stopped with an error:\n%s", traceback.format_exc())
        log.error("Progress up to the last checkpoint is preserved; fix the problem and re-run the same command.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
