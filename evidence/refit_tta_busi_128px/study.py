"""One-off evidence study (not part of the QENAS pipeline).
Usage (from QENAS/): python evidence/refit_tta_busi_128px/study.py   (~3-4 h on a CPU)

Evidence for the refit (data scaling) and flip-TTA decisions — train/validation splits only, test never loaded.

Selected BUSI architecture from the user's Kaggle run, reduced cost (128x128, 30 epochs) to fit a CPU.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qenas.config import build_config  # noqa: E402
from qenas.datasets.dataset import DataBundle  # noqa: E402
from qenas.datasets.discovery import discover  # noqa: E402
from qenas.datasets.splits import make_splits  # noqa: E402
from qenas.models.network import build_model  # noqa: E402
from qenas.training.losses import BCESoftMIoULoss  # noqa: E402
from qenas.training.trainer import Trainer, evaluate, load_best_weights  # noqa: E402
from qenas.utils.logging_utils import setup_logging  # noqa: E402
from qenas.utils.seed import seed_everything  # noqa: E402

OUT = Path(__file__).resolve().parent
CH = [4, 3, 3, 2, 2, 3, 3, 3]
ALPHA = 0.2
EPOCHS = 30
FRACTIONS = [0.5, 0.75, 1.0]

setup_logging(OUT / "evidence.log")
seed_everything(42)
dev = torch.device("cpu")
cfg = build_config("BUSI", overrides={"data.image_size": [128, 128]})
splits = make_splits(discover(cfg), cfg)
bundle = DataBundle(splits, cfg)
full_imgs, full_masks = bundle.train_images, bundle.train_masks
perm = np.random.default_rng(0).permutation(len(full_imgs))       # nested subsets
tcfg = dict(cfg["final_training"], epochs=EPOCHS)
results = json.loads((OUT / "results.json").read_text()) if (OUT / "results.json").exists() else {}
for frac in FRACTIONS:
    key = f"{frac:.2f}"
    if key in results:
        continue
    idx = np.sort(perm[: int(round(frac * len(full_imgs)))])
    bundle.train_images, bundle.train_masks = full_imgs[idx], full_masks[idx]
    torch.manual_seed(42)
    model = build_model(CH, cfg)
    tr = Trainer(model=model, data=bundle, tcfg=tcfg, alpha=ALPHA, device=dev, ckpt_dir=OUT / f"ckpt_{key}",
                 seed=42, fingerprint="evidence", tag=f"frac_{key}", progress_prefix=f"frac {key} (n={len(idx)})",
                 early_stopping=None, history_csv=OUT / f"history_{key}.csv")
    out = tr.fit()
    load_best_weights(model, OUT / f"ckpt_{key}", dev)
    lf = BCESoftMIoULoss(ALPHA)
    vl = bundle.val_loader(8)
    plain = evaluate(model, vl, lf, dev, tta=False)
    tta = evaluate(model, vl, lf, dev, tta=True)
    hist = out["history"]
    results[key] = {"n_train": int(len(idx)), "best_epoch": out["best_epoch"], "best_val_dice": out["best_val_dice"],
                    "mean_val_dice_last5": float(np.mean([h["val_dice"] for h in hist[-5:]])),
                    "best_ckpt_val": plain, "best_ckpt_val_hflip_tta": tta}
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
