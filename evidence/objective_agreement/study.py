"""One-off evidence study (not part of the QENAS pipeline): do the three search objectives agree or conflict?

Usage (from QENAS/):  python evidence/objective_agreement/study.py BUSI

For a fixed random sample of architectures (plus the five homogeneous ones) it computes exactly the search
objectives of QENAS - parameters, SynFlow and ZiCo (same initialisation, same two training batches, same
loss) - and stores them in evidence/objective_agreement/<DATASET>/results.json. Training split only;
validation and test images are never used. Resumable (results are appended after every architecture).
The analysis is written by analyse.py.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qenas.config import build_config  # noqa: E402
from qenas.datasets.dataset import DataBundle  # noqa: E402
from qenas.datasets.discovery import discover  # noqa: E402
from qenas.datasets.splits import make_splits  # noqa: E402
from qenas.search.objectives import ZICO_BATCHES, evaluate_architecture, zico_batches_digest  # noqa: E402
from qenas.utils.logging_utils import setup_logging  # noqa: E402
from qenas.utils.seed import seed_everything  # noqa: E402

N_RANDOM = 60
SAMPLE_SEED = 2026

ds = sys.argv[1].upper() if len(sys.argv) > 1 else "BUSI"
out_dir = Path(__file__).resolve().parent / ds
out_dir.mkdir(parents=True, exist_ok=True)
log = setup_logging(out_dir / "study.log")
seed_everything(42)
dev = torch.device("cpu")
cfg = build_config(ds)
splits = make_splits(discover(cfg), cfg)
data = DataBundle(splits, cfg)
batches = data.zico_batches(int(cfg["candidate_training"]["batch_size"]), ZICO_BATCHES, int(cfg["seed"]))
digest = zico_batches_digest(batches)

rng = np.random.default_rng(SAMPLE_SEED)
archs = [[g] * 8 for g in range(5)]
seen = {tuple(a) for a in archs}
while len(archs) < 5 + N_RANDOM:
    a = rng.integers(0, 5, 8).tolist()
    if tuple(a) not in seen:
        seen.add(tuple(a))
        archs.append(a)

res_file = out_dir / "results.json"
results = json.loads(res_file.read_text()) if res_file.exists() else {"dataset": ds, "zico_batches_sha256": digest,
                                                                     "image_size": cfg["data"]["image_size"],
                                                                     "rows": []}
if results["zico_batches_sha256"] != digest:
    raise SystemExit("existing results were computed with different ZiCo batches")
done = {tuple(r["chromosome"]) for r in results["rows"]}
t_all = time.time()
for i, a in enumerate(archs, 1):
    if tuple(a) in done:
        continue
    r = evaluate_architecture(a, cfg, dev, batches)
    results["rows"].append(r)
    res_file.write_text(json.dumps(results, indent=1))
    log.info("%d/%d %s params %.3fM | ln(1+S) %.2f | ZiCo %.2f | SynFlow %.1fs ZiCo %.1fs | elapsed %.0fs",
             i, len(archs), a, r["params"] / 1e6, r["synflow_log"], r["zico"], r["synflow_seconds"],
             r["zico_seconds"], time.time() - t_all)
log.info("done: %d architectures", len(results["rows"]))
