# QENAS evaluation study v2 — results

This branch holds only the results of evaluation study v2 (QENAS 3.0, seeds 45, 46, 47), pushed from Kaggle by
the notebook cell in README.md §6 of `main`. The code is on `main`; study v1 (QENAS 2.1) is at tag `study-v1`
with its results on branch `kaggle-study`.

* `results/<DATASET>/<method>/seed_<s>/` — one experiment;
* `results/analysis/` — tables T1–T15 (`python study.py analyse`);
* `results/study_status.json`, `results/study_logs/` — scheduler state and job logs;
* `kaggle_notebook.log` — the notebook's own log.
