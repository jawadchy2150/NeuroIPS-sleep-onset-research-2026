# Muse experiments

Team code for Track 03 sleep-onset work on the Muse EEG set. The official
benchmark under `tracks/`, `benchmark_utils/`, `codabench/`, and `solution/`
stays as the organizers shipped it.

Setup for a new machine is in [`docs/team_setup.md`](../docs/team_setup.md).

## Layout

```text
muse_experiments/
  configs/             YAML settings (device, splits, paths, LightGBM)
  preprocessing/       Muse and Sleep-EDF preview, feature extraction
  models/              LightGBM regressor built from a config
  training/            unweighted and weighted LightGBM entry points
  evaluation/          validation error table and local W-bMAE
  results/             new feature tables, models, figures, analysis
  muse_features_100.csv
  lightgbm_100.joblib
  lightgbm_error_analysis.csv
```

The three files at the bottom are the existing run. Training and analysis
read them. New writes go to `results/` and do not replace a file that is
already there.

`tracks/sleep_onset/solvers/eegnet_30epochs.py` stays in `solvers/` so
Benchopt can still find `EEGNet-30epochs`.

## Commands

From the repo root:

```bash
python -m muse_experiments.preprocessing.prepare_muse_features
python -m muse_experiments.training.train_lightgbm
python -m muse_experiments.training.train_lightgbm_weighted
python -m muse_experiments.evaluation.analyze_lightgbm
```

Configs live in `configs/`. `device: auto` selects CUDA, then Apple MPS,
then CPU. Override one run with `MUSE_DEVICE`. LightGBM training stays on
CPU; the device setting is recorded in the saved checkpoint for torch runs.

## Splits

Feature extraction keeps Muse recordings whose official `sessions.tsv`
split is `train`. LightGBM then holds out 20% of those subjects with
`GroupShuffleSplit` (`random_state=42`).

The weighted bin score printed by training is a local check. Competition
numbers still come from `tracks/sleep_onset/objective.py` and Codabench.

## Device boundary

Official solvers use `benchmark_utils.data.get_device` (`COMPET_DEVICE`,
otherwise CUDA or CPU). Team torch code should use `resolve_device`.
