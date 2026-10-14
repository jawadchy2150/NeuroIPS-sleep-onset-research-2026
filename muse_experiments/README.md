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
  submission/          Codabench submission.py (no LightGBM import)
  results/             new feature tables, models, figures, analysis
  muse_features_100.csv
  lightgbm_100.joblib
  lightgbm_error_analysis.csv
```

The three files at the bottom are the earlier run. That table includes
windows after N2, so its 68.25 s figure is not a competition score. New
writes go to `results/` and do not replace a file that is already there.

`tracks/sleep_onset/solvers/eegnet_30epochs.py` stays in `solvers/` so
Benchopt can still find `EEGNet-30epochs`.

## Commands

From the repo root:

```bash
python -m muse_experiments.preprocessing.prepare_muse_features
python -m muse_experiments.training.train_lightgbm
python -m muse_experiments.training.train_lightgbm_weighted
python -m muse_experiments.evaluation.analyze_lightgbm
python -m muse_experiments.evaluation.compare_lightgbm
python -m muse_experiments.training.train_tcn
python -m muse_experiments.training.train_cnn
```

The corrected table is
`results/features/muse_features_100_pre_n2.csv`, from the approved
100-recording run. A repeat does not replace an existing file. Do not run
NeuralBench `--prepare` on this Mac. The temporal CNN commands read raw
windows into `results/tcn/` and write their own checkpoints. They do not
replace the LightGBM table or models.

Configs live in `configs/`. `device: auto` selects CUDA, then Apple MPS,
then CPU. Override one run with `MUSE_DEVICE`. LightGBM training stays on
CPU; the device setting is recorded in the saved checkpoint for torch runs.

## Splits and score

Feature extraction keeps Muse recordings whose official `sessions.tsv`
split is `train`. Subjects who also have a test recording stay in
training. Of the others, 20% are validation, using sorted subject ids and
`train_test_split` with `random_state=33`.

Windows are non-overlapping 5 s blocks at 128 Hz that end at or before
the first N2. The target is the remaining time, clipped to 0–600 s.
Features are computed in microvolts. The printed W-bMAE is the mean of
the per-recording weighted bin scores. The Codabench file is
`submission/submission.py`. Packaging steps are in
[`docs/muse_codabench.md`](../docs/muse_codabench.md).

## Device boundary

Official solvers use `benchmark_utils.data.get_device` (`COMPET_DEVICE`,
otherwise CUDA or CPU). Team torch code should use `resolve_device`.
