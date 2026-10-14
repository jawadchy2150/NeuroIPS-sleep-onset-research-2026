# Team setup

Track 03 work on top of the organizers' benchmark. The Git remote `origin`
is `https://github.com/neural-interfaces26/2026-competition.git`. Do not
push there. Add a private remote later with `git remote add`; leave
`origin` as it is until the team decides how to track upstream.

Local environment that already runs this project: conda env
`sleep-onset-2026`, Python 3.12. System Python on the Mac is 3.14 and does
not have the stack.

## Local Mac

```bash
conda activate sleep-onset-2026
cd "/path/to/2026-competition"
```

If the env has to be recreated on another Mac, use Python 3.12 and install
the competition requirements first, without `--upgrade`:

```bash
conda create -n sleep-onset-2026 python=3.12 -y
conda activate sleep-onset-2026
pip install -r requirements.txt
pip install -r requirements-research.txt
```

`requirements.txt` is the organizers' list, including a git pin of the
neuroai stack. The existing `sleep-onset-2026` env was not rebuilt from
that file in this pass (it has PyPI `neuralbench` 0.3.1 and `exca` 0.6.0).
Reinstalling `requirements.txt` into that env can change those packages.
Do that only when you intend to refresh the stack.

Torch in the local env is conda-forge PyTorch 2.13 with MPS available.
Leave that install in place on the Mac.

Optional environment variables are listed in `.env.example`.

## AWS GPU

Use a Linux instance with an NVIDIA driver new enough for CUDA 12.6
(the competition image uses the `cu126` wheel so it does not follow PyPI's
newest CUDA). Python 3.12.

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -r requirements-research.txt
```

Install torch before `requirements.txt`. The requirements file lists
`torch` without a version, and a wheel already present is what satisfies
it. Do not copy the Mac conda environment onto the GPU machine.

Check the device before a long run:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Team code falls back to CPU when CUDA is missing. Official Benchopt runs do
the same unless `COMPET_DEVICE` is set.

## Data already on this machine

Nothing here downloads data.

| Data | Where it is | Git |
| --- | --- | --- |
| Sleep-EDF | `tracks/sleep_onset/data/` (about 25 GB) | ignored by `**/data/` |
| Benchopt caches and solver outputs | `tracks/sleep_onset/__cache__/`, `tracks/sleep_onset/outputs/` | ignored |
| Muse EEG | `~/.neuralbench/data1/Interaxon2026Muse/download/nm000287` | outside the repo |
| Muse feature table | `muse_experiments/muse_features_100.csv` | ignored by `*.csv` |

A teammate needs the Muse download on their own machine (`neuralbench eeg
sleep_onset` is the organizers' download path; see `tracks/sleep_onset/README.md`).
Point `NEURALBENCH_MUSE_ROOT` at that directory if it is not in the default
cache. Feature extraction reads the official `sessions.tsv` split and keeps
`train` recordings only.

Sleep-EDF preparation, when a machine does not already have it:

```bash
benchopt prepare tracks/sleep_onset --config tracks/sleep_onset/training.yml
```

That step is large. Do not start it on a laptop that already has the files.

## Reproduce the current experiments

From the repo root, with `sleep-onset-2026` active. Corrected runs write
under `muse_experiments/results/` and leave `muse_features_100.csv`,
`lightgbm_100.joblib`, and `lightgbm_error_analysis.csv` in place. If a
destination already exists, the run adds a timestamp instead of replacing
it. The corrected 100-recording table and both LightGBM checkpoints are
already in `muse_experiments/results/`. Scores are in
`docs/muse_codabench.md`.

```bash
python -m muse_experiments.preprocessing.prepare_muse_features
python -m muse_experiments.training.train_lightgbm
python -m muse_experiments.training.train_lightgbm_weighted
python -m muse_experiments.evaluation.analyze_lightgbm
```

The same commands still work through the old filenames
(`python prepare_muse_features.py`, `python muse_experiments/train_lightgbm.py`,
and so on). Pass a config name to select another YAML file, for example
`python -m muse_experiments.evaluation.analyze_lightgbm lightgbm_weighted`.

`device: auto` in `muse_experiments/configs/default.yaml` uses CUDA, then
Apple MPS, then CPU. LightGBM itself stays on CPU. Torch models should call
`resolve_device` with that setting. Official Benchopt solvers still use
`benchmark_utils.data.get_device`.

Benchopt baselines, also from the repo root:

```bash
benchopt run tracks/sleep_onset --config tracks/sleep_onset/training.yml
```

`training.yml` trains `Torch-Linear`. The 30-epoch EEGNet solver is
`EEGNet-30epochs` in `tracks/sleep_onset/solvers/eegnet_30epochs.py`. Select
it with Benchopt `-s` when you want that run. Its outputs already live
under `tracks/sleep_onset/outputs/`.

## Scores

This checkout's `tracks/sleep_onset/objective.py` still reports unweighted
binned MAE. The Muse warm-up score is weighted binned MAE inside each
recording, then the mean across recordings. That calculation is in
`muse_experiments/evaluation/metrics.py`. The earlier LightGBM figure of
68.25 s came from a table that kept windows after N2, so it is not a
competition result. Packaging and the corrected run are described in
`docs/muse_codabench.md`.

## Where the team scripts live

| Script | Location |
| --- | --- |
| Feature extraction | `muse_experiments/preprocessing/prepare_muse_features.py` |
| Previews | `muse_experiments/preprocessing/explore_muse.py`, `explore_eeg.py` |
| LightGBM training | `muse_experiments/training/` |
| Error analysis | `muse_experiments/evaluation/analyze_lightgbm.py` |
| Settings | `muse_experiments/configs/` |
| Existing feature table and checkpoint | `muse_experiments/muse_features_100.csv`, `lightgbm_100.joblib` |
| New run outputs | `muse_experiments/results/` |
| EEGNet-30epochs | `tracks/sleep_onset/solvers/eegnet_30epochs.py` |
