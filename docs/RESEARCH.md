# Track 03 sleep-onset research

Private research workspace for the NeurIPS 2026 EEG/EMG Foundation
Challenge, Track 03 (sleep onset). It sits on the organizers' benchmark
from [neural-interfaces26/2026-competition](https://github.com/neural-interfaces26/2026-competition).
That upstream repository has no `LICENSE` file. Keep this history, the
upstream `README.md`, and the source headers. Dataset and model licenses
named by the organizers still apply.

The Git remote `origin` in a clone of the organizers' repository is
`https://github.com/neural-interfaces26/2026-competition.git`. Do not
push there. The team remote, once connected, is `research`:

`https://github.com/jawadchy2150/NeuroIPS-sleep-onset-research-2026.git`

## Install

Python 3.12. The Mac environment that already runs this tree is the
conda env `sleep-onset-2026` (PyTorch 2.13, MPS available). Recreate it
only when a machine does not already have it. Do not pass `--upgrade`.

```bash
conda create -n sleep-onset-2026 python=3.12 -y
conda activate sleep-onset-2026
pip install -r requirements.txt
pip install -r requirements-research.txt
```

`requirements.txt` is the organizers' list. `requirements-research.txt`
adds the Muse LightGBM stack (LightGBM 4.7.0, joblib 1.6.0, PyYAML 6.0.3)
and does not reinstall torch.

On AWS, install the CUDA 12.6 torch wheels before `requirements.txt`,
matching `tools/Dockerfile`:

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -r requirements-research.txt
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Do not copy the Mac conda environment onto the GPU machine.

## Devices

Team configs use `device: auto` in `muse_experiments/configs/default.yaml`.
`muse_experiments/device.py` tries CUDA, then Apple MPS, then CPU.
`MUSE_DEVICE` overrides one run (`cpu`, `mps`, or `cuda`). LightGBM
training stays on CPU. The resolved torch device is stored in new
checkpoints.

Official Benchopt solvers use `benchmark_utils.data.get_device`
(`COMPET_DEVICE`, otherwise CUDA or CPU). That function is unchanged.

## Data

These paths are local. They are not in Git.

| Data | Location |
| --- | --- |
| Sleep-EDF | `tracks/sleep_onset/data/` |
| Benchopt cache and solver outputs | `tracks/sleep_onset/__cache__/`, `tracks/sleep_onset/outputs/` |
| Muse EEG | `~/.neuralbench/data1/Interaxon2026Muse/download/nm000287` |
| Prepared Muse features | `muse_experiments/muse_features_100.csv` |
| Saved LightGBM model | `muse_experiments/lightgbm_100.joblib` |

A new machine needs its own downloads. Sleep-EDF preparation is
`benchopt prepare tracks/sleep_onset --config tracks/sleep_onset/training.yml`.
Muse download follows `tracks/sleep_onset/README.md`. Set
`NEURALBENCH_MUSE_ROOT` if the Muse files are not in the default cache.
Optional variables are listed in `.env.example`.

Feature extraction keeps recordings whose official `sessions.tsv` split
is `train`. LightGBM then holds out 20% of those subjects.

## Reproduce

From the repository root, with the env active. New runs write under
`muse_experiments/results/` and do not replace an existing feature
table, checkpoint, or error table.

```bash
python -m muse_experiments.preprocessing.prepare_muse_features
python -m muse_experiments.training.train_lightgbm
python -m muse_experiments.training.train_lightgbm_weighted
python -m muse_experiments.evaluation.analyze_lightgbm
```

Benchopt baseline:

```bash
benchopt run tracks/sleep_onset --config tracks/sleep_onset/training.yml
```

The 30-epoch EEGNet solver is `EEGNet-30epochs` in
`tracks/sleep_onset/solvers/eegnet_30epochs.py`.

Sleep-EDF scores from this repository are binned MAE and plain MAE in
`tracks/sleep_onset/objective.py`. The LightGBM weighted score is a
local check on the train split, not a Codabench result.

## Local compatibility fix

`benchmark_utils/nb_task.py` no longer sets `neuro.infra.permissions`.
The installed data stack rejects that key. Do not restore the line when
merging upstream without checking the stack.
