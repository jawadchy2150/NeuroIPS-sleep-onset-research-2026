# Corrected Muse LightGBM run

The earlier local score of 68.25 s is not a competition result. That
feature table kept windows after the first N2 and stored their target as
0. The files are still on disk:

- `muse_experiments/muse_features_100.csv`
- `muse_experiments/lightgbm_100.joblib`
- `muse_experiments/lightgbm_error_analysis.csv`

Do not upload that joblib file.

## Warm-up contract used here

Checked against the organizers' current streaming metric and this
machine's Muse download (metadata and one 5 s window only):

- 500 train sessions, 40 test sessions, 203 subjects. Forty subjects
  have a test session.
- Channels, in file order: TP9, AF7, AF8, TP10. 128 Hz.
  `SamplingInterval=7812.5` µs.
- BrainVision resolution is 0.1 µV. MNE returns volts. One window had
  standard deviations of about 7–25 µV after multiplying by `1e6`, which
  is the unit Codabench passes in.
- No extra filter, notch, scaler, or clamp. The header says
  `SoftwareFilters: n/a`.
- Non-overlapping 640-sample windows whose end is at or before the
  `n2_onset` event. The target is remaining seconds, clipped to 0–600.
- Subjects with any official test session stay in training. Validation
  is 20% of the other subjects via `train_test_split`, `random_state=33`,
  after sorting the eligible ids. Sorting makes the cut stable. It can
  differ from NeuralBench if that code's subject order is not sorted.
- W-bMAE: inside each recording, MAE on `[0, 40)`, `[40, 90)`,
  `[90, 300)`, `[300, 600]` with weights 10, 5, 3, 1, combined as
  `sum(w * MAE) / sum(w)` over non-empty bins, then the mean across
  recordings.
- The controlled run still takes the first 100 distinct train subjects,
  one train session each. It does not use the 40 test recordings.

`tracks/sleep_onset/objective.py` in this checkout is the older
unweighted scorer. The weighted recording mean lives in
`muse_experiments/evaluation/metrics.py`. The organizer benchmark files
were not changed.

## Mac

```bash
conda activate sleep-onset-2026
cd "/path/to/2026-competition"
python -m unittest \
  muse_experiments.tests.test_infrastructure \
  muse_experiments.tests.test_official_contract
```

LightGBM training is CPU-only. `device: auto` still records CUDA, then
Apple MPS, then CPU for torch checkpoints. Do not run NeuralBench
`--prepare` on this 16 GB machine.

The approved 100-recording run finished on 14 October 2026. Extraction
took 6.2 seconds and wrote 15,347 windows. Each LightGBM fit took 0.3
seconds. Re-running the same commands will not replace those files:

```bash
python -m muse_experiments.preprocessing.prepare_muse_features
python -m muse_experiments.training.train_lightgbm
python -m muse_experiments.training.train_lightgbm_weighted
python -m muse_experiments.evaluation.compare_lightgbm
```

Outputs:

- `muse_experiments/results/features/muse_features_100_pre_n2.csv`
- `muse_experiments/results/models/lightgbm_100_pre_n2_split33.joblib`
- `muse_experiments/results/models/lightgbm_100_pre_n2_split33_booster.json`
- `muse_experiments/results/models/lightgbm_weighted_100_pre_n2_split33.joblib`
- `muse_experiments/results/models/lightgbm_weighted_100_pre_n2_split33_booster.json`

An existing destination is not overwritten.

## AWS Linux

Use a new environment. Do not copy the Mac conda prefix. For torch
solvers, install the CUDA 12.6 wheels before `requirements.txt`, as in
`tools/Dockerfile` and `docs/team_setup.md`. Then:

```bash
pip install -r requirements.txt
pip install -r requirements-research.txt
```

The LightGBM fit itself does not need the GPU. The submission walker is
NumPy and SciPy and runs on CPU. Set `NEURALBENCH_MUSE_ROOT` if the Muse
download is not under `~/.neuralbench`.

## Codabench file

`muse_experiments/submission/submission.py` defines `Solver(CompetSolver)`.
`load_model` reads `lightgbm_booster.json` from `meta["submission_dir"]`.
`predict(X)` accepts `(B, 4, 640)`, including a torch tensor, reorders
channels when `ch_names` is present, and returns finite seconds of shape
`(B,)` clipped to 0–600. It uses only the current window. `reset_state`
clears a counter that is not a feature, so a `deepcopy` plus reset starts
the next recording clean.

LightGBM is not in `requirements.txt`. The JSON dump's leaf values
already include shrinkage; summing them matched `Booster.predict` in the
unit test. SciPy's Welch function is used for the same 28 features as
training. SciPy is not pinned directly; the worker installs MNE and
scikit-learn, which depend on it.

No upload ZIP is in the repo. Both corrected boosters exist. The constant
35-second baseline still has a lower validation W-bMAE than either
LightGBM model, so a model has not been chosen for upload. The old
checkpoint must not be packaged. After a model is chosen:

```bash
cp muse_experiments/results/models/<chosen>_booster.json \
  lightgbm_booster.json
zip -j muse_lightgbm_pre_n2.zip \
  muse_experiments/submission/submission.py \
  lightgbm_booster.json
```

The ZIP root must contain only those two names. Do not upload it from
here.

## Tests on 14 October 2026

`python -m unittest muse_experiments.tests.test_infrastructure muse_experiments.tests.test_official_contract`

36 tests passed in the `sleep-onset-2026` env (Python 3.12). They covered:

- windows stop at the first N2, and the target clips to 0–600
- offline microvolt features match the submission function; raw volts do not
- a subject with an official test session stays in training
- a two-recording hand calculation of W-bMAE, including the closed last bin
- the JSON tree walk matches LightGBM, including a missing value
- prediction shape `(B,)`, float64, finite, clipped, and independent of window order
- channel reordering, `deepcopy` plus `reset_state`, and a CPU torch tensor
- an MPS tensor when MPS is available
- a missing artifact and a 120 Hz meta value are rejected
- a sample ZIP has `submission.py` and `lightgbm_booster.json` at its root
- `requirements.txt` does not mention LightGBM, and the submission does not import it

The old feature table, joblib, and error CSV were not modified.

On the corrected 100-recording table, both models used the same 83/17
subject split (12,744 / 2,603 windows). Unweighted validation MAE was
157.07 s and W-bMAE was 188.55 s. Weighted validation MAE was 195.29 s
and W-bMAE was 66.56 s. The constant 35 s baseline W-bMAE was 63.31 s.
The training-median baseline (395 s) W-bMAE was 324.49 s. JSON tree walks
matched LightGBM on all 15,347 rows for both models, with a maximum
absolute difference of 0.
