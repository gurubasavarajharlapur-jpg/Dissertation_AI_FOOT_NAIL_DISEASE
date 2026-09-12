# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MSc dissertation: four-class classification of foot and nail images (`healthy`,
`nail_fungal`, `foot_wound`, `foot_ulcer`) using transfer learning, comparing
MobileNetV2 (primary, deployment candidate) against ResNet50 (accuracy
reference). Target context is phone-based screening in rural clinics, which is
why the small-model-vs-accuracy trade-off is the central experimental question.

The author is writing this up for a supervisor and a viva. Code is expected to
carry the *reasoning* behind methodological choices, not just the mechanics —
design comments that explain "why" are load-bearing here, not clutter.

## Commands

```bash
source .venv/bin/activate        # or: pip install -r requirements.txt

python src/download_data.py --list     # preview source datasets, no download
python src/download_data.py            # fetch into data/raw/
python src/inspect_data.py             # audit data/raw/; writes results/metrics/
python src/inspect_data.py --dir data/raw/figshare_nail --sample 500 --no-hash

python src/preprocessing.py            # Phase 2 — stub
python src/train.py --model mobilenetv2  # Phase 3 — stub
python src/evaluate.py                 # Phase 5 — stub
streamlit run src/prototype/app.py     # Phase 6 — stub
```

There is no test suite or linter configured yet.

## Architecture

**`src/config.py` is the hub.** Every path, hyper-parameter, split ratio, class
name and augmentation setting lives there and is imported by everything else.
Changing an experimental setting means editing that file, never a script. It
calls `validate()` at import time, so an inconsistent config fails immediately
rather than mid-training.

**Scripts are standalone entry points**, not a library. Each does
`sys.path.insert(0, <project root>)` then `from src import config`, so it runs
identically from the repo root, from `src/`, or from a notebook.

**Phase-staged development.** `preprocessing.py`, `train.py`, `evaluate.py` and
`prototype/app.py` currently exist as stubs whose docstrings specify the planned
design in detail. They print that plan and exit 1. The user wants to review and
understand each phase before it is built — **do not implement ahead of the
agreed phase**, and check in between phases.

Phase order: 1 setup (done) → 2 preprocessing → 3 MobileNetV2 → 4 ResNet50 →
5 evaluation → 6 prototype.

## Constraints that are easy to get wrong

- **Keras 3, not Keras 2.** TensorFlow 2.16+ bundles Keras 3, which removed
  `ImageDataGenerator`. Augmentation uses `tf.data` plus preprocessing layers
  (`RandomFlip`, `RandomRotation`, `RandomTranslation`, `RandomZoom`,
  `RandomBrightness`, `RandomContrast`). `config.AUGMENTATION` holds values in
  *those* layers' units (e.g. `rotation_factor` is a fraction of 2π, not
  degrees). Never pin `keras<3`.

- **`config.CLASS_NAMES` order is the label encoding** and the confusion-matrix
  axis order. Reordering it silently mislabels every already-trained model.

- **Normalisation is per-model, applied inside the model graph.** MobileNetV2
  wants `[-1, 1]`; ResNet50 wants caffe-style mean subtraction. Do not bake
  either into `data/processed/` — that would handicap one architecture and
  invalidate the comparison.

- **Both architectures must share the code path, splits and hyper-parameters.**
  The comparison is only meaningful as a controlled experiment.

- **The test split is evaluated exactly once, at the end.** Model selection and
  early stopping use validation only.

- **Report macro-averaged F1 and per-class recall**, not accuracy alone — the
  classes are imbalanced and a missed ulcer is the costly error.

- **Folder → class mapping is decided by a human** after reading the
  `inspect_data.py` report. Do not infer it from folder names.

## Data

`data/`, `models/` and `results/` contents are git-ignored. The source datasets
(Mendeley `hsj38fwnvr` v3, Figshare `5398573`) are not redistributable here and
must be cited in the dissertation.

Notebooks are Colab-compatible: the first cell detects Colab, mounts Drive,
clones the repo, and symlinks `data/raw/` to Drive.

## Git

Development happens on `claude/foot-nail-disease-ai-fyrdsb`. The user wants
**incremental commits after each meaningful step**, not one large commit, so
progress is legible in the history.
