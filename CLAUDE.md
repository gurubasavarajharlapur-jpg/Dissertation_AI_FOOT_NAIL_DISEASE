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

python src/extract_montages.py         # tile the Figshare contact sheets
python src/preprocessing.py            # build data/processed/ + the split CSVs
python src/preprocessing.py --dry-run  # show what it would do, write nothing
python src/train.py --model mobilenetv2  # two-stage transfer learning
python src/train.py --model resnet50
python src/evaluate.py                 # test metrics, Grad-CAM, comparison
python src/calibrate.py                # temperature scaling + referral threshold
python src/ablate_border.py --model mobilenetv2   # occlusion confound test
python src/compare_models.py           # paired significance, from saved CSVs
python src/audit_triage.py             # what the triage policy does to the test set
python src/sweep_thresholds.py         # tune triage thresholds on VALIDATION
streamlit run src/prototype/app.py     # two-tab screening prototype

python src/colab_sync.py status        # Colab: what is saved to Drive
python src/colab_sync.py save --what models results
```

`compare_models.py`, `audit_triage.py` and `sweep_thresholds.py` need neither
GPU nor TensorFlow — they read `results/metrics/<model>_predictions.csv`, which carries the full
probability vector per image, so any policy layered on the classifier can be
audited and re-tuned in seconds without re-running inference.

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

**All six phases are built** (1 setup → 2 preprocessing → 3 MobileNetV2 →
4 ResNet50 → 5 evaluation → 6 prototype). Both models are trained on the final
class × source stratified split and evaluated in one run. What remains is
evidence-gathering and dissertation writing, not implementation.

**Check in before starting anything substantial.** The user reviews each step
before it is built and has asked for this explicitly. That applied to the phases
and it still applies now — do not add scope unasked.

**Results live in two documents, and both must stay true to the runs.**
`docs/findings_and_analysis.md` holds every measured figure with the reasoning
behind it, written to be quoted directly into the dissertation;
`docs/implementation_requirements.md` traces proposal requirements to code. When
a run produces new numbers, update them — and never soften or drop a measured
result. The framing can be results-forward; the figures are not negotiable.

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
