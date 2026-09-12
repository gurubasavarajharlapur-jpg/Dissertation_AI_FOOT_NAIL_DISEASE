# AI-Based Early Detection and Classification of Foot and Nail Conditions Using Transfer Learning for Rural Healthcare

MSc dissertation project. Four-class classification of foot and nail images
using transfer learning, with two architectures compared to quantify the
accuracy-vs-portability trade-off that matters for deployment in rural clinics.

## Problem

Foot and nail conditions — fungal nail infection, wounds, and especially
diabetic foot ulcers — are treatable when caught early and severe when they are
not. In rural settings the specialist who would recognise them is often absent.
A model small enough to run on a health worker's phone could act as a screening
aid, flagging cases that need referral.

That deployment constraint drives the architecture comparison below.

## Classification task

| # | Class (`config.CLASS_NAMES`) | Description |
|---|------------------------------|-------------|
| 0 | `healthy`      | Healthy Foot/Nail |
| 1 | `nail_fungal`  | Nail Fungal Infection (onychomycosis) |
| 2 | `foot_wound`   | Foot Wound/Injury |
| 3 | `foot_ulcer`   | Foot Ulcer (e.g. diabetic foot ulcer) |

The order of this list is the integer label the network learns and the row/column
order of the confusion matrix. It must not be changed after a model is trained.

## Models

| | MobileNetV2 (primary) | ResNet50 (comparison) |
|---|---|---|
| Parameters | ~3.5M | ~25M |
| Role | The deployment candidate — small enough for a mid-range phone | The accuracy reference — establishes what portability costs |

Both are ImageNet-pretrained and fine-tuned through the **same code path, on the
same splits, with the same hyper-parameters**. That is what makes this a
controlled comparison: any difference in the results is attributable to the
architecture rather than to the training setup.

## Data sources

| Source | Contents | Link |
|---|---|---|
| Mendeley Data `hsj38fwnvr` v3 | Foot imagery (wound / ulcer) | https://data.mendeley.com/datasets/hsj38fwnvr/3 |
| Figshare `5398573` | Onychomycosis (nail fungal) training + validation sets | https://figshare.com/articles/dataset/5398573 |

Neither dataset is committed to this repository: they are large, and
redistribution is governed by their own licences. **Record each dataset's
licence and citation before using it in the dissertation** — both must appear in
the methodology chapter.

## Project structure

```
.
├── data/
│   ├── raw/              Downloaded datasets, untouched (git-ignored)
│   └── processed/        Cleaned, resized, split data (git-ignored)
├── models/               Trained weights (git-ignored)
├── notebooks/
│   └── 01_data_exploration.ipynb    Colab-compatible EDA
├── results/
│   ├── figures/          Confusion matrices, training curves, plots
│   └── metrics/          JSON/CSV metrics and data inventories
├── src/
│   ├── config.py         Every experimental setting, in one place
│   ├── download_data.py  Fetch the source datasets into data/raw/
│   ├── inspect_data.py   Audit data/raw/: counts, formats, sizes, quality
│   ├── preprocessing.py  Clean, resize, split, augment      [Phase 2]
│   ├── train.py          Fine-tune MobileNetV2 and ResNet50 [Phases 3-4]
│   ├── evaluate.py       Metrics and confusion matrices     [Phase 5]
│   └── prototype/
│       └── app.py        Streamlit upload -> prediction UI  [Phase 6]
├── requirements.txt
└── README.md
```

`src/config.py` is the file to read first. Every hyper-parameter, path, split
ratio and augmentation setting is defined there and imported everywhere else, so
an experiment is fully described by one file.

## Setup

Requires Python 3.10 or newer (tested on 3.11; Colab currently ships 3.13).

```bash
git clone https://github.com/gurubasavarajharlapur-jpg/Dissertation_AI_FOOT_NAIL_DISEASE.git
cd Dissertation_AI_FOOT_NAIL_DISEASE

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On a machine without an NVIDIA GPU, swap `tensorflow` for `tensorflow-cpu` in
`requirements.txt` — same API, far smaller download.

This project targets **Keras 3** (bundled with TensorFlow 2.16+). Keras 3 removed
`ImageDataGenerator`; augmentation uses `tf.data` with preprocessing layers
instead. Tutorials using `ImageDataGenerator` are written for Keras 2 and will
not work here.

### Google Colab

Open `notebooks/01_data_exploration.ipynb` in Colab. Its first cell detects
Colab, mounts Drive, clones this repo and installs the requirements. The dataset
is symlinked to Drive so it survives the runtime being recycled.

The notebook clones the `claude/foot-nail-disease-ai-fyrdsb` branch explicitly.
The project code is not on `main` yet, so a plain `git clone` checks out an
empty repository.

## Usage

```bash
# 1. Fetch the datasets into data/raw/
python src/download_data.py --list      # preview without downloading
python src/download_data.py

# 2. Audit what actually arrived
python src/inspect_data.py
```

Steps 3 onward are implemented phase by phase — see the plan below.

## Project plan

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Environment, project structure, config, data download + inspection | **Done** |
| 2 | `preprocessing.py` — clean, resize, split, augment | Not started |
| 3 | `train.py` — MobileNetV2 (primary) | Not started |
| 4 | `train.py` — ResNet50 (comparison) | Not started |
| 5 | `evaluate.py` — metrics, confusion matrices, model comparison | Not started |
| 6 | `prototype/app.py` — image upload → prediction interface | Not started |

## Methodology notes

These are deliberate choices, each with a reason that belongs in the write-up.

**Why transfer learning.** Clinical foot/nail datasets run to a few thousand
images. Training a modern CNN from scratch on that much data overfits badly.
ImageNet-pretrained filters already detect edges, textures and colour gradients —
exactly the low-level features that distinguish a healthy nail from a thickened,
discoloured one — so only the later, task-specific layers need to be learned.

**Why two-stage training.** The new classifier head starts from random weights.
Training it end-to-end immediately would push large, meaningless gradients back
through the pretrained backbone and destroy the features being transferred. So
the backbone is frozen first while the head learns, then the top layers are
unfrozen at a learning rate ~100× lower.

**Why the test split is touched once.** Model selection, early stopping and
hyper-parameter choices all use the validation split. The test split is evaluated
exactly once, at the end. Selecting anything against test scores would make them
an optimistic estimate rather than a measurement.

**Why macro-F1 rather than accuracy.** The classes are not balanced. A model
that never predicts the rarest condition can still post a high accuracy while
being useless for the case that matters most. Macro-averaging weights all four
classes equally.

**Why recall is the priority metric.** For a screening tool, a missed ulcer is
far more costly than a false alarm: the false alarm produces an unnecessary
referral, the miss produces an untreated ulcer.

**Why normalisation is not baked into `data/processed/`.** MobileNetV2 expects
inputs in `[-1, 1]`; ResNet50 expects caffe-style mean subtraction. Each model
applies its own `preprocess_input` inside its own graph. Writing one model's
scaling to disk would quietly handicap the other and invalidate the comparison.

**Why the split is saved as a manifest.** The train/val/test assignment is
written out as CSV rather than recomputed each run, so every model is trained and
evaluated on provably identical data.

## Reproducibility

`config.RANDOM_SEED = 42` seeds Python, NumPy and TensorFlow, covering the split,
head initialisation, shuffling and augmentation. Full determinism on GPU also
requires `TF_DETERMINISTIC_OPS=1`, at some cost in speed.

## Disclaimer

This is a research prototype built for an MSc dissertation. It is **not a medical
device** and must not be used for clinical diagnosis.
