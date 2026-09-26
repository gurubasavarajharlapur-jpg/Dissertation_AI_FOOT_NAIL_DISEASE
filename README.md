# AI-Based Early Detection and Classification of Foot and Nail Conditions Using Transfer Learning for Rural Healthcare

MSc Data Science and Artificial Intelligence dissertation, Middlesex University.
Four-class classification of foot and nail photographs using transfer learning,
comparing two architectures to quantify the accuracy against portability
trade-off that decides whether a screening tool can run on a phone.

**Headline result.** MobileNetV2 reached 98.32 percent accuracy on 1,248
held-out images against ResNet50's 98.96 percent. The difference is not
statistically significant (McNemar's exact test, p = 0.1516) and a paired
bootstrap bounds it at 1.4 accuracy points, while MobileNetV2 is a tenth of the
size and roughly twice as fast on CPU. On ten photographs taken on phones
outside all three source datasets, accuracy fell to 6 of 10. See
[RESULTS.md](RESULTS.md) for the full picture.

## Problem

Foot and nail conditions such as fungal nail infection, wounds and especially
diabetic foot ulcers are treatable when caught early and severe when they are
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
| Parameters | 2,263,108 | 23,595,908 |
| Size on disk | 21.78 MB | 210.55 MB |
| CPU inference, per image | 248.29 ms | 460.56 ms |
| Role | The deployment candidate, small enough for a mid-range phone | The accuracy reference, establishing what portability costs |

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
licence and citation before using it in the dissertation**, both must appear in
the methodology chapter.

## Project structure

```
.
├── data/
│   ├── raw/                     Downloaded datasets, untouched (git-ignored)
│   └── processed/               Cleaned, resized, split data (git-ignored)
├── models/                      Trained weights (git-ignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb      Colab-compatible data audit
│   ├── 02_run_pipeline.ipynb          End-to-end run: fetch, prepare, train
│   └── 03_regenerate_gradcam.ipynb    Rebuild the Grad-CAM figure
├── results/
│   ├── figures/                 Confusion matrices, curves, Grad-CAM (git-ignored)
│   └── metrics/                 Per-image predictions and JSON metrics (git-ignored)
├── docs/
│   ├── findings_and_analysis.md       Every measured figure, with its reasoning
│   ├── implementation_requirements.md Proposal requirements traced to code
│   └── figures/                       Result figures used in the dissertation
├── src/
│   ├── config.py                Every experimental setting, in one place
│   ├── download_data.py         Fetch the source datasets into data/raw/
│   ├── inspect_data.py          Audit data/raw/: counts, formats, sizes, quality
│   ├── extract_montages.py      Tile the contact sheets into single images
│   ├── preprocessing.py         Clean, resize, split by class and source
│   ├── train.py                 Two-stage transfer learning, either architecture
│   ├── evaluate.py              Test metrics, confusion matrices, Grad-CAM
│   ├── calibrate.py             Temperature scaling and the referral threshold
│   ├── ablate_border.py         Controlled occlusion test for dataset shortcuts
│   ├── compare_models.py        Paired significance tests from saved predictions
│   ├── audit_triage.py          What the referral policy does to the test set
│   ├── sweep_thresholds.py      Tune the escalation threshold on validation
│   ├── ood_test.py              Score a condition outside the four classes
│   ├── triage.py                Turn a prediction into a recommended action
│   ├── stats.py                 McNemar, paired bootstrap, Wilson intervals
│   ├── colab_sync.py            Persist work across Colab runtime restarts
│   └── prototype/app.py         Screening and batch evaluation interface
├── requirements.txt
├── RESULTS.md
└── README.md
```

`src/config.py` is the file to read first. Every hyper-parameter, path, split
ratio and augmentation setting is defined there and imported everywhere else, so
an experiment is fully described by one file. It validates itself on import, so
an inconsistent configuration fails immediately rather than part way through a
training run.

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
`requirements.txt`, same API, far smaller download.

This project targets **Keras 3** (bundled with TensorFlow 2.16+). Keras 3 removed
`ImageDataGenerator`; augmentation uses `tf.data` with preprocessing layers
instead. Tutorials using `ImageDataGenerator` are written for Keras 2 and will
not work here.

### Google Colab

Open `notebooks/01_data_exploration.ipynb` in Colab. Its first cell detects
Colab, mounts Drive, clones this repo and installs the requirements. The dataset
is symlinked to Drive so it survives the runtime being recycled.

The notebooks clone the `claude/foot-nail-disease-ai-fyrdsb` branch explicitly,
which is the development branch. `main` carries the same code.

## Usage

```bash
source .venv/bin/activate          # or: pip install -r requirements.txt

python src/download_data.py        # fetch the source datasets into data/raw/
python src/inspect_data.py         # audit what actually arrived
python src/extract_montages.py     # tile the contact sheets
python src/preprocessing.py        # build data/processed/ and the split files

python src/train.py --model mobilenetv2
python src/train.py --model resnet50
python src/evaluate.py             # test metrics, confusion matrices, Grad-CAM
python src/calibrate.py            # temperature scaling, referral threshold
python src/ablate_border.py --model mobilenetv2
python src/ood_test.py             # behaviour on a condition outside the classes

streamlit run src/prototype/app.py # the screening interface
```

`compare_models.py`, `audit_triage.py` and `sweep_thresholds.py` need neither a
GPU nor TensorFlow. They read the saved per-image predictions, which carry the
full probability vector for every image, so any policy layered on the classifier
can be re-audited and re-tuned in seconds without repeating inference.

## Status

Complete. All six phases are built, both architectures are trained and evaluated
on the final split, and the dissertation is written. `docs/findings_and_analysis.md`
records every measured figure with the reasoning behind it, and
`docs/implementation_requirements.md` traces each proposal requirement to the
code that satisfies it.

## Methodology notes

These are deliberate choices, each with a reason that belongs in the write-up.

**Why transfer learning.** Clinical foot/nail datasets run to a few thousand
images. Training a modern CNN from scratch on that much data overfits badly.
ImageNet-pretrained filters already detect edges, textures and colour gradients.
Those are exactly the low-level features that distinguish a healthy nail from a
thickened, discoloured one, so only the later, task-specific layers need to be
learned.

**Why two-stage training.** The new classifier head starts from random weights.
Training it end-to-end immediately would push large, meaningless gradients back
through the pretrained backbone and destroy the features being transferred. So
the backbone is frozen first while the head learns, then the top layers are
unfrozen at a learning rate one hundred times lower.

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
