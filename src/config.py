"""Central configuration for the whole project.

Every number that affects a reported result lives here rather than being
hard-coded in a script. Two reasons, both of which matter for the viva:

1. Reproducibility. One file fully describes an experiment, so a run can be
   repeated exactly and the settings can be quoted in the methodology chapter.
2. Fair comparison. MobileNetV2 and ResNet50 must be trained under identical
   conditions, otherwise any difference in their scores could be down to the
   training setup rather than the architecture. Sharing this config is what
   makes the comparison a controlled experiment.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. Everything is derived from the project root so the code runs the same
# way from the repo root, from src/, or from a notebook in notebooks/.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def ensure_dirs() -> None:
    """Create the output directories if they are missing."""
    for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Fixed seed for NumPy, Python's `random`, and TensorFlow. Used for the
# train/val/test split, weight initialisation of the classifier head, shuffling
# and augmentation. Changing it changes the results, so it is reported in the
# dissertation alongside the numbers.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Classification task
# ---------------------------------------------------------------------------
# The four target categories. ORDER IS SIGNIFICANT: the index of each class in
# this list is the integer label the network learns, and the row/column order of
# the confusion matrix. Never reorder this list after training a model, or the
# saved weights will be silently mislabelled.
CLASS_NAMES: list[str] = [
    "healthy",            # Healthy foot / nail
    "nail_fungal",        # Nail fungal infection (onychomycosis)
    "foot_wound",         # Foot wound / injury
    "foot_ulcer",         # Foot ulcer (e.g. diabetic foot ulcer)
]

# Human-readable labels for plots, reports and the prototype interface.
CLASS_DISPLAY_NAMES: dict[str, str] = {
    "healthy": "Healthy Foot/Nail",
    "nail_fungal": "Nail Fungal Infection",
    "foot_wound": "Foot Wound/Injury",
    "foot_ulcer": "Foot Ulcer",
}

NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS: dict[int, str] = {i: name for name, i in CLASS_TO_INDEX.items()}

# ---------------------------------------------------------------------------
# Source folder -> project class mapping
# ---------------------------------------------------------------------------
# Decided by hand after reading the src/inspect_data.py report, never inferred
# from folder names. Keys are path fragments matched against each image's path
# under data/raw/; the first match wins, so more specific patterns come first.
# A value of None means "recognised but deliberately excluded", which keeps the
# exclusion explicit and auditable rather than silent.
FOLDER_TO_CLASS: dict[str, str | None] = {
    # --- Mendeley hsj38fwnvr v3 (whole-foot photographs) -------------------
    "mendeley_foot/Normal": "healthy",
    "mendeley_foot/wound_main": "foot_wound",

    # --- FUSeg / AZH chronic wound (Wang et al., 2020) --------------------
    # Every image in this dataset is a foot ulcer; the split sub-folders are
    # the publisher's own segmentation-challenge splits, not ours.
    "ulcer_fuseg": "foot_ulcer",

    # --- Figshare 5398573, per-image folders ------------------------------
    "figshare_nail/datasets (B1, B2, C, D, E)": None,  # overridden below
    # --- Figshare 5398573, tiles recovered from the montage sheets --------
    "figshare_nail_tiles/healthy": "healthy",
}

# Checked before FOLDER_TO_CLASS so the Figshare per-image sub-folders resolve
# correctly regardless of which parent pattern would otherwise match.
LEAF_FOLDER_TO_CLASS: dict[str, str | None] = {
    "onychomycosis": "nail_fungal",
    # Nail dystrophy is a real condition but not one of the four classes fixed
    # by the proposal (Methods; 5.4), so it is excluded rather than reassigned.
    "naildystrophy": None,
}

# Sources whose images are letterboxed onto a black background. The FUSeg and
# Medetec ulcer images were cropped to the wound and zero-padded to square by
# their publishers, leaving 25-32% of every image pure black — a marking no
# other class carries. Left in place, a model would learn "black border =>
# ulcer" and score near-perfectly without learning anything about ulcers.
# Images from these sources are cropped back to their non-black content.
CROP_BLACK_BORDERS = ("ulcer_fuseg",)
# A pixel at or below this intensity counts as padding rather than dark tissue.
BLACK_THRESHOLD = 12

# ---------------------------------------------------------------------------
# Image settings
# ---------------------------------------------------------------------------
# 224x224 is the resolution MobileNetV2 and ResNet50 were pretrained on at
# ImageNet. Feeding them the same size means the pretrained filters see the
# scale of detail they were trained to recognise, which is the whole point of
# transfer learning.
IMAGE_SIZE: tuple[int, int] = (224, 224)
IMAGE_CHANNELS = 3
INPUT_SHAPE: tuple[int, int, int] = (*IMAGE_SIZE, IMAGE_CHANNELS)

# File extensions treated as images when scanning data/raw/.
VALID_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# Cleaning thresholds applied during preprocessing (see src/preprocessing.py).
# Images smaller than this in either dimension are too low-resolution to carry
# diagnostic detail once resized, and are dropped rather than upscaled.
MIN_IMAGE_DIMENSION = 64
# Files below this size are almost always truncated downloads or placeholders.
MIN_FILE_SIZE_BYTES = 1024

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
# Stratified split: each class keeps these proportions, so a rare class is not
# accidentally absent from the test set. Must sum to 1.0.
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ---------------------------------------------------------------------------
# Training hyper-parameters (shared by both architectures)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32

# Transfer learning runs in two stages:
#   Stage 1 (head training)  - backbone frozen, train only the new classifier.
#                              A random head would otherwise send large, noisy
#                              gradients through the pretrained filters and
#                              destroy the features being transferred.
#   Stage 2 (fine-tuning)    - unfreeze the top of the backbone at a much lower
#                              learning rate so the general ImageNet features
#                              adapt to skin/nail texture without being erased.
EPOCHS_HEAD = 15
EPOCHS_FINETUNE = 25

LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 1e-5

# How many layers at the top of the backbone to unfreeze in stage 2. Set per
# architecture because the two networks have very different depths.
FINETUNE_LAYERS = {
    "mobilenetv2": 30,   # MobileNetV2 has 154 layers
    "resnet50": 30,      # ResNet50 has 175 layers
}

DROPOUT_RATE = 0.3
L2_REGULARIZATION = 1e-4

# Stop when validation loss has not improved for this many epochs, and restore
# the best weights. Guards against overfitting on a small medical dataset.
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5

# Metric used to select the best checkpoint. Validation accuracy is a poor
# choice on imbalanced data because a model that ignores the rarest condition
# can still score highly; macro-averaged behaviour is preferred, and val_loss is
# the closest proxy available as a stock Keras checkpoint monitor.
MONITOR_METRIC = "val_loss"
MONITOR_MODE = "min"

# Counteract class imbalance by weighting the loss inversely to class frequency,
# so the model is penalised more for missing a rare condition.
USE_CLASS_WEIGHTS = True

# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------
# Applied to the training split only - never to validation or test, which must
# stay a fixed, honest measurement.
#
# Deliberately conservative on colour: nail and skin conditions are diagnosed
# partly by hue (yellowing in onychomycosis, the red/black of an ulcer bed), so
# aggressive colour shifts would destroy the signal the model should learn.
#
# These values feed Keras 3 preprocessing layers (RandomFlip, RandomRotation,
# RandomTranslation, RandomZoom, RandomBrightness, RandomContrast), which run on
# the GPU as part of the tf.data pipeline. Keras 3 removed the older
# `ImageDataGenerator`, so any tutorial using that class targets Keras 2.
AUGMENTATION = {
    # Fraction of 2*pi. 0.055 ~= 20 degrees; photos are taken at odd angles.
    "rotation_factor": 0.055,
    # Fraction of image height/width the subject may be shifted by.
    "translation_height": 0.15,
    "translation_width": 0.15,
    # +/- fraction of the original size; distance from the foot varies.
    "zoom_factor": 0.15,
    # Left foot vs right foot are mirror images, so this is a real symmetry.
    # "horizontal" only - an upside-down clinical photo is not realistic.
    "flip_mode": "horizontal",
    # Lighting differs between clinics; kept small to preserve diagnostic hue.
    "brightness_factor": 0.15,
    "contrast_factor": 0.15,
    # Healthy nail images come from ~102px montage tiles and are upscaled to
    # 224, while the onychomycosis images are downscaled from several hundred
    # pixels. Sharpness therefore correlates with class, and a model can exploit
    # that instead of learning pathology. Randomised blur removes it as a
    # reliable cue. Set to 0.0 to disable.
    "blur_factor": 0.25,
}

# ---------------------------------------------------------------------------
# Models to train and compare
# ---------------------------------------------------------------------------
# MobileNetV2 is the primary model: ~3.5M parameters and low inference cost, so
# it can plausibly run on a mid-range phone at a rural health post, which is the
# deployment story of this dissertation. ResNet50 (~25M parameters) is the
# comparison model - a heavier, more accurate architecture that establishes what
# accuracy is being traded away for that portability.
PRIMARY_MODEL = "mobilenetv2"
COMPARISON_MODEL = "resnet50"
MODEL_NAMES: list[str] = [PRIMARY_MODEL, COMPARISON_MODEL]


def model_path(model_name: str) -> Path:
    """Where a trained model's weights are saved."""
    return MODELS_DIR / f"{model_name}_best.keras"


def history_path(model_name: str) -> Path:
    """Where a model's per-epoch training history is saved."""
    return METRICS_DIR / f"{model_name}_history.json"


def metrics_path(model_name: str) -> Path:
    """Where a model's final evaluation metrics are saved."""
    return METRICS_DIR / f"{model_name}_metrics.json"


def validate() -> None:
    """Fail fast on a self-inconsistent config rather than mid-training."""
    total = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"TRAIN/VAL/TEST splits must sum to 1.0, got {total}")
    if len(set(CLASS_NAMES)) != len(CLASS_NAMES):
        raise ValueError(f"CLASS_NAMES contains duplicates: {CLASS_NAMES}")
    if set(CLASS_DISPLAY_NAMES) != set(CLASS_NAMES):
        raise ValueError("CLASS_DISPLAY_NAMES must have exactly one entry per class")
    missing = set(MODEL_NAMES) - set(FINETUNE_LAYERS)
    if missing:
        raise ValueError(f"FINETUNE_LAYERS is missing an entry for: {sorted(missing)}")

    for name, mapping in (("FOLDER_TO_CLASS", FOLDER_TO_CLASS),
                          ("LEAF_FOLDER_TO_CLASS", LEAF_FOLDER_TO_CLASS)):
        bad = {k: v for k, v in mapping.items() if v is not None and v not in CLASS_NAMES}
        if bad:
            raise ValueError(
                f"{name} maps to labels that are not in CLASS_NAMES: {bad}. "
                f"valid classes: {CLASS_NAMES}"
            )


validate()
