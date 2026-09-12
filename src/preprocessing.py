"""Clean, resize, split and augment the raw images into data/processed/.

STATUS: not yet implemented — this is PHASE 2 of the project plan.
Implemented after the raw data has been inspected, because the cleaning rules
and the folder-to-class mapping depend on what `src/inspect_data.py` reports.

Planned pipeline
----------------
1. MAP     Read a source-folder -> project-class mapping (the four classes in
           config.CLASS_NAMES) and apply it to the files found in data/raw/.
           The mapping is written down explicitly, not inferred, so it can be
           justified in the methodology chapter.

2. CLEAN   Drop, and log the reason for, every image that is:
             - unreadable or truncated
             - below config.MIN_IMAGE_DIMENSION on either side
             - below config.MIN_FILE_SIZE_BYTES
             - an exact duplicate of another image (keep the first occurrence)
           Cross-class duplicates are reported and left for a manual decision:
           the correct label is genuinely ambiguous and guessing would corrupt
           the ground truth.

3. CONVERT Force every image to 3-channel RGB. Greyscale and RGBA files would
           otherwise produce the wrong tensor shape for a pretrained backbone.

4. RESIZE  To config.IMAGE_SIZE (224x224), the resolution both MobileNetV2 and
           ResNet50 were pretrained at.

5. SPLIT   Stratified 70/15/15 train/val/test using config.RANDOM_SEED, so each
           class keeps its proportions in every split and the rarest condition
           cannot vanish from the test set. The split is written out as CSV
           manifests, making it a fixed, inspectable artefact rather than
           something recomputed per run.

6. AUGMENT Training split only, via Keras 3 preprocessing layers configured in
           config.AUGMENTATION. Validation and test are never augmented — they
           must stay a fixed, honest measurement.

Note on normalisation: pixel scaling is NOT baked into data/processed/.
MobileNetV2 expects inputs in [-1, 1] and ResNet50 expects caffe-style mean
subtraction, so each model applies its own `preprocess_input` inside its graph
at training time. Baking one model's scaling into the files on disk would
quietly handicap the other and invalidate the comparison.

Logging: image counts and per-class totals are printed after every step, so the
effect of each stage is visible and quotable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    print(__doc__)
    print(
        "This script is a placeholder for Phase 2 and does nothing yet.\n"
        "Run `python src/inspect_data.py` first and review its report."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
