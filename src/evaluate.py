"""Evaluate the trained models on the held-out test split and compare them.

STATUS: not yet implemented — this is PHASE 5 of the project plan.

    python src/evaluate.py                      # both models, side by side
    python src/evaluate.py --model mobilenetv2

Planned metrics
---------------
Reported per model, on the test split only — which is touched exactly once, at
the end. Any model selection done against test scores would make them
meaningless as an estimate of real-world performance.

  Accuracy           Overall correctness. Reported, but NOT relied upon: on an
                     imbalanced set a model that never predicts the rarest
                     condition can still post a high accuracy.
  Precision          Per class and macro-averaged. Of the images called
                     "ulcer", how many were ulcers? Low precision means false
                     alarms and unnecessary referrals.
  Recall             Per class and macro-averaged. Of the actual ulcers, how
                     many were caught? This is the critical metric for a rural
                     screening tool: a missed ulcer is far more costly than a
                     false alarm.
  F1-score           Harmonic mean of the two. The macro-averaged F1 is the
                     headline comparison figure, because it weights all four
                     classes equally regardless of how many images each has.
  Confusion matrix   Which conditions get mistaken for which. Clinically the
                     most informative output: wound/ulcer confusion matters far
                     more than either being confused with healthy.

Also produced:
  - Per-class support counts, so a metric computed over 6 test images is not
    read as though it were computed over 600.
  - A side-by-side MobileNetV2 vs ResNet50 table, plus parameter count and
    inference time per image — the portability argument of the dissertation
    rests on that trade-off, not on accuracy alone.

Outputs:
  results/metrics/<model>_metrics.json
  results/metrics/model_comparison.csv
  results/figures/<model>_confusion_matrix.png
  results/figures/model_comparison.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    print(__doc__)
    print(
        "This script is a placeholder for Phase 5 and does nothing yet.\n"
        "Train at least one model (Phases 3-4) first."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
