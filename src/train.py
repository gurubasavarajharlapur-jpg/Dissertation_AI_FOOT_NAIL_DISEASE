"""Fine-tune MobileNetV2 (primary) and ResNet50 (comparison) on four classes.

STATUS: not yet implemented — this is PHASE 3 (MobileNetV2) and PHASE 4
(ResNet50) of the project plan.

    python src/train.py --model mobilenetv2
    python src/train.py --model resnet50

Planned design
--------------
Both architectures are trained through the SAME code path, on the SAME splits,
with the SAME hyper-parameters from src/config.py. That is what makes the
comparison a controlled experiment: any difference in the results is
attributable to the architecture, not to the training setup.

Two-stage transfer learning, per model:

  Stage 1 — head training (config.EPOCHS_HEAD, lr=config.LEARNING_RATE_HEAD)
    Load the ImageNet-pretrained backbone with `include_top=False` and freeze
    it. Attach a new head: GlobalAveragePooling2D -> Dropout -> Dense(4,
    softmax). Only the head trains. The head starts from random weights, and
    its large early gradients would otherwise flow back and destroy the
    pretrained filters this project depends on.

  Stage 2 — fine-tuning (config.EPOCHS_FINETUNE, lr=config.LEARNING_RATE_FINETUNE)
    Unfreeze the top config.FINETUNE_LAYERS[model] layers and continue at a
    learning rate ~100x lower. The early layers keep their general edge and
    texture detectors; the later, more task-specific layers adapt to skin and
    nail appearance. BatchNormalization layers stay frozen in inference mode —
    updating their statistics on a small medical dataset is a classic source of
    a train/validation gap that looks like a bug but is not.

Per-model input scaling: MobileNetV2 and ResNet50 ship different
`preprocess_input` functions ([-1,1] vs caffe mean-subtraction). Each is applied
inside its own model graph, so the saved .keras file is self-contained and the
prototype cannot accidentally feed it the wrong scaling.

Class imbalance: config.USE_CLASS_WEIGHTS weights the loss inversely to class
frequency, so missing a rare condition costs more than missing a common one —
the right trade-off for a screening tool.

Callbacks: EarlyStopping (restore best weights), ReduceLROnPlateau, and
ModelCheckpoint on config.MONITOR_METRIC.

Outputs:
  models/<model>_best.keras          best weights
  results/metrics/<model>_history.json   per-epoch loss and accuracy
  results/figures/<model>_curves.png     training curves
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    print(__doc__)
    print(
        "This script is a placeholder for Phases 3-4 and does nothing yet.\n"
        "Complete preprocessing (Phase 2) first."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
