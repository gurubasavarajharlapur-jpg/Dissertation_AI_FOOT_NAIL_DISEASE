"""Fine-tune MobileNetV2 (primary) and ResNet50 (comparison) on four classes.

Implements WBS 4.1-4.7.

    python src/train.py --model mobilenetv2      # primary
    python src/train.py --model resnet50         # comparison
    python src/train.py --model all              # both, in sequence

Both architectures run through this one code path, on the same splits, with the
same hyper-parameters from src/config.py. Section 2.5.8 of the literature review
identifies inconsistent comparative evaluation as a gap in the existing work —
studies using different preprocessing and hyper-parameters per architecture,
which makes their comparisons uninterpretable. Sharing everything but the
backbone is what makes this a controlled experiment: any difference in the
results is attributable to the architecture itself.

Two-stage transfer learning (2.5.5 "feature extraction" then "fine-tuning"):

  Stage 1 - head training, config.EPOCHS_HEAD at config.LEARNING_RATE_HEAD.
      The backbone is frozen and only the new classifier trains. The head starts
      from random weights; training end-to-end immediately would push large,
      meaningless gradients back through the pretrained filters and destroy the
      very features being transferred.

  Stage 2 - fine-tuning, config.EPOCHS_FINETUNE at config.LEARNING_RATE_FINETUNE.
      The top config.FINETUNE_LAYERS[model] layers unfreeze at a learning rate
      100x lower, so general ImageNet features adapt to skin and nail texture
      without being erased. BatchNormalization layers stay frozen throughout:
      updating their statistics on a small medical dataset produces a
      train/validation gap that looks like a bug but is not.

Per-architecture input scaling lives inside each model's own graph, because
MobileNetV2 expects [-1, 1] and ResNet50 expects caffe-style mean subtraction.
The saved .keras file is therefore self-contained, and the prototype cannot feed
a model the wrong normalisation.

Outputs:
    models/<model>_best.keras
    results/metrics/<model>_history.json
    results/figures/<model>_curves.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE

# Backbone constructor and its matching input-scaling function.
ARCHITECTURES = {
    "mobilenetv2": (
        keras.applications.MobileNetV2,
        keras.applications.mobilenet_v2.preprocess_input,
    ),
    "resnet50": (
        keras.applications.ResNet50,
        keras.applications.resnet50.preprocess_input,
    ),
}


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow so a run can be reproduced."""
    keras.utils.set_random_seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_manifest(split: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        raise SystemExit(
            f"'{path}' not found. Run `python src/preprocessing.py` first."
        )
    frame = pd.read_csv(path)
    unknown = set(frame["label"]) - set(config.CLASS_NAMES)
    if unknown:
        raise SystemExit(f"manifest contains unknown labels: {sorted(unknown)}")
    return frame


def build_augmentation() -> keras.Sequential:
    """Training-only augmentation, from config.AUGMENTATION.

    Applied in the tf.data pipeline rather than inside the model, so it can
    never touch validation or test — those must stay a fixed, honest
    measurement.

    Colour augmentation is deliberately mild: nail and skin conditions are
    partly diagnosed by hue (yellowing in onychomycosis, the red/black of an
    ulcer bed), so aggressive shifts would destroy the signal being learned.
    """
    aug = config.AUGMENTATION
    layers: list = [
        keras.layers.RandomFlip(aug["flip_mode"]),
        keras.layers.RandomRotation(aug["rotation_factor"], fill_mode="nearest"),
        keras.layers.RandomTranslation(
            aug["translation_height"], aug["translation_width"], fill_mode="nearest"
        ),
        keras.layers.RandomZoom(aug["zoom_factor"], fill_mode="nearest"),
        keras.layers.RandomBrightness(aug["brightness_factor"], value_range=(0, 255)),
        keras.layers.RandomContrast(aug["contrast_factor"]),
    ]

    # Healthy nails are upscaled from ~102px tiles while onychomycosis images are
    # downscaled from several hundred, so sharpness correlates with class.
    # Randomised blur stops the model using it as a shortcut.
    if aug.get("blur_factor", 0) > 0:
        if hasattr(keras.layers, "RandomGaussianBlur"):
            layers.append(keras.layers.RandomGaussianBlur(factor=aug["blur_factor"]))
        else:
            print(
                "  NOTE: this Keras build has no RandomGaussianBlur, so blur "
                "augmentation is skipped. Sharpness remains a potential class cue."
            )
    return keras.Sequential(layers, name="augmentation")


def _decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=config.IMAGE_CHANNELS)
    image = tf.cast(image, tf.float32)
    image.set_shape((*config.IMAGE_SIZE, config.IMAGE_CHANNELS))
    return image, label


def build_dataset(split: str, shuffle: bool, augment: bool) -> tuple[tf.data.Dataset, int]:
    """One tf.data pipeline per split. Returns the dataset and its length."""
    frame = load_manifest(split)
    paths = [str(config.PROJECT_ROOT / p) for p in frame["path"]]
    labels = [config.CLASS_TO_INDEX[label] for label in frame["label"]]

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=config.RANDOM_SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(_decode, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(config.BATCH_SIZE)

    if augment:
        augmentation = build_augmentation()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE
        )
    return dataset.prefetch(AUTOTUNE), len(paths)


def class_weights(frame: pd.DataFrame) -> dict[int, float]:
    """Inverse-frequency weights, so missing a rare condition costs more.

    The right trade-off for a screening tool: a missed ulcer is far more costly
    than a false alarm, which only produces an unnecessary referral.
    """
    counts = frame["label"].value_counts()
    total = len(frame)
    weights = {}
    for name in config.CLASS_NAMES:
        count = int(counts.get(name, 0))
        if count == 0:
            raise SystemExit(f"class '{name}' has no training images")
        weights[config.CLASS_TO_INDEX[name]] = total / (config.NUM_CLASSES * count)
    return weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(model_name: str) -> tuple[keras.Model, keras.Model]:
    """Build the classifier. Returns (full model, backbone) for later unfreezing."""
    if model_name not in ARCHITECTURES:
        raise SystemExit(f"unknown model '{model_name}'. choose from {sorted(ARCHITECTURES)}")
    constructor, preprocess_input = ARCHITECTURES[model_name]

    backbone = constructor(
        include_top=False, weights="imagenet", input_shape=config.INPUT_SHAPE
    )
    backbone.trainable = False

    inputs = keras.Input(shape=config.INPUT_SHAPE, name="image")
    # Scaling lives inside the graph so the saved model is self-contained and
    # the two architectures cannot be fed each other's normalisation.
    x = preprocess_input(inputs)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dropout(config.DROPOUT_RATE, name="dropout")(x)
    outputs = keras.layers.Dense(
        config.NUM_CLASSES,
        activation="softmax",
        kernel_regularizer=keras.regularizers.l2(config.L2_REGULARIZATION),
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs, name=model_name)
    return model, backbone


def unfreeze_top(backbone: keras.Model, n_layers: int) -> int:
    """Unfreeze the top N backbone layers, leaving BatchNormalization frozen.

    BN layers keep running mean/variance estimated over ImageNet. Re-estimating
    them from small medical batches destabilises training and opens a
    train/validation gap that is easily mistaken for a bug.
    """
    backbone.trainable = True
    trainable = 0
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers[-n_layers:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable += 1
    return trainable


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def callbacks_for(model_name: str, stage: str) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(config.model_path(model_name)),
            monitor=config.MONITOR_METRIC,
            mode=config.MONITOR_MODE,
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor=config.MONITOR_METRIC,
            mode=config.MONITOR_MODE,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=config.MONITOR_METRIC,
            mode=config.MONITOR_MODE,
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def plot_curves(history: dict, model_name: str, boundary: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].plot(epochs, history["loss"], label="train", marker="o", ms=3)
    axes[0].plot(epochs, history["val_loss"], label="validation", marker="o", ms=3)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{model_name}: loss")

    axes[1].plot(epochs, history["accuracy"], label="train", marker="o", ms=3)
    axes[1].plot(epochs, history["val_accuracy"], label="validation", marker="o", ms=3)
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title=f"{model_name}: accuracy", ylim=(0, 1.02))

    for ax in axes:
        if boundary:
            ax.axvline(boundary + 0.5, color="crimson", ls="--", lw=1,
                       label="fine-tuning starts")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = config.FIGURES_DIR / f"{model_name}_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  curves -> {out}")


def train_one(model_name: str, epochs_head: int, epochs_finetune: int) -> dict:
    print("\n" + "=" * 70)
    print(f"TRAINING {model_name}")
    print("=" * 70)
    set_seeds(config.RANDOM_SEED)
    config.ensure_dirs()

    train_ds, n_train = build_dataset("train", shuffle=True, augment=True)
    val_ds, n_val = build_dataset("val", shuffle=False, augment=False)
    train_frame = load_manifest("train")

    weights = class_weights(train_frame) if config.USE_CLASS_WEIGHTS else None
    print(f"train={n_train}  val={n_val}  batch={config.BATCH_SIZE}")
    if weights:
        print("class weights: " + "  ".join(
            f"{config.CLASS_NAMES[i]}={w:.2f}" for i, w in sorted(weights.items())
        ))

    model, backbone = build_model(model_name)
    total_params = model.count_params()
    print(f"\nbackbone layers: {len(backbone.layers)}   total parameters: {total_params:,}")

    history: dict[str, list] = {}
    started = time.time()

    # --- Stage 1: head only -------------------------------------------------
    print(f"\n--- Stage 1: head training ({epochs_head} epochs @ lr={config.LEARNING_RATE_HEAD}) ---")
    compile_model(model, config.LEARNING_RATE_HEAD)
    stage1 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_head,
        # The tf.data pipeline already reshuffles each iteration; Keras would
        # otherwise warn that its own shuffle is being ignored.
        shuffle=False,
        class_weight=weights, callbacks=callbacks_for(model_name, "head"), verbose=2,
    )
    for key, values in stage1.history.items():
        history.setdefault(key, []).extend(values)
    boundary = len(history["loss"])

    # --- Stage 2: fine-tuning ----------------------------------------------
    n_unfrozen = unfreeze_top(backbone, config.FINETUNE_LAYERS[model_name])
    print(f"\n--- Stage 2: fine-tuning ({epochs_finetune} epochs @ "
          f"lr={config.LEARNING_RATE_FINETUNE}) ---")
    print(f"unfroze {n_unfrozen} of the top {config.FINETUNE_LAYERS[model_name]} backbone "
          f"layers (BatchNorm left frozen)")
    # Recompiling is required for the new trainable flags to take effect.
    compile_model(model, config.LEARNING_RATE_FINETUNE)
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    print(f"trainable parameters: {trainable:,} of {total_params:,}")

    stage2 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_finetune,
        shuffle=False,
        class_weight=weights, callbacks=callbacks_for(model_name, "finetune"), verbose=2,
    )
    for key, values in stage2.history.items():
        history.setdefault(key, []).extend(values)

    elapsed = time.time() - started
    best_epoch = int(np.argmin(history["val_loss"])) + 1
    summary = {
        "model": model_name,
        "history": history,
        "stage_boundary_epoch": boundary,
        "epochs_run": len(history["loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": float(min(history["val_loss"])),
        "best_val_accuracy": float(history["val_accuracy"][best_epoch - 1]),
        "total_parameters": int(total_params),
        "trainable_parameters_finetune": int(trainable),
        "training_seconds": elapsed,
        "train_images": n_train,
        "val_images": n_val,
        "class_weights": {config.CLASS_NAMES[i]: w for i, w in (weights or {}).items()},
        "seed": config.RANDOM_SEED,
        "batch_size": config.BATCH_SIZE,
    }

    config.history_path(model_name).write_text(json.dumps(summary, indent=2))
    plot_curves(history, model_name, boundary)

    print(f"\n{model_name}: best val_loss {summary['best_val_loss']:.4f} "
          f"(epoch {best_epoch}), val accuracy {summary['best_val_accuracy']:.4f}")
    print(f"trained in {elapsed / 60:.1f} min -> {config.model_path(model_name)}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default=config.PRIMARY_MODEL,
        choices=[*config.MODEL_NAMES, "all"],
        help=f"which architecture to train (default: {config.PRIMARY_MODEL})",
    )
    parser.add_argument("--epochs-head", type=int, default=config.EPOCHS_HEAD)
    parser.add_argument("--epochs-finetune", type=int, default=config.EPOCHS_FINETUNE)
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__} | "
          f"GPU: {[g.name for g in gpus] or 'none (CPU only — this will be slow)'}")

    targets = config.MODEL_NAMES if args.model == "all" else [args.model]
    for name in targets:
        train_one(name, args.epochs_head, args.epochs_finetune)

    print("\nNext: python src/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
