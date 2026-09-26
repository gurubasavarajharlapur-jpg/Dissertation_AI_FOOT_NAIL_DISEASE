"""Test whether the models rely on the image border rather than the lesion.

    python src/ablate_border.py                    # all trained models
    python src/ablate_border.py --model mobilenetv2 --width 20

Why this exists. Border brightness differs systematically by class in this
dataset: measured over the test split, healthy images average about 170 at the
edges against 107-127 for the diseased classes, and that gap holds *within* a
single source (Mendeley healthy 170 vs Mendeley wound 127). The cause is
probably framing — wound and ulcer photographs are close-ups that fill the
frame, healthy feet are shot further back with background visible — but a
shortcut does not have to be deliberate to be a shortcut. A model could learn
"bright edge => healthy" and score well without reading any pathology.

Grad-CAM suggests it does not: attention falls on toes and nail plates. That is
qualitative evidence from a handful of images. This measures it.

Two ablations, each applied identically to every class so the manipulation
itself cannot favour one:

  BORDER MASKED   The outer `width` pixels are replaced with neutral grey.

  INTERIOR CONTROL
                  An equal AREA of interior is replaced instead. This is what
                  makes the border arm interpretable: masking the outer band
                  removes the border's information *and* confronts the model
                  with a large grey frame it never saw in training, and a model
                  degrades on any such occlusion. Only the difference between
                  the two arms is attributable to the border's content.

  CENTRE MASKED   Everything except the outer `width` pixels is replaced.
                  Only the border survives. Accuracy at or near the
                  majority-class rate means the border alone carries no usable
                  class information. Well above it means it does.

The second is the sharper test. A model can be robust to losing the border while
the border still carries enough signal to classify on its own.

Inference only — no retraining, and the test split is read but nothing is
selected from it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402

# Mid grey. A constant, rather than each image's own mean, so the fill cannot
# reintroduce the brightness difference being tested for.
FILL = 128.0


def _decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=config.IMAGE_CHANNELS)
    image = tf.cast(image, tf.float32)
    image.set_shape((*config.IMAGE_SIZE, config.IMAGE_CHANNELS))
    return image, label


def build_dataset(frame: pd.DataFrame, mask: str, width: int) -> tf.data.Dataset:
    """Stream the test split, applying one mask on the fly.

    Materialising all 1,248 images as float32 costs ~750MB, and a separate
    full-size copy per ablation arm pushed a CPU runtime into ~3GB of arrays and
    made each arm take minutes. Masking inside the pipeline keeps one batch in
    memory at a time.
    """
    paths = [str(config.PROJECT_ROOT / p) for p in frame["path"]]
    labels = [config.CLASS_TO_INDEX[c] for c in frame["label"]]
    dataset = (
        tf.data.Dataset.from_tensor_slices((paths, labels))
        .map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    )
    if mask != "none":
        dataset = dataset.map(
            lambda image, label: (_apply_mask(image, mask, width), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def _keep_mask(mask: str, width: int) -> np.ndarray:
    """(H, W, 1) array: 1 where the pixel survives, 0 where it is replaced."""
    side = config.IMAGE_SIZE[0]
    keep = np.ones((side, side, 1), dtype=np.float32)
    border = np.zeros((side, side, 1), dtype=np.float32)
    border[:width], border[-width:], border[:, :width], border[:, -width:] = 1, 1, 1, 1

    if mask == "border":
        return keep - border
    if mask == "centre":
        return border
    if mask == "interior":
        # Equal AREA to the border ring, but sited inside it, so the two arms
        # remove the same number of pixels and differ only in location.
        border_area = side**2 - (side - 2 * width) ** 2
        outer = side - 2 * width
        inner = int(round(np.sqrt(max(outer**2 - border_area, 0))))
        thickness = max((outer - inner) // 2, 1)
        lo, hi = width, side - width
        ring = np.zeros((side, side, 1), dtype=np.float32)
        ring[lo:lo + thickness, lo:hi] = 1
        ring[hi - thickness:hi, lo:hi] = 1
        ring[lo:hi, lo:lo + thickness] = 1
        ring[lo:hi, hi - thickness:hi] = 1
        return keep - ring
    raise ValueError(f"unknown mask '{mask}'")


def _apply_mask(image: tf.Tensor, mask: str, width: int) -> tf.Tensor:
    keep = tf.constant(_keep_mask(mask, width))
    return image * keep + FILL * (1.0 - keep)


def load_images(frame: pd.DataFrame) -> np.ndarray:
    """A handful of images, for the example figure only."""
    return np.stack([
        np.asarray(Image.open(config.PROJECT_ROOT / p).convert("RGB"), dtype=np.float32)
        for p in frame["path"]
    ])


def mask_border(images: np.ndarray, width: int) -> np.ndarray:
    """Replace the outer `width` pixels with grey, keeping the centre."""
    out = images.copy()
    out[:, :width, :, :] = FILL
    out[:, -width:, :, :] = FILL
    out[:, :, :width, :] = FILL
    out[:, :, -width:, :] = FILL
    return out


def mask_interior_ring(images: np.ndarray, width: int) -> np.ndarray:
    """Grey out an interior ring of the same pixel area as the border ring.

    The control the border test needs. Masking the outer band removes two things
    at once: whatever information the border carried, and the model's assumption
    that images do not have grey frames. A drop therefore cannot be read as
    reliance on the border — a model trained without grey borders degrades on
    any large unfamiliar occlusion.

    Removing an equal area from the interior separates them. If the two cost
    about the same, the loss is the perturbation. If the border costs materially
    more, the border carried information.
    """
    side = images.shape[1]
    border_area = side**2 - (side - 2 * width) ** 2

    outer = side - 2 * width                       # ring sits inside the border
    inner_sq = max(outer**2 - border_area, 0)
    inner = int(round(np.sqrt(inner_sq)))
    thickness = max((outer - inner) // 2, 1)

    lo, hi = width, side - width                   # interior region bounds
    out = images.copy()
    out[:, lo:lo + thickness, lo:hi, :] = FILL
    out[:, hi - thickness:hi, lo:hi, :] = FILL
    out[:, lo:hi, lo:lo + thickness, :] = FILL
    out[:, lo:hi, hi - thickness:hi, :] = FILL
    return out


def mask_centre(images: np.ndarray, width: int) -> np.ndarray:
    """Replace everything except the outer `width` pixels, keeping the border."""
    out = np.full_like(images, FILL)
    out[:, :width, :, :] = images[:, :width, :, :]
    out[:, -width:, :, :] = images[:, -width:, :, :]
    out[:, :, :width, :] = images[:, :, :width, :]
    out[:, :, -width:, :] = images[:, :, -width:, :]
    return out


def score(model, dataset: tf.data.Dataset, y_true: np.ndarray) -> dict:
    probs = model.predict(dataset, verbose=0)
    y_pred = probs.argmax(axis=1)
    labels = list(range(config.NUM_CLASSES))
    per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": {config.CLASS_NAMES[i]: float(per_class[i]) for i in labels},
        "mean_confidence": float(probs.max(axis=1).mean()),
    }


def plot_examples(images: np.ndarray, width: int, model_name: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, arr, title in (
        (axes[0], images[0], "original"),
        (axes[1], mask_border(images[:1], width)[0], f"border masked ({width}px)"),
        (axes[2], mask_centre(images[:1], width)[0], f"centre masked ({width}px)"),
    ):
        ax.imshow(arr.astype(np.uint8))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{model_name}: border ablation inputs", fontsize=11)
    fig.tight_layout()
    out = config.FIGURES_DIR / f"{model_name}_ablation_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def run(model_name: str, width: int, frame: pd.DataFrame, y_true: np.ndarray) -> dict:
    print("\n" + "=" * 74)
    print(f"{model_name.upper()} — BORDER ABLATION ({width}px, {len(y_true)} test images)")
    print("=" * 74)

    model = keras.models.load_model(config.model_path(model_name))
    results = {}
    for key, mask in (("baseline", "none"), ("border_masked", "border"),
                      ("interior_control", "interior"), ("centre_masked", "centre")):
        print(f"  scoring {key}…", flush=True)
        results[key] = score(model, build_dataset(frame, mask, width), y_true)

    # Predicting the commonest class every time is the floor any result must beat.
    majority = float(pd.Series(y_true).value_counts(normalize=True).max())

    ARMS = (("baseline", "unmodified"), ("border_masked", "border masked"),
            ("interior_control", "interior control"), ("centre_masked", "centre masked"))

    print(f"{'':<18}{'accuracy':>10}{'macro F1':>11}{'mean conf':>11}")
    for key, label in ARMS:
        r = results[key]
        print(f"{label:<18}{r['accuracy']:>10.4f}{r['macro_f1']:>11.4f}"
              f"{r['mean_confidence']:>11.4f}")
    print(f"{'majority class':<18}{majority:>10.4f}{'-':>11}{'-':>11}")

    print("\nper-class recall:")
    print(f"{'':<18}" + "".join(f"{c[:11]:>13}" for c in config.CLASS_NAMES))
    for key, label in ARMS:
        row = results[key]["per_class_recall"]
        print(f"{label:<18}" + "".join(f"{row[c]:>13.4f}" for c in config.CLASS_NAMES))

    drop = results["baseline"]["accuracy"] - results["border_masked"]["accuracy"]
    control_drop = results["baseline"]["accuracy"] - results["interior_control"]["accuracy"]
    excess = drop - control_drop
    border_only = results["centre_masked"]["accuracy"]

    print("\ninterpretation:")

    # Primary evidence. Unlike the masked arms this one is free of the
    # perturbation confound: it asks directly whether the border predicts the
    # class, rather than whether removing it hurts.
    margin = border_only - majority
    print(f"  [border alone]  With only the border visible, accuracy is {border_only:.4f} "
          f"against a")
    print(f"  majority-class floor of {majority:.4f} ({margin:+.4f}).")
    if margin < 0.05:
        print("  The border alone carries no usable class information.")
    else:
        print("  The border alone is predictive of the class. Report this whether or not")
        print("  the full model turns out to depend on it.")

    # Secondary. Both masked arms remove the same area and are equally
    # unfamiliar, so their difference isolates the border's content — but only
    # while both retain some signal. Once both collapse to chance there is no
    # headroom left for a difference to appear in, and a null result there means
    # nothing.
    floor = majority + 0.05
    collapsed = (results["border_masked"]["accuracy"] < floor
                 and results["interior_control"]["accuracy"] < floor)
    print(f"\n  [occlusion]     Masking the border costs {drop:.4f}; masking an equal area")
    print(f"  of interior costs {control_drop:.4f}; difference {excess:+.4f}.")
    if collapsed:
        print("  Both arms fall to near chance, so this comparison has no headroom and")
        print("  cannot be interpreted. Rely on the border-alone result above.")
    elif abs(excess) * len(y_true) < 15:
        print(f"  That difference is {abs(excess) * len(y_true):.0f} image(s) out of "
              f"{len(y_true)} — within noise.")
    elif excess < 0.02:
        print("  The border is worth no more than any comparable region.")
    elif excess < 0.10:
        print("  The border is worth modestly more than a comparable region — some")
        print("  reliance, worth reporting as a limitation.")
    else:
        print("  The border is worth substantially more than a comparable region.")

    # One image is enough for the illustration; no need to hold the split.
    figure = plot_examples(load_images(frame.head(1)), width, model_name)
    print(f"\n  figure -> {figure.relative_to(config.PROJECT_ROOT)}")

    summary = {"model": model_name, "border_width_px": width,
               "majority_class_rate": majority, **results}
    out = config.METRICS_DIR / f"{model_name}_border_ablation.json"
    out.write_text(json.dumps(summary, indent=2))
    keras.backend.clear_session()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="all", choices=[*config.MODEL_NAMES, "all"])
    parser.add_argument("--width", type=int, default=20,
                        help="border thickness in pixels (default 20 of 224)")
    args = parser.parse_args()

    if not 1 <= args.width < config.IMAGE_SIZE[0] // 2:
        raise SystemExit(f"--width must be between 1 and {config.IMAGE_SIZE[0] // 2 - 1}")

    config.ensure_dirs()
    test_csv = config.PROCESSED_DIR / "test.csv"
    if not test_csv.exists():
        raise SystemExit(f"'{test_csv}' not found. Run `python src/preprocessing.py` first.")

    frame = pd.read_csv(test_csv)
    y_true = np.array([config.CLASS_TO_INDEX[c] for c in frame["label"]])

    targets = config.MODEL_NAMES if args.model == "all" else [args.model]
    available = [n for n in targets if config.model_path(n).exists()]
    if not available:
        raise SystemExit("No trained models found. Run src/train.py first.")

    from src.evaluate import assert_model_matches_split
    for name in available:
        assert_model_matches_split(name)
    for name in available:
        run(name, args.width, frame, y_true)

    print(f"\nSaved to {config.METRICS_DIR.relative_to(config.PROJECT_ROOT)}/"
          f"<model>_border_ablation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
