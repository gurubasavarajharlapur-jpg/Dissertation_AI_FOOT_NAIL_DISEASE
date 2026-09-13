"""Evaluate the trained models on the held-out test split and compare them.

Implements WBS 5.1-5.7.

    python src/evaluate.py                    # every trained model, side by side
    python src/evaluate.py --model mobilenetv2
    python src/evaluate.py --no-gradcam       # skip the saliency figures

The test split is read here and nowhere else, once, at the end of the project.
Model selection, early stopping and every hyper-parameter choice used the
validation split. Had anything been chosen against test scores, those scores
would be an optimistic estimate of the model rather than a measurement of it.

Reported per model (5.4, WBS 5.1-5.5):

  Accuracy        Overall correctness. Reported but not relied on: the classes
                  are imbalanced, so a model that never predicts the rarest
                  condition can still post a high figure.
  Precision       Per class. Of the images called "ulcer", how many were? Low
                  precision means false alarms and unnecessary referrals.
  Recall          Per class. Of the actual ulcers, how many were caught? The
                  critical metric for a screening tool, where a missed ulcer
                  costs far more than a false alarm.
  F1              Harmonic mean of the two, macro-averaged across classes so
                  all four count equally regardless of how many images each has.
  Confusion       Which conditions are mistaken for which. Clinically the most
                  informative output: wound/ulcer confusion matters far more
                  than either being confused with healthy.

Also produced, because 2.5.8 claims computational efficiency and deployment
suitability as part of this dissertation's contribution:

  Parameters, on-disk size, and single-image inference time per model.

And, because 2.5.8 identifies model interpretability as an underdeveloped area:

  Grad-CAM overlays showing which region drove each prediction. These are also
  the check on a specific risk in this dataset: each class is drawn largely from
  one source, so a model could score well by recognising which dataset an image
  came from rather than the condition. A within-source summary is printed
  alongside, since those comparisons are the ones the confound cannot explain.
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
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

AUTOTUNE = tf.data.AUTOTUNE

# Which backbone each saved model uses, for Grad-CAM's input scaling.
PREPROCESS = {
    "mobilenetv2": keras.applications.mobilenet_v2.preprocess_input,
    "resnet50": keras.applications.resnet50.preprocess_input,
}


# ---------------------------------------------------------------------------
# Staleness guard
# ---------------------------------------------------------------------------
def assert_model_matches_split(model_name: str) -> None:
    """Refuse to score a model that was trained against a different split.

    A checkpoint outlives the data it was trained on. Re-running preprocessing
    with changed settings produces a new split while the old .keras file sits
    there looking perfectly usable — and a model trained on the previous split
    has very likely seen images that are now in the test set. The resulting
    numbers would be inflated by leakage and look *better*, which is the kind of
    error that survives review.

    The training history records how many validation images the run saw, so
    comparing that against the current split catches the mismatch.
    """
    history_path = config.history_path(model_name)
    if not history_path.exists():
        print(f"NOTE: no training history for {model_name}; cannot verify it matches "
              f"the current split.")
        return

    val_csv = config.PROCESSED_DIR / "val.csv"
    if not val_csv.exists():
        return

    current = len(pd.read_csv(val_csv))
    recorded = json.loads(history_path.read_text()).get("val_images")
    if recorded is None or recorded == current:
        return

    raise SystemExit(
        f"\n'{model_name}' was trained against a different split.\n"
        f"  its training run saw {recorded} validation images; the current split "
        f"has {current}.\n\n"
        f"A model trained on the earlier split has probably seen images that are "
        f"now in the\n  test set, so scoring it here would leak and overstate its "
        f"performance.\n\n"
        f"Retrain it:  python src/train.py --model {model_name}\n"
        f"Or evaluate only the current model(s): "
        f"python src/evaluate.py --model <name>"
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_test_set() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "test.csv"
    if not path.exists():
        raise SystemExit(f"'{path}' not found. Run `python src/preprocessing.py` first.")
    return pd.read_csv(path)


def _decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=config.IMAGE_CHANNELS)
    image = tf.cast(image, tf.float32)
    image.set_shape((*config.IMAGE_SIZE, config.IMAGE_CHANNELS))
    return image, label


def build_test_dataset(frame: pd.DataFrame) -> tf.data.Dataset:
    """Unshuffled, unaugmented — the test split must be a fixed measurement."""
    paths = [str(config.PROJECT_ROOT / p) for p in frame["path"]]
    labels = [config.CLASS_TO_INDEX[c] for c in frame["label"]]
    return (
        tf.data.Dataset.from_tensor_slices((paths, labels))
        .map(_decode, num_parallel_calls=AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    labels = list(range(config.NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(config.CLASS_NAMES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    # One-vs-rest AUC, skipping any class absent from the test split.
    aucs = {}
    for i, name in enumerate(config.CLASS_NAMES):
        binary = (y_true == i).astype(int)
        if 0 < binary.sum() < len(binary):
            aucs[name] = float(roc_auc_score(binary, y_prob[:, i]))
    if aucs:
        result["per_class_auc"] = aucs
        result["macro_auc"] = float(np.mean(list(aucs.values())))
    return result


def _time_inference(model: keras.Model, device: str, runs: int) -> dict | None:
    """Median single-image latency on one device, or None if unavailable."""
    sample = np.random.rand(1, *config.INPUT_SHAPE).astype("float32") * 255
    try:
        with tf.device(device):
            for _ in range(5):  # warm up: the first calls include graph tracing
                model(sample, training=False)
            timings = []
            for _ in range(runs):
                start = time.perf_counter()
                model(sample, training=False)
                timings.append((time.perf_counter() - start) * 1000)
    except (RuntimeError, tf.errors.InvalidArgumentError):
        return None
    return {
        "mean": round(float(np.mean(timings)), 2),
        "median": round(float(np.median(timings)), 2),
        "p95": round(float(np.percentile(timings, 95)), 2),
    }


def measure_efficiency(model: keras.Model, model_name: str, runs: int = 50) -> dict:
    """Parameters, on-disk size and single-image latency on both CPU and GPU.

    Batch size 1 is measured deliberately: the deployment story is one photo at
    a time on a health worker's phone, not a batched server workload.

    Both devices are timed because they rank the architectures differently.
    MobileNetV2's depthwise separable convolutions cut parameters and FLOPs but
    leave a GPU's dense-matrix hardware underused, so on a GPU it can be the
    slower of the two despite being an order of magnitude smaller. CPU latency
    is the figure that speaks to running on a phone, and reporting only the GPU
    number would argue against the deployment case the model size supports.
    """
    path = config.model_path(model_name)
    result = {
        "total_parameters": int(model.count_params()),
        "model_size_mb": round(path.stat().st_size / 1e6, 2),
    }

    cpu = _time_inference(model, "/CPU:0", runs)
    if cpu:
        result["inference_ms_cpu_median"] = cpu["median"]
        result["inference_ms_cpu_p95"] = cpu["p95"]

    if tf.config.list_physical_devices("GPU"):
        gpu = _time_inference(model, "/GPU:0", runs)
        if gpu:
            result["inference_ms_gpu_median"] = gpu["median"]
            result["inference_ms_gpu_p95"] = gpu["p95"]

    # The headline latency is CPU where measured: it is the device a rural
    # deployment actually resembles.
    result["inference_ms_median"] = result.get(
        "inference_ms_cpu_median", result.get("inference_ms_gpu_median", float("nan"))
    )
    result["inference_device"] = "cpu" if "inference_ms_cpu_median" in result else "gpu"
    return result


def within_source_summary(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy broken down by the dataset each image came from.

    Each class here is drawn largely from one source, so overall accuracy is
    partly explainable by the model recognising acquisition conditions rather
    than pathology. The comparisons that confound cannot explain are the ones
    *inside* a single source, and this reports those.
    """
    out: dict = {}
    for source in sorted(frame["source"].unique()):
        mask = (frame["source"] == source).to_numpy()
        classes_here = sorted(set(y_true[mask]))
        out[source] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "classes_present": [config.CLASS_NAMES[i] for i in classes_here],
            # A source containing only one class cannot demonstrate
            # discrimination: any constant prediction scores 100%.
            "informative": len(classes_here) > 1,
        }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(matrix: list[list[int]], model_name: str, normalize: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.asarray(matrix, dtype=float)
    counts = data.copy()
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums > 0)

    names = [config.CLASS_DISPLAY_NAMES[c] for c in config.CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    image = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if normalize else data.max())
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(len(names)), yticks=range(len(names)),
        xticklabels=names, yticklabels=names,
        xlabel="Predicted", ylabel="True",
        title=f"{model_name} — test confusion matrix"
              f"{' (row-normalised)' if normalize else ''}",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    for i in range(len(names)):
        for j in range(len(names)):
            label = f"{data[i, j]:.2f}\n({int(counts[i, j])})" if normalize else f"{int(counts[i, j])}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > 0.5 else "black")

    fig.tight_layout()
    out = config.FIGURES_DIR / f"{model_name}_confusion_matrix.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_comparison(summaries: dict[str, dict]) -> Path | None:
    """Accuracy against cost — the trade-off this dissertation is about."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(summaries) < 2:
        return None

    names = list(summaries)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    metrics = ["accuracy", "macro_f1", "macro_recall"]
    width = 0.35
    positions = np.arange(len(metrics))
    for offset, name in enumerate(names):
        values = [summaries[name]["metrics"][m] for m in metrics]
        bars = axes[0].bar(positions + offset * width, values, width, label=name)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[0].set(xticks=positions + width / 2,
                xticklabels=["Accuracy", "Macro F1", "Macro recall"],
                ylim=(0, 1.08), title="Test performance")
    axes[0].legend()

    latency_key = ("inference_ms_cpu_median"
                   if all("inference_ms_cpu_median" in summaries[n]["efficiency"] for n in names)
                   else "inference_ms_median")
    latency_label = "Inference time, CPU (1 image)" if latency_key.endswith("cpu_median") \
        else "Inference time (1 image)"
    for ax, key, title, unit in (
        (axes[1], "model_size_mb", "Model size", "MB"),
        (axes[2], latency_key, latency_label, "ms"),
    ):
        values = [summaries[n]["efficiency"][key] for n in names]
        bars = ax.bar(names, values, color=["#4C72B0", "#DD8452"][: len(names)])
        ax.bar_label(bars, fmt="%.1f", fontsize=9)
        ax.set(title=f"{title} ({unit})")
        ax.margins(y=0.18)

    fig.suptitle("MobileNetV2 vs ResNet50 — accuracy against deployment cost", fontsize=12)
    fig.tight_layout()
    out = config.FIGURES_DIR / "model_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def gradcam_heatmap(
    model: keras.Model, model_name: str, image: np.ndarray, class_index: int | None = None
) -> tuple[np.ndarray, int]:
    """Grad-CAM for one image. Returns the heatmap and the class explained.

    The head is re-applied by hand rather than by slicing the saved model. The
    backbone is a nested Model, so its feature map is not a tensor in the outer
    graph and cannot simply be read out of it; calling the layers explicitly
    inside a GradientTape is both robust and clearer about what is differentiated.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.
    """
    preprocess = PREPROCESS[model_name]
    backbone = model.get_layer(index=1)
    gap = model.get_layer("gap")
    dropout = model.get_layer("dropout")
    head = model.get_layer("predictions")

    batch = tf.convert_to_tensor(image[None].astype("float32"))
    with tf.GradientTape() as tape:
        features = backbone(preprocess(batch), training=False)
        tape.watch(features)
        logits = head(dropout(gap(features), training=False))
        if class_index is None:
            class_index = int(tf.argmax(logits[0]))
        score = logits[:, class_index]

    grads = tape.gradient(score, features)
    if grads is None:
        raise RuntimeError("no gradient reached the feature map; the graph is disconnected")

    # Channel importance is the spatially averaged gradient; the map is the
    # ReLU'd weighted sum of the activation channels.
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.einsum("bhwc,bc->bhw", features, weights))[0].numpy()

    cam -= cam.min()
    peak = cam.max()
    # A flat map means this layer contributed nothing for this class; returning
    # zeros is honest, whereas dividing by ~0 manufactures noise.
    return (cam / peak if peak > 1e-8 else np.zeros_like(cam)), class_index


def plot_gradcam_grid(
    model: keras.Model, model_name: str, frame: pd.DataFrame,
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, per_class: int = 2,
) -> Path:
    """Grad-CAM overlays: correct and incorrect predictions for each class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(config.RANDOM_SEED)
    chosen: list[tuple[int, str]] = []
    for index, name in enumerate(config.CLASS_NAMES):
        correct = np.flatnonzero((y_true == index) & (y_pred == index))
        wrong = np.flatnonzero((y_true == index) & (y_pred != index))
        for pool, tag in ((correct, "correct"), (wrong, "misclassified")):
            if len(pool):
                for pick in rng.choice(pool, size=min(per_class, len(pool)), replace=False):
                    chosen.append((int(pick), tag))

    if not chosen:
        raise RuntimeError("no test predictions to explain")

    cols = 4
    rows = (len(chosen) + cols - 1) // cols
    # Extra height per row: each panel carries a two-line title, which overlaps
    # the image above it at tighter spacing.
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 4.0), squeeze=False)
    flat = axes.ravel()

    for ax, (idx, tag) in zip(flat, chosen):
        path = config.PROJECT_ROOT / frame.iloc[idx]["path"]
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        cam, _ = gradcam_heatmap(model, model_name, image, class_index=int(y_pred[idx]))

        cam_up = np.asarray(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (image.shape[1], image.shape[0]), Image.BILINEAR
            )
        ) / 255.0
        # Weighted towards the photograph: the point of the figure is to show
        # the lesion the model attended to, which a heavier heatmap hides.
        overlay = 0.65 * (image / 255.0) + 0.35 * matplotlib.colormaps["jet"](cam_up)[..., :3]

        ax.imshow(np.clip(overlay, 0, 1))
        ax.axis("off")
        true_name = config.CLASS_DISPLAY_NAMES[config.CLASS_NAMES[y_true[idx]]]
        pred_name = config.CLASS_DISPLAY_NAMES[config.CLASS_NAMES[y_pred[idx]]]
        colour = "green" if tag == "correct" else "crimson"
        ax.set_title(
            f"true: {true_name}\npred: {pred_name} ({y_prob[idx].max():.2f})",
            fontsize=7.5, color=colour,
        )
    for ax in flat[len(chosen):]:
        ax.axis("off")

    fig.suptitle(
        f"{model_name} — Grad-CAM on test predictions\n"
        f"red/yellow marks the region that drove the prediction",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.subplots_adjust(hspace=0.30)
    out = config.FIGURES_DIR / f"{model_name}_gradcam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_model_report(model_name: str, metrics: dict, efficiency: dict,
                       y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("\n" + "=" * 74)
    print(f"{model_name.upper()} — TEST SET ({metrics['n_test']} images)")
    print("=" * 74)
    print(
        classification_report(
            y_true, y_pred,
            labels=list(range(config.NUM_CLASSES)),
            target_names=[config.CLASS_DISPLAY_NAMES[c] for c in config.CLASS_NAMES],
            digits=4, zero_division=0,
        )
    )
    print(f"Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"Cohen's kappa     : {metrics['cohen_kappa']:.4f}")
    if "macro_auc" in metrics:
        print(f"Macro AUC (OvR)   : {metrics['macro_auc']:.4f}")
    latency = []
    if "inference_ms_cpu_median" in efficiency:
        latency.append(f"CPU {efficiency['inference_ms_cpu_median']} ms")
    if "inference_ms_gpu_median" in efficiency:
        latency.append(f"GPU {efficiency['inference_ms_gpu_median']} ms")
    print(
        f"\nParameters {efficiency['total_parameters']:,} | "
        f"size {efficiency['model_size_mb']} MB | "
        f"inference per image: {', '.join(latency)}"
    )

    print("\nConfusion matrix (rows = true, columns = predicted):")
    names = config.CLASS_NAMES
    header = "".join(f"{n[:11]:>13}" for n in names)
    print(f"{'':<14}{header}")
    for i, row in enumerate(metrics["confusion_matrix"]):
        print(f"{names[i][:13]:<14}" + "".join(f"{v:>13}" for v in row))


def print_comparison(summaries: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, summary in summaries.items():
        m, e = summary["metrics"], summary["efficiency"]
        rows.append({
            "model": name,
            "accuracy": round(m["accuracy"], 4),
            "macro_precision": round(m["macro_precision"], 4),
            "macro_recall": round(m["macro_recall"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "parameters": e["total_parameters"],
            "size_mb": e["model_size_mb"],
            "inference_ms_cpu": e.get("inference_ms_cpu_median"),
            "inference_ms_gpu": e.get("inference_ms_gpu_median"),
        })
    table = pd.DataFrame(rows)

    print("\n" + "=" * 74)
    print("MODEL COMPARISON (WBS 5.6)")
    print("=" * 74)
    print(table.to_string(index=False))

    if len(table) == 2:
        primary = table[table["model"] == config.PRIMARY_MODEL]
        other = table[table["model"] != config.PRIMARY_MODEL]
        if not primary.empty and not other.empty:
            p, o = primary.iloc[0], other.iloc[0]
            message = (
                f"\n{o['model']} is {o['accuracy'] - p['accuracy']:+.4f} accuracy "
                f"({(o['accuracy'] - p['accuracy']) * 100:+.2f} points) over {p['model']},\n"
                f"for {o['parameters'] / p['parameters']:.1f}x the parameters and "
                f"{o['size_mb'] / p['size_mb']:.1f}x the on-disk size."
            )
            if p.get("inference_ms_cpu") and o.get("inference_ms_cpu"):
                message += (
                    f"\nOn CPU — the device a phone resembles — {o['model']} takes "
                    f"{o['inference_ms_cpu'] / p['inference_ms_cpu']:.1f}x as long "
                    f"({o['inference_ms_cpu']} ms vs {p['inference_ms_cpu']} ms)."
                )
            if p.get("inference_ms_gpu") and o.get("inference_ms_gpu"):
                message += (
                    f"\nOn GPU the ordering differs ({o['inference_ms_gpu']} ms vs "
                    f"{p['inference_ms_gpu']} ms): depthwise separable convolutions cut "
                    f"FLOPs\nbut underuse dense-matrix hardware, so the lighter model's "
                    f"advantage is a CPU one."
                )
            print(message)
    return table


def print_within_source(model_name: str, summary: dict) -> None:
    print(f"\n{model_name} — accuracy by source dataset:")
    for source, stats in summary.items():
        note = "" if stats["informative"] else "   (single class — not evidence of discrimination)"
        print(f"  {source:<24}{stats['n']:>6} images   acc {stats['accuracy']:.4f}{note}")
    print(
        "  Each class comes largely from one source, so overall accuracy is partly\n"
        "  explainable by recognising acquisition conditions. The confusion matrix\n"
        "  cells between classes sharing a source are the ones that are not."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="all", choices=[*config.MODEL_NAMES, "all"])
    parser.add_argument("--no-gradcam", action="store_true", help="skip the saliency figures")
    args = parser.parse_args()

    config.ensure_dirs()
    frame = load_test_set()
    dataset = build_test_dataset(frame)
    y_true = np.array([config.CLASS_TO_INDEX[c] for c in frame["label"]])

    targets = config.MODEL_NAMES if args.model == "all" else [args.model]
    available = [n for n in targets if config.model_path(n).exists()]
    missing = [n for n in targets if not config.model_path(n).exists()]
    for name in missing:
        print(f"NOTE: no trained {name} at {config.model_path(name)} — skipping. "
              f"Train it with: python src/train.py --model {name}")
    if not available:
        raise SystemExit("No trained models found. Run src/train.py first.")

    print(f"Test set: {len(frame)} images, {config.NUM_CLASSES} classes")
    print("This split has not been used for any training or selection decision.")

    for name in available:
        assert_model_matches_split(name)

    summaries: dict[str, dict] = {}
    for name in available:
        model = keras.models.load_model(config.model_path(name))
        y_prob = model.predict(dataset, verbose=0)
        y_pred = y_prob.argmax(axis=1)

        metrics = compute_metrics(y_true, y_pred, y_prob)
        efficiency = measure_efficiency(model, name)
        by_source = within_source_summary(frame, y_true, y_pred)

        print_model_report(name, metrics, efficiency, y_true, y_pred)
        print_within_source(name, by_source)

        cm_path = plot_confusion_matrix(metrics["confusion_matrix"], name)
        print(f"\n  confusion matrix -> {cm_path.relative_to(config.PROJECT_ROOT)}")

        if not args.no_gradcam:
            cam_path = plot_gradcam_grid(model, name, frame, y_true, y_pred, y_prob)
            print(f"  Grad-CAM         -> {cam_path.relative_to(config.PROJECT_ROOT)}")

        summary = {"model": name, "metrics": metrics, "efficiency": efficiency,
                   "by_source": by_source}
        config.metrics_path(name).write_text(json.dumps(summary, indent=2))
        summaries[name] = summary
        del model
        keras.backend.clear_session()

    if len(summaries) > 1:
        table = print_comparison(summaries)
        table.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
        comparison_plot = plot_comparison(summaries)
        print(f"\n  comparison table -> results/metrics/model_comparison.csv")
        if comparison_plot:
            print(f"  comparison plot  -> {comparison_plot.relative_to(config.PROJECT_ROOT)}")
    else:
        print("\nOnly one trained model available, so no comparison was produced.")
        print("WBS 5.6 requires both — train the other with src/train.py.")

    print(f"\nPer-model metrics -> {config.METRICS_DIR.relative_to(config.PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
