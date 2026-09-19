"""Assess and correct the confidence scores of the trained models.

    python src/calibrate.py               # every trained model
    python src/calibrate.py --model mobilenetv2

Why this exists. A softmax output of 0.90 is routinely taken to mean "right nine
times in ten", and for modern deep networks it usually is not: they are
systematically overconfident (Guo et al., "On Calibration of Modern Neural
Networks", ICML 2017). For a screening tool aimed at a health worker with no
specialist to consult, a confidently wrong answer is worse than an admission of
uncertainty, so the trustworthiness of the number matters as much as the
accuracy behind it.

Three things are produced:

  1. Expected Calibration Error (ECE) and a reliability diagram, quantifying the
     gap between stated confidence and observed accuracy.

  2. A temperature, fitted by minimising negative log-likelihood on the
     VALIDATION split. Dividing the logits by it rescales confidence without
     changing any prediction — accuracy is identical before and after, only the
     probabilities move. Fitting on test would make the reported calibration a
     property of the fit rather than a measurement of the model.

  3. An abstention threshold from the risk-coverage trade-off (selective
     prediction). Below it the prototype declines to name a condition and
     recommends consulting a clinician. Chosen on validation as the lowest
     threshold reaching config.SELECTIVE_TARGET_ACCURACY among accepted cases
     while still answering at least config.SELECTIVE_MIN_COVERAGE of them: a
     tool that abstains on everything is safe and useless.

The test split is used only to report the outcome, never to choose anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.stats import apply_temperature, probs_to_logits  # noqa: E402
from src.evaluate import assert_model_matches_split  # noqa: E402

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE
EPS = 1e-12


# ---------------------------------------------------------------------------
# Probabilities and temperature
# ---------------------------------------------------------------------------
def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, bool]:
    """Temperature minimising validation NLL.

    T > 1 softens overconfident predictions; T < 1 sharpens underconfident ones.
    One parameter, so it cannot overfit the validation split in any meaningful
    sense, which is what makes this the standard choice.

    Returns (temperature, hit_bound). See below for why the second matters.
    """
    def nll(temperature: float) -> float:
        probs = apply_temperature(logits, temperature)
        return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + EPS)))

    lower, upper = 0.05, 10.0
    result = minimize_scalar(nll, bounds=(lower, upper), method="bounded")
    temperature = float(result.x)

    # Hitting a bound means the optimiser wanted to go further, which happens
    # when validation accuracy is so near perfect that the likelihood keeps
    # improving as confidence is pushed to 1. The fitted value is then an
    # artefact of the bound rather than a property of the model, and the caller
    # is told so instead of being handed a number that looks meaningful.
    at_bound = temperature <= lower * 1.01 or temperature >= upper * 0.99
    return temperature, at_bound


# ---------------------------------------------------------------------------
# Calibration error
# ---------------------------------------------------------------------------
def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, bins: int = 15
) -> tuple[float, list[dict]]:
    """ECE and the per-bin detail behind a reliability diagram.

    Predictions are grouped by confidence; within each bin the mean confidence
    is compared with the observed accuracy. ECE is the average gap, weighted by
    bin population. A perfectly calibrated model scores 0.
    """
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    detail, ece = [], 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        # Include the upper edge in the final bin so confidence == 1.0 counts.
        in_bin = (confidence > low) & (confidence <= high) if high < 1.0 else (confidence > low)
        count = int(in_bin.sum())
        if count == 0:
            detail.append({"low": float(low), "high": float(high), "count": 0,
                           "confidence": None, "accuracy": None})
            continue
        bin_conf = float(confidence[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (count / len(y_true)) * abs(bin_acc - bin_conf)
        detail.append({"low": float(low), "high": float(high), "count": count,
                       "confidence": bin_conf, "accuracy": bin_acc})
    return float(ece), detail


# ---------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------
def risk_coverage_curve(probs: np.ndarray, y_true: np.ndarray, steps: int = 100) -> list[dict]:
    """Accuracy among accepted predictions as the confidence threshold rises."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)

    curve = []
    for threshold in np.linspace(0.0, 0.999, steps):
        accepted = confidence >= threshold
        n = int(accepted.sum())
        if n == 0:
            break
        curve.append({
            "threshold": float(threshold),
            "coverage": float(n / len(y_true)),
            "accuracy": float(correct[accepted].mean()),
            "n_accepted": n,
        })
    return curve


def choose_threshold(curve: list[dict], target_accuracy: float, min_coverage: float) -> dict:
    """Lowest threshold reaching the target accuracy while keeping enough coverage.

    Lowest rather than highest: among thresholds that meet the target, the one
    that abstains least is preferable. If the target is unreachable at acceptable
    coverage the fallback maximises accuracy subject to the coverage floor, and
    says so, rather than silently returning something that misses the target.
    """
    viable = [p for p in curve if p["accuracy"] >= target_accuracy
              and p["coverage"] >= min_coverage]
    if viable:
        best = min(viable, key=lambda p: p["threshold"])
        return {**best, "target_met": True}

    fallback = [p for p in curve if p["coverage"] >= min_coverage]
    if not fallback:
        return {**curve[0], "target_met": False}
    best = max(fallback, key=lambda p: p["accuracy"])
    return {**best, "target_met": False}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=config.IMAGE_CHANNELS)
    image = tf.cast(image, tf.float32)
    image.set_shape((*config.IMAGE_SIZE, config.IMAGE_CHANNELS))
    return image, label


def predict_split(model: keras.Model, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = config.PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        raise SystemExit(f"'{path}' not found. Run `python src/preprocessing.py` first.")
    frame = pd.read_csv(path)
    paths = [str(config.PROJECT_ROOT / p) for p in frame["path"]]
    labels = np.array([config.CLASS_TO_INDEX[c] for c in frame["label"]])

    dataset = (
        tf.data.Dataset.from_tensor_slices((paths, labels))
        .map(_decode, num_parallel_calls=AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    return model.predict(dataset, verbose=0), labels


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_calibration(model_name: str, before: list[dict], after: list[dict],
                     ece_before: float, ece_after: float, curve: list[dict],
                     chosen: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for detail, label, colour in ((before, f"before (ECE {ece_before:.4f})", "#C44E52"),
                                  (after, f"after (ECE {ece_after:.4f})", "#4C72B0")):
        points = [(d["confidence"], d["accuracy"]) for d in detail if d["count"]]
        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, "o-", color=colour, ms=5, label=label)
    ax.set(xlabel="Mean predicted confidence", ylabel="Observed accuracy",
           title=f"{model_name} — reliability diagram (validation)",
           xlim=(0, 1.02), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot([p["coverage"] for p in curve], [p["accuracy"] for p in curve],
            color="#55A868", lw=2)
    ax.axvline(chosen["coverage"], color="crimson", ls="--", lw=1)
    ax.axhline(chosen["accuracy"], color="crimson", ls="--", lw=1)
    ax.plot([chosen["coverage"]], [chosen["accuracy"]], "o", color="crimson", ms=8,
            label=(f"threshold {chosen['threshold']:.3f}\n"
                   f"coverage {chosen['coverage']:.1%}, acc {chosen['accuracy']:.4f}"))
    ax.set(xlabel="Coverage (fraction of cases answered)", ylabel="Accuracy on answered cases",
           title=f"{model_name} — risk-coverage trade-off (validation)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = config.FIGURES_DIR / f"{model_name}_calibration.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def calibrate_one(model_name: str) -> dict:
    print("\n" + "=" * 74)
    print(f"{model_name.upper()} — CALIBRATION")
    print("=" * 74)

    assert_model_matches_split(model_name)
    model = keras.models.load_model(config.model_path(model_name))
    val_probs, val_true = predict_split(model, "val")
    test_probs, test_true = predict_split(model, "test")

    logits = probs_to_logits(val_probs)
    ece_before, detail_before = expected_calibration_error(
        val_probs, val_true, config.CALIBRATION_BINS
    )
    # ECE is a binned statistic: with few samples per occupied bin the estimate
    # is dominated by noise, and can move in either direction after a fit that
    # genuinely improved calibration. Say so rather than letting a spurious
    # change be read as a result.
    occupied = [d for d in detail_before if d["count"]]
    per_bin = len(val_true) / max(len(occupied), 1)
    if per_bin < 30:
        print(
            f"  NOTE: {len(val_true)} validation images across {len(occupied)} occupied "
            f"bins is ~{per_bin:.0f} per bin.\n  ECE is unreliable at that density and "
            f"small changes in it should not be interpreted."
        )
    val_accuracy = float(np.mean(val_probs.argmax(1) == val_true))
    temperature, at_bound = fit_temperature(logits, val_true)
    val_calibrated = apply_temperature(logits, temperature)
    ece_after, detail_after = expected_calibration_error(
        val_calibrated, val_true, config.CALIBRATION_BINS
    )

    print(f"fitted temperature : {temperature:.4f}  "
          f"({'softening overconfident' if temperature > 1 else 'sharpening underconfident'} "
          f"predictions)")
    if at_bound:
        print(
            "  WARNING: the fit hit its bound. Validation accuracy is "
            f"{val_accuracy:.4f}, high enough that the likelihood keeps improving "
            "as\n  confidence is pushed towards 1, so this temperature reflects the "
            "bound rather than\n  the model. Treat the calibrated confidences with "
            "caution and report the raw ECE too."
        )
    print(f"validation ECE     : {ece_before:.4f} -> {ece_after:.4f}")
    print(f"mean confidence    : {val_probs.max(axis=1).mean():.4f} -> "
          f"{val_calibrated.max(axis=1).mean():.4f}   "
          f"(accuracy {np.mean(val_probs.argmax(1) == val_true):.4f})")

    # Report on test, having chosen nothing from it.
    test_calibrated = apply_temperature(probs_to_logits(test_probs), temperature)
    ece_test_before, _ = expected_calibration_error(test_probs, test_true, config.CALIBRATION_BINS)
    ece_test_after, _ = expected_calibration_error(
        test_calibrated, test_true, config.CALIBRATION_BINS
    )
    print(f"test ECE           : {ece_test_before:.4f} -> {ece_test_after:.4f}  "
          f"(temperature fitted on validation only)")

    accuracy_unchanged = bool(
        np.array_equal(test_probs.argmax(1), test_calibrated.argmax(1))
    )
    print(f"predictions unchanged by scaling: {accuracy_unchanged}  "
          f"(temperature rescales confidence, it does not reorder classes)")

    curve = risk_coverage_curve(val_calibrated, val_true)
    chosen = choose_threshold(curve, config.SELECTIVE_TARGET_ACCURACY,
                              config.SELECTIVE_MIN_COVERAGE)

    print(f"\nabstention threshold : {chosen['threshold']:.3f}")
    if chosen["threshold"] <= 1e-6:
        print(f"  The model already reaches {config.SELECTIVE_TARGET_ACCURACY} accuracy "
              f"answering every case, so no abstention is needed on this data.")
    print(f"  on validation      : answers {chosen['coverage']:.1%} of cases at "
          f"{chosen['accuracy']:.4f} accuracy")
    if not chosen["target_met"]:
        print(f"  NOTE: target accuracy {config.SELECTIVE_TARGET_ACCURACY} was not "
              f"reachable while keeping {config.SELECTIVE_MIN_COVERAGE:.0%} coverage; "
              f"this is the best accuracy available at that coverage floor.")

    accepted = test_calibrated.max(axis=1) >= chosen["threshold"]
    test_coverage = float(accepted.mean())
    test_accuracy = float(
        (test_calibrated.argmax(1)[accepted] == test_true[accepted]).mean()
    ) if accepted.any() else float("nan")
    print(f"  on test            : answers {test_coverage:.1%} of cases at "
          f"{test_accuracy:.4f} accuracy")

    figure = plot_calibration(model_name, detail_before, detail_after,
                              ece_before, ece_after, curve, chosen)
    print(f"\n  figure -> {figure.relative_to(config.PROJECT_ROOT)}")

    summary = {
        "model": model_name,
        "temperature": temperature,
        "temperature_hit_bound": at_bound,
        "validation_accuracy": val_accuracy,
        "abstention_threshold": chosen["threshold"],
        "target_accuracy": config.SELECTIVE_TARGET_ACCURACY,
        "min_coverage": config.SELECTIVE_MIN_COVERAGE,
        "target_met": chosen["target_met"],
        "validation": {
            "ece_before": ece_before, "ece_after": ece_after,
            "coverage": chosen["coverage"], "accuracy_on_accepted": chosen["accuracy"],
        },
        "test": {
            "ece_before": ece_test_before, "ece_after": ece_test_after,
            "coverage": test_coverage, "accuracy_on_accepted": test_accuracy,
        },
        "predictions_unchanged": accuracy_unchanged,
        "reliability_before": detail_before,
        "reliability_after": detail_after,
        "risk_coverage_curve": curve,
    }
    config.calibration_path(model_name).write_text(json.dumps(summary, indent=2))
    keras.backend.clear_session()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="all", choices=[*config.MODEL_NAMES, "all"])
    args = parser.parse_args()

    config.ensure_dirs()
    targets = config.MODEL_NAMES if args.model == "all" else [args.model]
    available = [n for n in targets if config.model_path(n).exists()]
    if not available:
        raise SystemExit("No trained models found. Run src/train.py first.")

    summaries = [calibrate_one(name) for name in available]

    if len(summaries) > 1:
        print("\n" + "=" * 74)
        print("CALIBRATION SUMMARY")
        print("=" * 74)
        table = pd.DataFrame([{
            "model": s["model"],
            "temperature": round(s["temperature"], 3),
            "val_ECE_before": round(s["validation"]["ece_before"], 4),
            "val_ECE_after": round(s["validation"]["ece_after"], 4),
            "test_ECE_before": round(s["test"]["ece_before"], 4),
            "test_ECE_after": round(s["test"]["ece_after"], 4),
            "threshold": round(s["abstention_threshold"], 3),
            "test_coverage": round(s["test"]["coverage"], 4),
            "test_acc_accepted": round(s["test"]["accuracy_on_accepted"], 4),
        } for s in summaries])
        print(table.to_string(index=False))
        table.to_csv(config.METRICS_DIR / "calibration_summary.csv", index=False)
        print(f"\n  -> results/metrics/calibration_summary.csv")

    print("\nNext: streamlit run src/prototype/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
