"""Score images of a condition the model has no category for.

The classifier has four classes and a softmax output, so every image it
receives is assigned to one of those four with a confidence attached. It has no
way to say "this is something else". Section 6.6.3 of the dissertation states
that as a limitation; this script measures it.

The Figshare collection contains 578 images of **nail dystrophy**, a real nail
condition that is not one of the four classes fixed by the proposal. Those
images were excluded during preprocessing and have never been seen by either
model, in training, validation or test. That makes them an unusually clean
out-of-distribution probe: genuinely medical, genuinely nail photographs,
genuinely outside the label space, and already downloaded.

Three questions, in order of how much they matter:

  1. **Does the safety net catch them?** Below the abstention threshold the
     system names no condition and refers the person. That is the correct
     behaviour for an input it cannot represent. The proportion caught is the
     headline number.
  2. **What does it call the rest?** "Nail fungal infection" is a wrong label
     that still routes someone to care, so it is a tolerable failure.
     "Healthy" is not: it tells somebody with a real nail condition that
     nothing is wrong.
  3. **Is it less confident than on data it was trained for?** If confidence on
     unknown input is indistinguishable from confidence on known input, then
     confidence carries no information about whether the model is out of its
     depth, and no threshold can fix that.

Usage:
    python src/ood_test.py
    python src/ood_test.py --model resnet50 --limit 200
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, triage  # noqa: E402
from src.stats import apply_temperature, probs_to_logits, wilson_interval  # noqa: E402

# The leaf folder name that marks the excluded condition, taken from the same
# table preprocessing uses so the two cannot drift apart.
EXCLUDED_LEAF = [name for name, target in config.LEAF_FOLDER_TO_CLASS.items()
                 if target is None]


def find_images(limit: int | None) -> list[Path]:
    """Every image under a leaf folder that preprocessing maps to None."""
    if not EXCLUDED_LEAF:
        raise SystemExit("No excluded leaf folders in config.LEAF_FOLDER_TO_CLASS.")
    found: list[Path] = []
    for path in sorted(config.RAW_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in config.VALID_EXTENSIONS:
            continue
        relative = path.relative_to(config.RAW_DIR).as_posix()
        if any(f"/{leaf}/" in f"/{relative}" for leaf in EXCLUDED_LEAF):
            found.append(path)
    if not found:
        raise SystemExit(
            f"No images found under any of {EXCLUDED_LEAF} in {config.RAW_DIR}.\n"
            f"These come with the Figshare download — run src/download_data.py, "
            f"or restore data/raw from Drive."
        )
    return found[:limit] if limit else found


def prepare(image: Image.Image) -> np.ndarray:
    """Resize exactly as preprocessing does, so inference matches training."""
    image = image.convert("RGB")
    target_w, target_h = config.IMAGE_SIZE
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(target_w, round(image.width * scale)), max(target_h, round(image.height * scale))),
        Image.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return np.asarray(resized.crop((left, top, left + target_w, top + target_h)),
                      dtype=np.float32)


def in_distribution_confidence(model_name: str, temperature: float) -> np.ndarray | None:
    """Calibrated confidence on the test set, for comparison."""
    path = config.METRICS_DIR / f"{model_name}_predictions.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    columns = [f"p_{c}" for c in config.CLASS_NAMES]
    if any(c not in frame.columns for c in columns):
        return None
    probs = apply_temperature(probs_to_logits(frame[columns].to_numpy()), temperature)
    return probs.max(axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=config.PRIMARY_MODEL, choices=[*config.MODEL_NAMES])
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N images (for a quick check)")
    args = parser.parse_args()

    paths = find_images(args.limit)
    calibration_path = config.calibration_path(args.model)
    if not calibration_path.exists():
        raise SystemExit(f"No calibration for {args.model}. Run: python src/calibrate.py")
    calibration = json.loads(calibration_path.read_text())
    temperature = float(calibration["temperature"])
    threshold = float(calibration["abstention_threshold"])

    model_path = config.model_path(args.model)
    if not model_path.exists():
        raise SystemExit(f"No trained model at {model_path}. Run: python src/train.py")

    from tensorflow import keras  # imported late so --help needs no GPU stack
    model = keras.models.load_model(model_path)

    print("=" * 78)
    print(f"OUT-OF-DISTRIBUTION TEST — {args.model}")
    print("=" * 78)
    print(f"Condition   : {', '.join(EXCLUDED_LEAF)} (excluded from every split)")
    print(f"Images      : {len(paths)}")
    print(f"Calibration : temperature {temperature:.4f}, abstention below {threshold:.3f}\n")

    batch, raw = [], []
    for path in paths:
        with Image.open(path) as image:
            batch.append(prepare(image))
        if len(batch) == config.BATCH_SIZE:
            raw.append(model.predict(np.stack(batch), verbose=0)); batch = []
    if batch:
        raw.append(model.predict(np.stack(batch), verbose=0))
    probs = apply_temperature(probs_to_logits(np.concatenate(raw)), temperature)
    confidence = probs.max(axis=1)
    predicted = [config.CLASS_NAMES[i] for i in probs.argmax(axis=1)]
    abstained = confidence < threshold

    # ---- 1. does the safety net catch them? --------------------------------
    print("-" * 78)
    print("1. DOES THE SYSTEM DECLINE TO ANSWER?")
    print("-" * 78)
    caught = int(abstained.sum())
    low, high = wilson_interval(caught, len(paths))
    print(f"  abstained (no condition named): {caught}/{len(paths)}  "
          f"{caught / len(paths):.1%}  [{low:.1%}, {high:.1%}]")
    if threshold <= 0:
        print("  NOTE: this model has no abstention threshold, so it cannot decline.")
    print("  This is the number that says whether the system knows it is out of its depth.")

    # ---- 2. what does it call the rest? ------------------------------------
    print("\n" + "-" * 78)
    print("2. WHAT LABEL DOES IT ASSIGN?")
    print("-" * 78)
    counts = Counter(predicted)
    print(f"  {'label':<26}{'all images':>14}{'answered only':>16}")
    answered = [p for p, a in zip(predicted, abstained) if not a]
    answered_counts = Counter(answered)
    for name in config.CLASS_NAMES:
        n_all, n_ans = counts.get(name, 0), answered_counts.get(name, 0)
        flag = ""
        if name == "healthy" and n_ans:
            flag = "   <- told nothing is wrong"
        print(f"  {config.CLASS_DISPLAY_NAMES[name]:<26}"
              f"{n_all:>7} {n_all / len(paths):>6.1%}"
              f"{n_ans:>9} {(n_ans / max(len(answered), 1)):>6.1%}{flag}")

    # ---- 3. what action would a user be given? -----------------------------
    print("\n" + "-" * 78)
    print("3. WHAT ACTION WOULD THE USER BE GIVEN?")
    print("-" * 78)
    bands = Counter()
    for row, flag in zip(probs, abstained):
        bands[triage.assess(row, abstained=bool(flag))["band"]] += 1
    for band in ("urgent", "prompt", "routine", "self_care"):
        n = bands.get(band, 0)
        flag = "   <- no action advised for a real condition" if band == "self_care" and n else ""
        print(f"  {config.TRIAGE_BANDS[band]['label']:<12}{n:>6} {n / len(paths):>7.1%}{flag}")
    routed = len(paths) - bands.get("self_care", 0)
    low, high = wilson_interval(routed, len(paths))
    print(f"\n  routed to some form of care: {routed}/{len(paths)}  {routed / len(paths):.1%}  "
          f"[{low:.1%}, {high:.1%}]")

    # ---- 4. is confidence lower than on known data? ------------------------
    print("\n" + "-" * 78)
    print("4. IS CONFIDENCE LOWER THAN ON DATA THE MODEL WAS TRAINED FOR?")
    print("-" * 78)
    known = in_distribution_confidence(args.model, temperature)
    print(f"  {'':<22}{'mean':>9}{'median':>9}{'below thr':>12}")
    print(f"  {'this condition':<22}{confidence.mean():>9.4f}{np.median(confidence):>9.4f}"
          f"{caught / len(paths):>11.1%}")
    if known is None:
        print("  (test-set predictions not found — run src/evaluate.py to enable comparison)")
    else:
        print(f"  {'test set (known)':<22}{known.mean():>9.4f}{np.median(known):>9.4f}"
              f"{(known < threshold).mean():>11.1%}")
        gap = known.mean() - confidence.mean()
        print(f"\n  confidence is {gap:+.4f} lower on the unknown condition.")
        if gap < 0.02:
            print("  That gap is small. Confidence carries little information about whether")
            print("  the input is something the model can represent, so no threshold on it")
            print("  would reliably separate the two.")

    out = pd.DataFrame({
        "path": [str(p.relative_to(config.PROJECT_ROOT)) for p in paths],
        "predicted": predicted,
        "confidence": confidence.round(6),
        "abstained": abstained.astype(int),
    })
    for index, name in enumerate(config.CLASS_NAMES):
        out[f"p_{name}"] = probs[:, index].round(6)
    path = config.METRICS_DIR / f"ood_{args.model}.csv"
    out.to_csv(path, index=False)
    print(f"\n-> {path.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
