"""Measure what the triage policy actually does to the test set.

The escalation rules in src/triage.py are a design decision biased toward
referral. That bias has a price, and a design decision whose price has not been
measured is an assertion. This script measures it, from the saved prediction
CSVs — no GPU, no TensorFlow, no re-run of inference.

Three questions, in order of how much they matter:

  1. **Does the safety net catch the ulcers the classifier misses?** Every ulcer
     the model calls something else is the error this project cares about. If
     escalation still routes those people to care, the policy earns its place.
  2. **What does it cost?** Every healthy person escalated to "prompt" or
     "urgent" is an unnecessary clinic visit. A screening tool that refers
     everyone is safe and useless.
  3. **Is the triage policy better than the classifier alone?** Treating this as
     the binary question a health worker actually asks — does this person need
     to be seen? — how do sensitivity and specificity compare between referring
     on the predicted class and referring on the triage band?

Usage:
    python src/audit_triage.py
    python src/audit_triage.py --model resnet50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, triage  # noqa: E402
from src.stats import apply_temperature, probs_to_logits, wilson_interval  # noqa: E402

# Conditions that need to be seen by someone. nail_fungal is excluded: it is a
# routine referral, not a reason to attend, and folding it in would flatter the
# sensitivity figures by counting a non-urgent class as a catch.
SERIOUS = ("foot_ulcer", "foot_wound")
REFERRING_BANDS = ("urgent", "prompt")


def load(model_name: str) -> pd.DataFrame:
    path = config.METRICS_DIR / f"{model_name}_predictions.csv"
    if not path.exists():
        raise SystemExit(f"No predictions at {path}. Run: python src/evaluate.py")
    frame = pd.read_csv(path)
    missing = [f"p_{c}" for c in config.CLASS_NAMES if f"p_{c}" not in frame.columns]
    if missing:
        raise SystemExit(
            f"{path.name} has no per-class probability columns ({', '.join(missing)}).\n"
            f"It predates this analysis — re-run: python src/evaluate.py"
        )
    return frame


def calibration_for(model_name: str) -> tuple[float, float, bool]:
    path = config.calibration_path(model_name)
    if not path.exists():
        print(f"NOTE: no calibration file for {model_name}; using raw probabilities "
              f"and no abstention. Run src/calibrate.py for the deployed behaviour.")
        return 1.0, 0.0, False
    data = json.loads(path.read_text())
    return float(data["temperature"]), float(data["abstention_threshold"]), True


def rate(count: int, total: int) -> str:
    if total == 0:
        return "   n/a"
    low, high = wilson_interval(count, total)
    return f"{count:>4}/{total:<5} {count / total:>6.1%}  [{low:.1%}, {high:.1%}]"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=config.PRIMARY_MODEL,
                        choices=[*config.MODEL_NAMES])
    args = parser.parse_args()

    frame = load(args.model)
    temperature, threshold, calibrated = calibration_for(args.model)

    raw = frame[[f"p_{c}" for c in config.CLASS_NAMES]].to_numpy()
    probs = apply_temperature(probs_to_logits(raw), temperature)
    confidence = probs.max(axis=1)
    abstained = confidence < threshold

    bands, escalated, base_bands = [], [], []
    for row, is_abstained in zip(probs, abstained):
        result = triage.assess(row, abstained=bool(is_abstained))
        bands.append(result["band"])
        base_bands.append(result["base_band"])
        escalated.append(result["escalated"])
    frame["band"] = bands
    frame["base_band"] = base_bands
    frame["escalated"] = escalated
    frame["referred"] = frame["band"].isin(REFERRING_BANDS)
    frame["truly_serious"] = frame["true"].isin(SERIOUS)
    frame["predicted_serious"] = frame["predicted"].isin(SERIOUS)

    n = len(frame)
    print("=" * 78)
    print(f"TRIAGE AUDIT — {args.model}, {n} test images")
    print("=" * 78)
    print(f"temperature {temperature:.4f} | abstention threshold {threshold:.3f}"
          f"{'' if calibrated else '  (UNCALIBRATED)'}")
    print(f"escalation rules: P(ulcer) >= {config.TRIAGE_URGENT_PROB:.0%} -> urgent; "
          f"P(ulcer)+P(wound) >= {config.TRIAGE_REVIEW_PROB:.0%} -> prompt\n")

    # ---- 1. How often does it fire? ----------------------------------------
    fired = int(frame["escalated"].sum())
    print("-" * 78)
    print("1. HOW OFTEN ESCALATION FIRES")
    print("-" * 78)
    print(f"  escalated: {rate(fired, n)}")
    if fired:
        moves = (frame[frame["escalated"]]
                 .groupby(["base_band", "band"]).size().sort_values(ascending=False))
        for (was, now), count in moves.items():
            print(f"    {was:<10} -> {now:<10} {count:>4}")
        by_true = frame[frame["escalated"]].groupby("true").size()
        print("  by true class:")
        for name in config.CLASS_NAMES:
            print(f"    {config.CLASS_DISPLAY_NAMES[name]:<24}{int(by_true.get(name, 0)):>5}")

    # ---- 2. Does it catch the missed ulcers? -------------------------------
    print("\n" + "-" * 78)
    print("2. THE ULCERS THE CLASSIFIER GETS WRONG  <- the question that matters")
    print("-" * 78)
    ulcers = frame[frame["true"] == "foot_ulcer"]
    missed = ulcers[ulcers["predicted"] != "foot_ulcer"]
    print(f"  ulcers in test set        : {len(ulcers)}")
    print(f"  misclassified by the model: {len(missed)}")
    if len(missed):
        caught = int(missed["referred"].sum())
        print(f"  still routed to care      : {rate(caught, len(missed))}")
        print()
        print(f"    {'predicted as':<24}{'band':<12}{'referred':<10}top prob")
        for _, row in missed.iterrows():
            print(f"    {config.CLASS_DISPLAY_NAMES[row['predicted']]:<24}"
                  f"{config.TRIAGE_BANDS[row['band']]['label']:<12}"
                  f"{'YES' if row['referred'] else 'NO — MISSED':<10}"
                  f"{row['confidence']:.1%}")
        if caught < len(missed):
            print(f"\n  {len(missed) - caught} ulcer(s) reach a patient as 'no referral "
                  f"needed'. This is the number to state in the limitations.")

    # ---- 3. What does it cost? ---------------------------------------------
    print("\n" + "-" * 78)
    print("3. COST — people referred who did not need it")
    print("-" * 78)
    for name in ("healthy", "nail_fungal"):
        group = frame[frame["true"] == name]
        print(f"  {config.CLASS_DISPLAY_NAMES[name]:<24} referred: "
              f"{rate(int(group['referred'].sum()), len(group))}")

    # ---- 4. Policy comparison ----------------------------------------------
    print("\n" + "-" * 78)
    print("4. TRIAGE POLICY vs CLASSIFIER ALONE")
    print("-" * 78)
    print('  "Does this person need to be seen?" — serious = ulcer or wound.\n')
    serious = frame["truly_serious"].to_numpy()
    print(f"  {'policy':<26}{'sensitivity':>26}{'specificity':>26}")
    for label, referred in (("classifier alone", frame["predicted_serious"].to_numpy()),
                            ("triage bands", frame["referred"].to_numpy())):
        sens = rate(int((referred & serious).sum()), int(serious.sum()))
        spec = rate(int((~referred & ~serious).sum()), int((~serious).sum()))
        print(f"  {label:<26}{sens:>26}{spec:>26}")
    print("\n  Sensitivity is the safety number: the fraction of people who needed")
    print("  care that the tool sent. Specificity is the cost: the fraction of")
    print("  people who did not need care that it correctly left alone.")

    path = config.METRICS_DIR / f"triage_audit_{args.model}.csv"
    frame.to_csv(path, index=False)
    print(f"\n-> {path.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
