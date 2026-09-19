"""Tune the triage escalation thresholds on the VALIDATION split.

The thresholds in config.py were set by judgement from the cost asymmetry, and
then measured on test. That makes them defensible but not tuned, and it also
means they must not be re-tuned against test: every other result in this project
depends on that split having been read exactly once. This script therefore reads
validation only, and will refuse to load the test predictions at all.

What it sweeps, and against what:

  TRIAGE_URGENT_PROB   P(ulcer) forcing the urgent band.
      benefit  ulcers reaching the urgent band — the right timeframe for the
               condition that can cost a limb
      cost     people needing no care sent for urgent attention

  TRIAGE_REVIEW_PROB   P(ulcer) + P(wound) forcing the prompt band.
      benefit  serious cases (ulcer or wound) referred at all
      cost     people needing no care referred at all

Lowering either threshold improves the benefit and worsens the cost, so the
sweep needs a stopping rule rather than an optimum. It uses the same shape as
the abstention threshold in calibrate.py: take the most protective value whose
cost stays inside a stated budget (config.TRIAGE_MAX_FALSE_URGENT and
TRIAGE_MAX_FALSE_REFERRAL). Without a budget "escalate everything" always wins,
and a tool that refers all comers is safe and useless.

Candidate thresholds are evaluated by calling the real triage.assess with the
value injected, not by reimplementing the rule — a reimplementation could drift
from what ships and quietly invalidate the tuning.

Usage:
    python src/sweep_thresholds.py
    python src/sweep_thresholds.py --model resnet50
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

SERIOUS = ("foot_ulcer", "foot_wound")
BENIGN = ("healthy", "nail_fungal")
REFERRING = ("urgent", "prompt")
GRID = np.round(np.arange(0.02, 0.51, 0.02), 2)


def load_validation(model_name: str) -> pd.DataFrame:
    path = config.METRICS_DIR / f"{model_name}_val_predictions.csv"
    if not path.exists():
        raise SystemExit(
            f"No validation predictions at {path}.\n"
            f"Run: python src/calibrate.py --model {model_name}"
        )
    frame = pd.read_csv(path)
    missing = [f"p_{c}" for c in config.CLASS_NAMES if f"p_{c}" not in frame.columns]
    if missing:
        raise SystemExit(f"{path.name} lacks {', '.join(missing)} — re-run src/calibrate.py")
    return frame


def bands_at(probs: np.ndarray, abstained: np.ndarray,
             urgent_prob: float, review_prob: float) -> np.ndarray:
    return np.array([
        triage.assess(row, abstained=bool(flag),
                      urgent_prob=urgent_prob, review_prob=review_prob)["band"]
        for row, flag in zip(probs, abstained)
    ])


def pick(rows: list[dict], benefit_key: str, cost_key: str,
         budget: float) -> tuple[dict | None, bool]:
    """Best attainable benefit within budget, at the least aggressive threshold.

    The obvious rule — take the lowest affordable threshold — is wrong, and
    running the sweep is what showed it. Escalation rules saturate: below some
    value every ulcer is already caught and lowering further changes nothing but
    still escalates more marginal cases. Taking the lowest affordable value then
    picks maximum aggression for zero benefit.

    So: find the best benefit any affordable candidate reaches, then among the
    candidates reaching it take the HIGHEST threshold. That is the knee of the
    curve — escalate as much as helps and no more. The second return value flags
    a benefit curve that never moved, where the choice is arbitrary within the
    affordable range and should be reported as such rather than defended.
    """
    affordable = [r for r in rows if r[cost_key] <= budget]
    if not affordable:
        return None, False
    best = max(r[benefit_key] for r in affordable)
    # A tolerance, because these are proportions over a few hundred cases and
    # a single image should not decide the operating point.
    at_best = [r for r in affordable if r[benefit_key] >= best - 1e-9]
    flat = abs(best - min(r[benefit_key] for r in affordable)) < 1e-9
    return max(at_best, key=lambda r: r["threshold"]), flat


def pct(count: int, total: int) -> str:
    if total == 0:
        return "    n/a"
    return f"{count:>4}/{total:<4} {count / total:>6.1%}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=config.PRIMARY_MODEL, choices=[*config.MODEL_NAMES])
    args = parser.parse_args()

    frame = load_validation(args.model)
    calibration_path = config.calibration_path(args.model)
    if not calibration_path.exists():
        raise SystemExit(f"No calibration for {args.model}. Run: python src/calibrate.py")
    calibration = json.loads(calibration_path.read_text())
    temperature = float(calibration["temperature"])
    threshold = float(calibration["abstention_threshold"])

    raw = frame[[f"p_{c}" for c in config.CLASS_NAMES]].to_numpy()
    probs = apply_temperature(probs_to_logits(raw), temperature)
    abstained = probs.max(axis=1) < threshold

    truth = frame["true"].to_numpy()
    predicted = frame["predicted"].to_numpy()
    is_ulcer = truth == "foot_ulcer"
    missed_ulcer = is_ulcer & (predicted != "foot_ulcer")
    is_serious = np.isin(truth, SERIOUS)
    is_benign = np.isin(truth, BENIGN)

    print("=" * 78)
    print(f"THRESHOLD SWEEP — {args.model}, VALIDATION split ({len(frame)} images)")
    print("=" * 78)
    print("Test is not read by this script. Tuning against it would spend the one")
    print("split every other result in this project depends on.\n")
    print(f"temperature {temperature:.4f} | abstention threshold {threshold:.3f}")
    print(f"validation composition: {int(is_ulcer.sum())} ulcers "
          f"({int(missed_ulcer.sum())} misclassified), {int(is_serious.sum())} serious, "
          f"{int(is_benign.sum())} benign\n")

    # ---- urgent threshold ---------------------------------------------------
    print("-" * 78)
    print("1. TRIAGE_URGENT_PROB — P(ulcer) forcing the urgent band")
    print("-" * 78)
    print(f"{'thresh':>7}{'ulcers at urgent':>22}{'missed ulcers at urgent':>27}"
          f"{'benign sent urgent':>22}")
    urgent_rows = []
    for candidate in GRID:
        bands = bands_at(probs, abstained, candidate, config.TRIAGE_REVIEW_PROB)
        at_urgent = bands == "urgent"
        row = {
            "threshold": float(candidate),
            "ulcer_urgency_recall": float(at_urgent[is_ulcer].mean()) if is_ulcer.any() else 0.0,
            "missed_ulcer_urgency": float(at_urgent[missed_ulcer].mean()) if missed_ulcer.any() else 0.0,
            "false_urgent": float(at_urgent[is_benign].mean()) if is_benign.any() else 0.0,
            "false_urgent_n": int(at_urgent[is_benign].sum()),
        }
        urgent_rows.append(row)
        flag = "" if row["false_urgent"] <= config.TRIAGE_MAX_FALSE_URGENT else "  over budget"
        print(f"{candidate:>7.2f}"
              f"{pct(int(at_urgent[is_ulcer].sum()), int(is_ulcer.sum())):>22}"
              f"{pct(int(at_urgent[missed_ulcer].sum()), int(missed_ulcer.sum())):>27}"
              f"{pct(row['false_urgent_n'], int(is_benign.sum())):>22}{flag}")

    chosen_urgent, urgent_flat = pick(urgent_rows, "ulcer_urgency_recall",
                                      "false_urgent", config.TRIAGE_MAX_FALSE_URGENT)
    print(f"\n  budget: at most {config.TRIAGE_MAX_FALSE_URGENT:.0%} of benign cases sent urgently")
    if chosen_urgent is None:
        print("  NO candidate meets the budget. Either the budget is too tight for this\n"
              "  model, or escalation is firing on benign cases far more than expected.")
    else:
        low, high = wilson_interval(
            int(round(chosen_urgent["missed_ulcer_urgency"] * missed_ulcer.sum())),
            int(missed_ulcer.sum()))
        label = ("no preference — curve flat" if urgent_flat
                 else "knee of the curve: best benefit, least aggressive")
        print(f"  -> {label}: {chosen_urgent['threshold']:.2f}")
        print(f"     misclassified ulcers reaching urgent: "
              f"{chosen_urgent['missed_ulcer_urgency']:.1%} [{low:.1%}, {high:.1%}]")
        print(f"     benign sent urgently: {chosen_urgent['false_urgent']:.1%} "
              f"({chosen_urgent['false_urgent_n']} of {int(is_benign.sum())})")
        if urgent_flat:
            print("     NOTE: the benefit curve never moves across the affordable range —\n"
                  "     on this validation split the threshold makes no difference to how\n"
                  "     many ulcers reach the urgent band. Any value in range is equally\n"
                  "     defensible, so the configured one stands; report it as untuned\n"
                  "     rather than as validated.")
        elif abs(chosen_urgent["threshold"] - config.TRIAGE_URGENT_PROB) < 1e-9:
            print(f"     this is the configured value — the judgement call is confirmed.")
        else:
            print(f"     configured value is {config.TRIAGE_URGENT_PROB:.2f}; "
                  f"the sweep prefers {chosen_urgent['threshold']:.2f}.")

    # ---- review threshold ---------------------------------------------------
    print("\n" + "-" * 78)
    print("2. TRIAGE_REVIEW_PROB — P(ulcer)+P(wound) forcing the prompt band")
    print("-" * 78)
    print(f"{'thresh':>7}{'serious referred':>22}{'benign referred':>22}")
    review_rows = []
    for candidate in GRID:
        bands = bands_at(probs, abstained, config.TRIAGE_URGENT_PROB, candidate)
        referred = np.isin(bands, REFERRING)
        row = {
            "threshold": float(candidate),
            "referral_recall": float(referred[is_serious].mean()) if is_serious.any() else 0.0,
            "false_referral": float(referred[is_benign].mean()) if is_benign.any() else 0.0,
            "false_referral_n": int(referred[is_benign].sum()),
        }
        review_rows.append(row)
        flag = "" if row["false_referral"] <= config.TRIAGE_MAX_FALSE_REFERRAL else "  over budget"
        print(f"{candidate:>7.2f}"
              f"{pct(int(referred[is_serious].sum()), int(is_serious.sum())):>22}"
              f"{pct(row['false_referral_n'], int(is_benign.sum())):>22}{flag}")

    chosen_review, review_flat = pick(review_rows, "referral_recall",
                                      "false_referral", config.TRIAGE_MAX_FALSE_REFERRAL)
    print(f"\n  budget: at most {config.TRIAGE_MAX_FALSE_REFERRAL:.0%} of benign cases referred")
    if chosen_review is None:
        print("  NO candidate meets the budget.")
    else:
        label = ("no preference — curve flat" if review_flat
                 else "knee of the curve: best benefit, least aggressive")
        print(f"  -> {label}: {chosen_review['threshold']:.2f}")
        print(f"     serious referred: {chosen_review['referral_recall']:.1%}")
        print(f"     benign referred: {chosen_review['false_referral']:.1%}")
        if review_flat:
            print("     NOTE: the benefit curve never moves across the affordable range;\n"
                  "     the configured value stands, untuned rather than validated.")
        elif abs(chosen_review["threshold"] - config.TRIAGE_REVIEW_PROB) < 1e-9:
            print(f"     this is the configured value — the judgement call is confirmed.")
        else:
            print(f"     configured value is {config.TRIAGE_REVIEW_PROB:.2f}; "
                  f"the sweep prefers {chosen_review['threshold']:.2f}.")

    # ---- what to do with it -------------------------------------------------
    print("\n" + "-" * 78)
    print("NEXT")
    print("-" * 78)
    changes = []
    if (chosen_urgent and not urgent_flat
            and abs(chosen_urgent["threshold"] - config.TRIAGE_URGENT_PROB) > 1e-9):
        changes.append(f"TRIAGE_URGENT_PROB = {chosen_urgent['threshold']:.2f}")
    if (chosen_review and not review_flat
            and abs(chosen_review["threshold"] - config.TRIAGE_REVIEW_PROB) > 1e-9):
        changes.append(f"TRIAGE_REVIEW_PROB = {chosen_review['threshold']:.2f}")
    if (urgent_flat or review_flat) and not changes:
        print("  At least one benefit curve is flat across the affordable range, so the\n"
              "  sweep cannot prefer a value. Keep the configured thresholds and report\n"
              "  them as set by judgement and bounded by this sweep — not as tuned.")
        changes = None
    if changes is None:
        pass
    elif changes:
        print("  The sweep prefers different values. To adopt them, edit src/config.py:")
        for line in changes:
            print(f"     {line}")
        print("  then re-run src/audit_triage.py to see the effect on test. Report both\n"
              "  the configured-by-judgement result and the tuned one — the difference\n"
              "  between them is itself a finding.")
    else:
        print("  The sweep confirms both configured values. Report that the thresholds\n"
              "  set by judgement survive tuning on validation, which is a stronger\n"
              "  claim than either alone.")

    out = pd.DataFrame(urgent_rows).merge(pd.DataFrame(review_rows), on="threshold")
    path = config.METRICS_DIR / f"threshold_sweep_{args.model}.csv"
    out.to_csv(path, index=False)
    print(f"\n-> {path.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
