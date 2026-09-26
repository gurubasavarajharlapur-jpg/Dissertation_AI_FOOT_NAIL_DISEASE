"""Paired comparison of two trained models from their saved predictions.

Runs in seconds on a CPU. It reads results/metrics/<model>_predictions.csv,
written by evaluate.py, so the statistical comparison can be repeated, extended
or checked without re-running inference or occupying a GPU.

Overall accuracy answers "which model scored higher". It does not answer the
three questions a comparison actually turns on, which is what this reports:

  1. Is the overall difference established, or within sampling noise?
  2. Is the difference on FOOT ULCER established? That is the clinically
     costly error, and it is a separate question from overall accuracy — a
     model can be better overall and no better on the class that matters.
  3. Does the difference survive on mendeley_foot, the one source holding two
     classes and therefore the only comparison where recognising the dataset
     cannot substitute for recognising the condition?

Usage:
    python src/compare_models.py
    python src/compare_models.py --models mobilenetv2 resnet50
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
from src.stats import mcnemar, paired_bootstrap_ci  # noqa: E402


def load(model_name: str) -> pd.DataFrame:
    path = config.METRICS_DIR / f"{model_name}_predictions.csv"
    if not path.exists():
        raise SystemExit(
            f"No predictions for {model_name} at {path}.\n"
            f"Run: python src/evaluate.py --model {model_name}"
        )
    return pd.read_csv(path)


def align(first: pd.DataFrame, second: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fail loudly if the two files do not describe the same test images.

    A silent mismatch here would invalidate every paired test below, so this is
    checked rather than assumed — the two files could easily come from
    different splits if one model was evaluated before a re-split.
    """
    if len(first) != len(second):
        raise SystemExit(f"prediction files differ in length ({len(first)} vs {len(second)}) "
                         "— they are not from the same test split. Re-run evaluate.py.")
    if not (first["path"].to_numpy() == second["path"].to_numpy()).all():
        raise SystemExit("prediction files list different images — re-run evaluate.py "
                         "so both models are scored on the same split.")
    if not (first["true"].to_numpy() == second["true"].to_numpy()).all():
        raise SystemExit("prediction files disagree on the true labels — re-run evaluate.py.")
    return first, second


def report(label: str, a_name: str, b_name: str,
           a_correct: np.ndarray, b_correct: np.ndarray, indent: str = "") -> dict:
    test = mcnemar(a_correct, b_correct)
    n = len(a_correct)
    a_acc, b_acc = a_correct.mean(), b_correct.mean()
    verdict = ("established" if test["p_value"] < 0.05
               else "not established — within sampling noise")
    print(f"{indent}{label}")
    print(f"{indent}  n = {n};  {a_name} {a_acc:.4f} ({int(n - a_correct.sum())} errors)"
          f"   {b_name} {b_acc:.4f} ({int(n - b_correct.sum())} errors)")
    print(f"{indent}  disagree on {test['discordant']}: "
          f"{test['only_second_correct']} to {b_name}, "
          f"{test['only_first_correct']} to {a_name}")
    print(f"{indent}  McNemar exact p = {test['p_value']:.4f}  ->  {verdict}")
    return {"n": n, f"{a_name}_accuracy": float(a_acc),
            f"{b_name}_accuracy": float(b_acc), **test}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs=2, default=list(config.MODEL_NAMES),
                        metavar=("FIRST", "SECOND"))
    args = parser.parse_args()
    a_name, b_name = args.models

    first, second = align(load(a_name), load(b_name))
    a = first["correct"].to_numpy()
    b = second["correct"].to_numpy()
    labels = first["true"].to_numpy()
    sources = first["source"].to_numpy()

    print("=" * 74)
    print(f"PAIRED COMPARISON — {a_name} vs {b_name}")
    print("=" * 74)
    print(f"Both models scored on the same {len(a)} test images, so every test "
          f"below is paired.\n")

    out: dict = {"models": [a_name, b_name]}

    out["overall"] = report("OVERALL", a_name, b_name, a, b)
    ci = paired_bootstrap_ci(a, b)
    print(f"  accuracy difference {ci['observed']:+.4f}, "
          f"95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] "
          f"(paired bootstrap, {ci['resamples']} resamples)")
    if ci["ci_low"] <= 0 <= ci["ci_high"]:
        print("  The interval spans zero, which agrees with the p-value above.")
    out["overall"]["accuracy_difference_ci"] = ci

    print(f"\n{'-' * 74}\nBY CLASS (recall — restricted to images whose true label is that class)")
    print(f"{'-' * 74}")
    out["by_class"] = {}
    for name in config.CLASS_NAMES:
        mask = labels == name
        if not mask.any():
            continue
        flag = "   <- the clinically costly error" if name == "foot_ulcer" else ""
        out["by_class"][name] = report(
            f"{config.CLASS_DISPLAY_NAMES[name]}{flag}", a_name, b_name,
            a[mask], b[mask], indent="  ")
        print()

    print(f"{'-' * 74}\nBY SOURCE")
    print(f"{'-' * 74}")
    out["by_source"] = {}
    for source in sorted(set(sources)):
        mask = sources == source
        classes_here = sorted(set(labels[mask]))
        note = ("" if len(classes_here) > 1
                else "   (single class — provenance alone could separate it)")
        out["by_source"][source] = report(
            f"{source}{note}", a_name, b_name, a[mask], b[mask], indent="  ")
        print()

    two_class = [s for s in sorted(set(sources)) if len(set(labels[sources == s])) > 1]
    if two_class:
        print("The source(s) above holding more than one class are the confound-free")
        print("comparison: there, recognising the dataset cannot substitute for")
        print("recognising the condition.")

    path = config.METRICS_DIR / f"paired_comparison_{a_name}_vs_{b_name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\n-> {path.relative_to(config.PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
