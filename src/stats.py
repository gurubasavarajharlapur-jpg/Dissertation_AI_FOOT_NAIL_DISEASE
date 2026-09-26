"""Numpy-only statistics shared across the analysis scripts.

Nothing here imports TensorFlow, which is the point: every result computed from
saved predictions — paired tests, confidence intervals, temperature scaling —
can then be recomputed in seconds on a CPU without loading a model or
re-running inference.

Both tests here are *paired*. Two models scored on the same images do not
produce independent results — an image that is hard for one is usually hard
for the other — so comparing two accuracy figures as though they were
independent samples overstates the evidence. Pairing removes the shared
difficulty and asks only about the images where the models actually differ.
"""
from __future__ import annotations

import numpy as np


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar's exact test on two models scored over the same images.

    Only the images the models disagree on carry information: b is the count
    the first gets right and the second wrong, c the reverse. Under the null
    that the two are equally accurate, each disagreement is a fair coin, so the
    exact binomial test on (b, b + c) is the p-value. Images both get right or
    both get wrong are uninformative and are correctly discarded.

    The exact test is used rather than the chi-square approximation because the
    discordant counts here are small, which is where the approximation is least
    reliable.
    """
    correct_a = np.asarray(correct_a)
    correct_b = np.asarray(correct_b)
    if correct_a.shape != correct_b.shape:
        raise ValueError("the two models must be scored on the same images")

    b = int(np.sum((correct_a == 1) & (correct_b == 0)))
    c = int(np.sum((correct_a == 0) & (correct_b == 1)))
    result = {"n": int(len(correct_a)), "only_first_correct": b,
              "only_second_correct": c, "discordant": b + c}
    if b + c == 0:
        result["p_value"] = 1.0
        return result
    from scipy.stats import binomtest
    result["p_value"] = float(binomtest(b, b + c, 0.5).pvalue)
    return result


def paired_bootstrap_ci(correct_a: np.ndarray, correct_b: np.ndarray,
                        resamples: int = 10000, seed: int = 42) -> dict:
    """Confidence interval for the accuracy difference (b - a), paired.

    Resampling *images* rather than the two models separately keeps each pair
    together, which is what makes the interval comparable to McNemar's test.
    The interval answers the question the p-value does not: not only "is there
    a difference" but "how large could it plausibly be", which is the more
    useful form for a deployment decision.
    """
    correct_a = np.asarray(correct_a, dtype=float)
    correct_b = np.asarray(correct_b, dtype=float)
    n = len(correct_a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    diffs = correct_b[idx].mean(axis=1) - correct_a[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "observed": float(correct_b.mean() - correct_a.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "resamples": resamples,
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion, Wilson's method.

    Used rather than the textbook normal approximation because that one
    misbehaves exactly where a small external validation set lands: near 0 or
    1, and at small n, it produces intervals running past 100% or of zero
    width. Wilson stays inside [0, 1] and stays sensible on a handful of
    images, which is the situation this is for.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denominator
    spread = z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
EPS = 1e-12


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Recover logits from softmax outputs.

    Softmax is invariant to an additive constant, so log(p) is a valid set of
    logits: softmax(log(p)) == p exactly. That is all temperature scaling needs,
    and it avoids rebuilding the models without their softmax layer.
    """
    return np.log(np.clip(probs, EPS, 1.0))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax at the given temperature. T > 1 softens, T < 1 sharpens."""
    scaled = logits / temperature
    scaled -= scaled.max(axis=1, keepdims=True)  # stabilise before exponentiating
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)
