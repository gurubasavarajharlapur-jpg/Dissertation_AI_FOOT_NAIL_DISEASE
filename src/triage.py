"""Turn a class prediction into a recommended action, biased toward referral.

Kept out of the prototype so the rules can be tested, quoted in the
dissertation, and reviewed by a clinician without reading Streamlit code.

The design principle, stated once because every rule here follows from it:
**the two errors are not equally costly.** Sending a healthy person to a clinic
wastes an appointment. Telling someone with a foot ulcer that nothing is wrong
can cost them a foot, and diabetic foot ulceration precedes roughly 80% of
non-traumatic lower-limb amputations. So where the evidence is ambiguous this
module escalates rather than reassures, and it never reports a clean result
without also saying what would override it.

Nothing here is learned. The model was trained on four class labels and has
never seen severity, infection, depth, perfusion or patient history, so it
cannot judge urgency. The bands are fixed clinical guidance attached to the
predicted class; the escalation rules are a fixed safety policy over the
probability vector. Both live in config.py and are a human decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

# Ordered most to least urgent, so "the higher of two bands" is a min() on index.
_SEVERITY = ["urgent", "prompt", "routine", "self_care"]


def _raise_band(current: str, candidate: str) -> str:
    """Return whichever band is more urgent. Escalation is one-directional."""
    return _SEVERITY[min(_SEVERITY.index(current), _SEVERITY.index(candidate))]


def assess(probs: np.ndarray, abstained: bool = False) -> dict:
    """Triage one prediction.

    `probs` is the calibrated probability vector in config.CLASS_NAMES order.
    `abstained` is True when confidence fell below the referral threshold.

    Returns the band, the action text, and — importantly for a report someone
    has to trust — the reason the band was chosen, including whether it was
    escalated above what the top class alone would have given.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (config.NUM_CLASSES,):
        raise ValueError(f"expected {config.NUM_CLASSES} probabilities, got {probs.shape}")

    top_index = int(probs.argmax())
    top_class = config.CLASS_NAMES[top_index]
    confidence = float(probs[top_index])

    ulcer = float(probs[config.CLASS_TO_INDEX["foot_ulcer"]])
    wound = float(probs[config.CLASS_TO_INDEX["foot_wound"]])

    base_band = config.CLASS_TRIAGE[top_class]
    band = base_band
    reasons: list[str] = []
    # The class whose guidance the report should show. When a result is
    # escalated this is NOT the top class: a report headed "URGENT" whose advice
    # reads "no concerning features detected" is worse than no report at all,
    # and that is precisely the falsely-reassuring failure this module exists to
    # prevent. The escalating condition owns the advice.
    escalation_class: str | None = None

    # 1. Abstention never reassures. A model that cannot decide is not evidence
    #    of absence, so an inconclusive result routes to a clinician rather than
    #    falling back on the most likely class.
    if abstained:
        band = _raise_band(band, "routine")
        reasons.append(
            "The model's confidence was below the referral threshold, so no "
            "condition is being reported. An inconclusive screening is not a "
            "negative result."
        )

    # 2. A serious condition holding real probability escalates even when it is
    #    not the top class — the case this whole module exists for.
    if ulcer >= config.TRIAGE_URGENT_PROB:
        band = _raise_band(band, "urgent")
        if top_class != "foot_ulcer":
            escalation_class = "foot_ulcer"
            reasons.append(
                f"Features consistent with a foot ulcer carry {ulcer:.0%} "
                f"probability. That is below the top class but high enough to "
                f"act on, because a missed ulcer is the costly error."
            )
    elif ulcer + wound >= config.TRIAGE_REVIEW_PROB:
        band = _raise_band(band, "prompt")
        if top_class not in ("foot_ulcer", "foot_wound"):
            escalation_class = "foot_wound"
            reasons.append(
                f"Features consistent with a wound or ulcer carry "
                f"{ulcer + wound:.0%} combined probability and warrant review "
                f"even though another class scored higher."
            )

    escalated = band != base_band
    # Escalating condition first, then the top class, then the inconclusive
    # text. Anything else can pair an urgent banner with reassuring advice.
    guidance_class = escalation_class or (None if abstained else top_class)
    guidance = (config.CLINICAL_GUIDANCE[guidance_class] if guidance_class
                else config.UNCERTAIN_GUIDANCE)
    if not reasons:
        reasons.append(
            f"Screening appearance is consistent with "
            f"{config.CLASS_DISPLAY_NAMES[top_class]}."
        )

    return {
        "band": band,
        "band_label": config.TRIAGE_BANDS[band]["label"],
        "timeframe": config.TRIAGE_BANDS[band]["timeframe"],
        "colour": config.TRIAGE_BANDS[band]["colour"],
        "top_class": None if abstained else top_class,
        "confidence": confidence,
        "abstained": bool(abstained),
        "escalated": escalated,
        "base_band": base_band,
        "guidance_class": guidance_class,
        "reasons": reasons,
        "guidance": guidance,
        "safety_net": config.SAFETY_NET,
    }
