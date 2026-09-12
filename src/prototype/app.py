"""Streamlit prototype: upload a foot/nail photo, get a predicted condition.

STATUS: not yet implemented — this is PHASE 6 of the project plan.

    streamlit run src/prototype/app.py

Planned interface
-----------------
  1. File uploader accepting JPG/PNG.
  2. The uploaded image, shown back with its resolution.
  3. The predicted class and the confidence for all four categories as a bar
     chart — not just the winner. A screening tool that says "ulcer, 34%" when
     the runner-up is 31% is telling the health worker something important, and
     a single-label output would hide it.
  4. A low-confidence warning when the top probability falls below a threshold,
     recommending referral rather than reliance on the prediction.
  5. A prominent, permanent disclaimer: this is a research prototype for an MSc
     dissertation, not a medical device, and it must not be used for diagnosis.
  6. A model selector (MobileNetV2 / ResNet50) so the two can be compared
     interactively during the viva demonstration.

The model is loaded once and cached (`@st.cache_resource`) rather than reloaded
on every interaction. The saved .keras file already contains its own
`preprocess_input` scaling, so this app only resizes and batches — it cannot
apply the wrong normalisation for the chosen architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main() -> int:
    print(__doc__)
    print(
        "This app is a placeholder for Phase 6 and does nothing yet.\n"
        "Train and evaluate a model (Phases 3-5) first."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
