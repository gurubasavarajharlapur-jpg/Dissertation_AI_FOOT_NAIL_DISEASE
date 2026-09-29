"""Entry point for the public demonstration of the screening prototype.

Built for the poster session. A visitor scans a code on the poster, the page
opens on their own phone, they photograph their own foot and read the result,
without the photograph ever passing through my laptop.

Nothing about the method lives here. The model loading, the preprocessing, the
temperature scaling, the triage bands and the report layout are all imported
from src/, so what a visitor sees is what Chapter 4 describes and what Chapter 5
measured. This file adds exactly two things: a shutter button, because a file
picker is the wrong input on a phone, and a line saying the photograph is not
kept.

The camera button is deliberately NOT added to src/prototype/app.py. Section
4.3.1 of the dissertation states that the prototype takes an upload or a test
split image, and the submitted code should say what the submitted document says.

Only MobileNetV2 is deployed. It is the architecture Chapter 6 recommends, so
the demonstration runs the model the work actually argues for, on a free CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import streamlit as st                                              # noqa: E402
from PIL import Image                                               # noqa: E402

# importing the prototype sets the page configuration, so it comes before any
# other Streamlit call
from src.prototype.app import (                                     # noqa: E402
    gradcam, load_calibration, load_model, overlay, predict, prepare,
    render_report,
)
from src import config                                              # noqa: E402

MODEL = "mobilenetv2"


def main() -> None:
    st.title("Foot and nail screening")
    st.caption("MSc dissertation prototype — Gurubasavaraj Harlapur, "
               "Middlesex University Dubai")
    st.error(config.DISCLAIMER)

    if not config.model_path(MODEL).exists():
        st.warning("The trained weights are not present in this deployment.")
        return

    calibration = load_calibration(MODEL)
    threshold = calibration["abstention_threshold"]

    st.info(
        "Photographs are used only to produce the result on this page. They are "
        "not saved, not logged and not used to train anything."
    )

    taken, uploaded = st.tabs(["Take a photograph", "Upload one"])
    with taken:
        shot = st.camera_input("Point at a foot or a nail", key="camera")
    with uploaded:
        picked = st.file_uploader("Choose an image", key="upload",
                                  type=["jpg", "jpeg", "png", "bmp", "webp"])

    # an UploadedFile is file-like, so test it explicitly rather than for truth
    source = shot if shot is not None else picked
    if source is None:
        st.stop()

    image = Image.open(source)
    array = prepare(image)
    model = load_model(MODEL)
    probs = predict(model, array, calibration["temperature"])

    order = np.argsort(probs)[::-1]
    top = int(order[0])
    confidence = float(probs[top])
    abstain = confidence < threshold

    render_report(probs, abstain, confidence, config.CLASS_NAMES[top], threshold,
                  MODEL, calibration)

    st.markdown("**All classes**")
    st.dataframe(
        pd.DataFrame({
            "Condition": [config.CLASS_DISPLAY_NAMES[config.CLASS_NAMES[i]]
                          for i in order],
            "Confidence": [float(probs[i]) * 100 for i in order],
        }),
        column_config={"Confidence": st.column_config.ProgressColumn(
            "Confidence", format="%.1f%%", min_value=0, max_value=100)},
        hide_index=True, use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### Where the model looked")
    with st.spinner("Computing Grad-CAM…"):
        cam = gradcam(model, MODEL, array, top)
    left, right = st.columns(2)
    left.image(array.astype(np.uint8), caption="What the model sees, 224×224",
               use_container_width=True)
    right.image(overlay(array, cam), caption="Red marks the deciding region",
                use_container_width=True)
    st.caption(
        "If the highlighted region sits on the background rather than on skin or "
        "nail, the prediction is not trustworthy however high its confidence."
    )

    st.markdown("---")
    st.caption(
        "Trained on clinical photographs. On photographs taken on an ordinary "
        "phone in an ordinary room it is measurably less accurate — Section 5.8 "
        "reports 6 of 10 on healthy feet — which is why it declines to name a "
        "condition when it is unsure rather than reassuring you."
    )


if __name__ == "__main__":
    main()
