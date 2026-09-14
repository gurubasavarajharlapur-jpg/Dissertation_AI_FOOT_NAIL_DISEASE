"""Streamlit prototype: foot and nail condition screening (WBS 6).

    streamlit run src/prototype/app.py

Two tabs:

  Screening       Upload one photograph and receive a predicted condition with
                  calibrated confidence, a Grad-CAM overlay showing the region
                  that drove the decision, and guidance keyed to the predicted
                  class. Below the abstention threshold it declines to name a
                  condition and recommends consulting a clinician.

  Batch evaluation
                  Score a whole labelled set and compare it against the stored
                  test-set result. Either the project's own test split, or a zip
                  of independently captured photographs organised into folders
                  by class. The second is the external validation that 2.5.8
                  identifies as missing from most published work: images from a
                  different camera, lighting and population than any training
                  source.

Two boundaries this interface is careful about, and which should be stated in
the dissertation:

  The model classifies appearance. It has never seen severity, infection status,
  wound depth or patient history, so it cannot judge urgency. The guidance text
  is a fixed clinical mapping keyed to the predicted CLASS (config.CLINICAL_
  GUIDANCE), not a model output.

  Confidence shown is temperature-scaled using the value fitted on the
  validation split by src/calibrate.py. Raw softmax outputs are systematically
  overconfident and would mislead exactly the user this tool is aimed at.
"""

from __future__ import annotations

import io
import json
import sys
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config  # noqa: E402

st.set_page_config(page_title="Foot & Nail Screening — MSc Prototype",
                   page_icon="🦶", layout="wide")


# ---------------------------------------------------------------------------
# Loading (cached: reloading a 200MB model per interaction is unusable)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(model_name: str):
    from tensorflow import keras
    return keras.models.load_model(config.model_path(model_name))


@st.cache_data
def load_calibration(model_name: str) -> dict:
    """Fitted temperature and abstention threshold, if calibration has been run.

    Falls back to raw softmax with no abstention, and the UI says so — silently
    showing uncalibrated confidence as though it were calibrated would defeat
    the purpose.
    """
    path = config.calibration_path(model_name)
    if path.exists():
        return json.loads(path.read_text())
    return {"temperature": 1.0, "abstention_threshold": 0.0, "_missing": True}


def available_models() -> list[str]:
    return [m for m in config.MODEL_NAMES if config.model_path(m).exists()]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def prepare(image: Image.Image) -> np.ndarray:
    """Resize exactly as src/preprocessing.py does, so inference matches training."""
    image = image.convert("RGB")
    target_w, target_h = config.IMAGE_SIZE
    scale = max(target_w / image.width, target_h / image.height)
    resized = image.resize(
        (max(target_w, round(image.width * scale)), max(target_h, round(image.height * scale))),
        Image.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    cropped = resized.crop((left, top, left + target_w, top + target_h))
    return np.asarray(cropped, dtype=np.float32)


def predict(model, array: np.ndarray, temperature: float) -> np.ndarray:
    """Class probabilities for one image, temperature-scaled."""
    raw = model.predict(array[None], verbose=0)[0]
    if abs(temperature - 1.0) < 1e-6:
        return raw
    logits = np.log(np.clip(raw, 1e-12, 1.0)) / temperature
    logits -= logits.max()
    exponentiated = np.exp(logits)
    return exponentiated / exponentiated.sum()


def gradcam(model, model_name: str, array: np.ndarray, class_index: int) -> np.ndarray:
    import tensorflow as tf
    from src.evaluate import gradcam_heatmap
    cam, _ = gradcam_heatmap(model, model_name, array, class_index=class_index)
    return cam


def overlay(image: np.ndarray, cam: np.ndarray) -> np.ndarray:
    import matplotlib
    resized = np.asarray(
        Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]), Image.BILINEAR
        )
    ) / 255.0
    heat = matplotlib.colormaps["jet"](resized)[..., :3]
    return np.clip(0.65 * (image / 255.0) + 0.35 * heat, 0, 1)


# ---------------------------------------------------------------------------
# Tab 1: screening
# ---------------------------------------------------------------------------
@st.cache_data
def test_set_index() -> pd.DataFrame:
    """The test split, for the sample picker. Empty frame if not prepared."""
    path = config.PROCESSED_DIR / "test.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["path", "label"])


def screening_tab(model_name: str) -> None:
    calibration = load_calibration(model_name)
    threshold = calibration["abstention_threshold"]

    samples = test_set_index()
    # The file uploader is a browser picker, so it only reaches the viewer's own
    # machine. When the app runs on Colab the test images are on the server and
    # unreachable that way, which makes the second option the only practical way
    # to try a known image without downloading it first.
    mode = st.radio(
        "Image source",
        ["Upload a photograph", "Pick one from the test set"],
        horizontal=True,
        disabled=samples.empty,
        key="source_mode",
    )

    image = None
    known_label = None

    if mode == "Upload a photograph" or samples.empty:
        uploaded = st.file_uploader(
            "Upload a photograph of a foot or nail",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="single",
        )
        if uploaded is not None:
            image = Image.open(uploaded)
    else:
        pick_class = st.selectbox(
            "Class",
            config.CLASS_NAMES,
            format_func=lambda c: config.CLASS_DISPLAY_NAMES[c],
            key="sample_class",
        )
        subset = samples[samples["label"] == pick_class].reset_index(drop=True)
        if subset.empty:
            st.warning(f"No test images for {config.CLASS_DISPLAY_NAMES[pick_class]}.")
            return
        index = st.slider("Image", 0, len(subset) - 1, 0, key="sample_index")
        row = subset.iloc[index]
        image = Image.open(config.PROJECT_ROOT / row["path"])
        known_label = pick_class
        st.caption(f"{Path(row['path']).name}  —  true class: "
                   f"**{config.CLASS_DISPLAY_NAMES[pick_class]}**  "
                   f"({index + 1} of {len(subset)})")

    if image is None:
        st.info("Choose an image to begin. Any photograph works — a dataset image "
                "or one taken on your phone.")
        return

    array = prepare(image)
    model = load_model(model_name)
    probs = predict(model, array, calibration["temperature"])

    order = np.argsort(probs)[::-1]
    top = int(order[0])
    top_name = config.CLASS_NAMES[top]
    confidence = float(probs[top])
    abstain = confidence < threshold

    left, right = st.columns([1, 1.15])
    with left:
        st.image(image, caption=f"{image.width}×{image.height}", use_container_width=True)

    with right:
        if abstain:
            st.warning("### Uncertain — no condition reported")
            st.write(config.UNCERTAIN_GUIDANCE)
            st.caption(
                f"Top candidate {config.CLASS_DISPLAY_NAMES[top_name]} at {confidence:.0%}, "
                f"below the {threshold:.0%} threshold at which this model's predictions "
                f"reach the required accuracy."
            )
        else:
            st.success(f"### {config.CLASS_DISPLAY_NAMES[top_name]}")
            # Never display 100%. A rounded 0.9997 shown as certainty overclaims,
            # and sits badly beside a dissertation section establishing that this
            # model's confidence needed correcting at all.
            shown = ">99.9%" if confidence > 0.999 else f"{confidence:.1%}"
            st.metric("Calibrated confidence", shown)
            st.write(config.CLINICAL_GUIDANCE[top_name])
            st.caption(
                "Guidance is fixed clinical advice for this class, not a model "
                "assessment of severity. The model classifies appearance only."
            )

        if known_label is not None:
            if known_label == top_name and not abstain:
                st.caption(f"Correct — the true class is "
                           f"{config.CLASS_DISPLAY_NAMES[known_label]}.")
            elif abstain:
                st.caption(f"Abstained. The true class is "
                           f"{config.CLASS_DISPLAY_NAMES[known_label]}.")
            else:
                st.caption(f"Incorrect — the true class is "
                           f"**{config.CLASS_DISPLAY_NAMES[known_label]}**.")

        st.markdown("**All classes**")
        # Percentages, not the raw 0-1 values: ProgressColumn formats whatever it
        # is given, so a probability of 0.505 with a "%%" format renders as
        # "0.5%" directly beneath a headline reading 50.5%.
        st.dataframe(
            pd.DataFrame({
                "Condition": [config.CLASS_DISPLAY_NAMES[config.CLASS_NAMES[i]] for i in order],
                "Confidence": [float(probs[i]) * 100 for i in order],
            }),
            column_config={"Confidence": st.column_config.ProgressColumn(
                "Confidence", format="%.1f%%", min_value=0, max_value=100)},
            hide_index=True, use_container_width=True,
        )

    if calibration.get("_missing"):
        st.info(
            "Confidence is raw softmax — no calibration file found, so no abstention "
            "threshold is applied. Run `python src/calibrate.py` for calibrated "
            "confidence. Uncalibrated deep networks are typically overconfident."
        )

    st.markdown("---")
    st.markdown("#### Where the model looked")
    with st.spinner("Computing Grad-CAM…"):
        cam = gradcam(model, model_name, array, top)
    cam_left, cam_right = st.columns(2)
    cam_left.image(array.astype(np.uint8), caption="Input (as the model sees it, 224×224)",
                   use_container_width=True)
    cam_right.image(overlay(array, cam), caption="Grad-CAM — red marks the deciding region",
                    use_container_width=True)
    st.caption(
        "If the highlighted region sits on the background, a ruler or an image "
        "border rather than on skin or nail, the prediction is not trustworthy "
        "however high its confidence."
    )


# ---------------------------------------------------------------------------
# Tab 2: batch evaluation
# ---------------------------------------------------------------------------
def _score(model, model_name: str, items: list[tuple[np.ndarray, int]],
           temperature: float, threshold: float) -> dict:
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    arrays = np.stack([a for a, _ in items])
    y_true = np.array([label for _, label in items])

    raw = model.predict(arrays, batch_size=config.BATCH_SIZE, verbose=0)
    if abs(temperature - 1.0) > 1e-6:
        logits = np.log(np.clip(raw, 1e-12, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        probs = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    else:
        probs = raw

    y_pred = probs.argmax(axis=1)
    labels = list(range(config.NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    accepted = probs.max(axis=1) >= threshold
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f1)),
        "macro_recall": float(np.mean(recall)),
        "per_class": pd.DataFrame({
            "Condition": [config.CLASS_DISPLAY_NAMES[c] for c in config.CLASS_NAMES],
            "Precision": precision, "Recall": recall, "F1": f1, "Support": support,
        }),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "coverage": float(accepted.mean()),
        "accuracy_accepted": float((y_pred[accepted] == y_true[accepted]).mean())
        if accepted.any() else float("nan"),
    }


def _folder_key(name: str) -> str:
    """Normalise a folder name so obvious spellings of a class still match.

    Someone organising photographs by hand will write "Foot Ulcer" or
    "foot-ulcer" as readily as "foot_ulcer", and on a case-sensitive filesystem
    the exact-match version of this silently discarded the lot. Lower-case and
    collapse every run of non-alphanumeric characters to one underscore.
    """
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


# Accepted spellings -> class index. Both the internal name and the display
# name are registered, so "foot_ulcer", "Foot Ulcer" and "Foot-Ulcer" all land
# on the same class.
FOLDER_ALIASES: dict[str, int] = {}
for _name, _index in config.CLASS_TO_INDEX.items():
    FOLDER_ALIASES[_folder_key(_name)] = _index
    FOLDER_ALIASES[_folder_key(config.CLASS_DISPLAY_NAMES[_name])] = _index


def _load_zip(data: bytes) -> tuple[list[tuple[np.ndarray, int]], list[str]]:
    """Read images from a zip organised as <class_name>/<image>."""
    items, notes = [], []
    unknown: set[str] = set()
    unreadable: dict[str, int] = {}
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in zf.namelist():
            path = Path(name)
            if path.is_dir() or name.startswith("__MACOSX/") or path.name.startswith("."):
                continue
            # The class is the nearest parent folder naming a project class.
            label = next((FOLDER_ALIASES[key] for key in
                          (_folder_key(part) for part in path.parts[::-1])
                          if key in FOLDER_ALIASES), None)
            if label is None:
                unknown.add(path.parent.as_posix())
                continue
            if path.suffix.lower() not in config.VALID_EXTENSIONS:
                # Counted and reported rather than dropped. A phone shooting
                # HEIC would otherwise produce an empty result with no
                # explanation of why.
                unreadable[path.suffix.lower() or "(no extension)"] = (
                    unreadable.get(path.suffix.lower() or "(no extension)", 0) + 1)
                continue
            with zf.open(name) as handle:
                items.append((prepare(Image.open(io.BytesIO(handle.read()))), label))
    if unknown:
        notes.append(
            f"{len(unknown)} folder(s) skipped — names must match a project class "
            f"({', '.join(config.CLASS_NAMES)}): {sorted(unknown)[:4]}"
        )
    if unreadable:
        listed = ", ".join(f"{n}x {ext}" for ext, n in sorted(unreadable.items()))
        notes.append(
            f"{sum(unreadable.values())} file(s) skipped — unsupported format ({listed}). "
            f"iPhones save HEIC by default: set Settings > Camera > Formats > "
            f"Most Compatible, or convert to JPEG before zipping."
        )
    return items, notes


def _load_test_split() -> list[tuple[np.ndarray, int]]:
    frame = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    return [
        (np.asarray(Image.open(config.PROJECT_ROOT / row.path).convert("RGB"), dtype=np.float32),
         config.CLASS_TO_INDEX[row.label])
        for row in frame.itertuples()
    ]


def batch_tab(model_name: str) -> None:
    calibration = load_calibration(model_name)
    st.markdown(
        "Score a labelled set of images. Comparing the project's own test split "
        "against photographs captured independently is the external validation "
        "step §2.5.8 identifies as missing from most published work."
    )

    source = st.radio(
        "What to evaluate",
        ["Project test split", "Upload my own photographs (zip)"],
        horizontal=True,
    )

    items: list[tuple[np.ndarray, int]] = []
    if source == "Project test split":
        if not (config.PROCESSED_DIR / "test.csv").exists():
            st.error("No test split found. Run `python src/preprocessing.py` first.")
            return
        if st.button("Evaluate test split", type="primary"):
            with st.spinner("Loading test images…"):
                items = _load_test_split()
    else:
        st.caption(
            "Zip organised one folder per class, using these exact names: "
            f"`{'`, `'.join(config.CLASS_NAMES)}`. Photographs of other people "
            "need their consent, and should not be committed to a public repository."
        )
        uploaded = st.file_uploader("Zip of labelled photographs", type=["zip"], key="batch")
        if uploaded is not None:
            items, notes = _load_zip(uploaded.read())
            for note in notes:
                st.warning(note)
            if not items:
                st.error("No labelled images found in that zip.")
                return

    if not items:
        return

    model = load_model(model_name)
    with st.spinner(f"Scoring {len(items)} image(s)…"):
        result = _score(model, model_name, items, calibration["temperature"],
                        calibration["abstention_threshold"])

    st.markdown(f"### Results — {result['n']} images")
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{result['accuracy']:.1%}")
    b.metric("Macro F1", f"{result['macro_f1']:.3f}")
    c.metric("Macro recall", f"{result['macro_recall']:.3f}")
    d.metric("Answered (not abstained)", f"{result['coverage']:.1%}")

    if result["n"] < 100:
        st.caption(
            f"With {result['n']} images these figures are noisy — a class with 5 "
            f"examples moves in 20% steps. Report counts alongside percentages."
        )

    st.dataframe(result["per_class"].style.format(
        {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
        hide_index=True, use_container_width=True)

    st.markdown("**Confusion matrix** (rows = true, columns = predicted)")
    names = [config.CLASS_DISPLAY_NAMES[c] for c in config.CLASS_NAMES]
    st.dataframe(pd.DataFrame(result["confusion"], index=names, columns=names),
                 use_container_width=True)

    stored = config.metrics_path(model_name)
    if stored.exists() and source != "Project test split":
        reference = json.loads(stored.read_text())["metrics"]
        st.markdown("### Against the project test split")
        st.dataframe(pd.DataFrame({
            "Metric": ["Accuracy", "Macro F1", "Macro recall", "Images"],
            "Test split (source datasets)": [
                f"{reference['accuracy']:.1%}", f"{reference['macro_f1']:.3f}",
                f"{reference['macro_recall']:.3f}", reference["n_test"]],
            "These photographs": [
                f"{result['accuracy']:.1%}", f"{result['macro_f1']:.3f}",
                f"{result['macro_recall']:.3f}", result["n"]],
        }), hide_index=True, use_container_width=True)
        drop = reference["accuracy"] - result["accuracy"]
        if drop > 0.05:
            st.warning(
                f"Accuracy falls {drop:.1%} on these images. That is a finding, not a "
                f"failure: it quantifies how far performance on curated clinical "
                f"photographs carries over to independently captured ones, and belongs "
                f"in the results and limitations chapters."
            )
        else:
            st.success(
                f"Accuracy holds to within {abs(drop):.1%} of the test split — evidence "
                f"the model generalises beyond its training sources."
            )


# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Foot & Nail Condition Screening")
    st.caption("MSc dissertation prototype — transfer learning for rural healthcare screening")
    st.error(config.DISCLAIMER)

    models = available_models()
    if not models:
        st.error(
            "No trained model found. Run `python src/train.py --model mobilenetv2` "
            "(and `--model resnet50`) first, or restore them with "
            "`python src/colab_sync.py restore`."
        )
        return

    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox(
            "Architecture", models,
            index=models.index(config.PRIMARY_MODEL) if config.PRIMARY_MODEL in models else 0,
        )
        calibration = load_calibration(model_name)
        if calibration.get("_missing"):
            st.warning("Not calibrated — raw softmax, no abstention.")
        else:
            st.metric("Temperature", f"{calibration['temperature']:.3f}")
            # A threshold of zero is a real outcome, not a missing value: the
            # search found the model already met its target accuracy answering
            # every validation case, so no abstention was warranted. Rendered as
            # "0.0%" it reads like calibration failed, which invites exactly the
            # wrong question in a viva.
            if calibration["abstention_threshold"] <= 0:
                st.metric("Abstention below", "none")
                st.caption(
                    f"Temperature fitted on the validation split; it rescales "
                    f"confidence without changing any prediction. No abstention "
                    f"threshold was set: this model reached the "
                    f"{config.SELECTIVE_TARGET_ACCURACY:.0%} target accuracy "
                    f"answering every validation case, so the selection procedure "
                    f"returned none. Every prediction below is therefore reported, "
                    f"however uncertain."
                )
            else:
                st.metric("Abstention below", f"{calibration['abstention_threshold']:.1%}")
                st.caption(
                    "Both fitted on the validation split. Temperature rescales "
                    "confidence without changing any prediction."
                )
        st.markdown("---")
        st.caption(
            f"Classes: {', '.join(config.CLASS_DISPLAY_NAMES[c] for c in config.CLASS_NAMES)}"
        )

    screening, batch = st.tabs(["Screening", "Batch evaluation"])
    with screening:
        screening_tab(model_name)
    with batch:
        batch_tab(model_name)


if __name__ == "__main__":
    main()
