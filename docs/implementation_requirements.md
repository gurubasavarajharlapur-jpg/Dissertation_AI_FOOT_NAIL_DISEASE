# Implementation requirements traced to the proposal and literature review

Source: `docs/Proposal_and_Literature_Review.docx` (Gurubasavaraj Harlapur,
M01093116, supervisor Dr Krishnadas Nanath, submitted 15/06/2026).

This file exists so that every implementation choice can be traced back to a
specific section of the proposal, and so that anything *not* traceable is
visible as such. Where the code and this table disagree, this table wins.

## Fixed by the proposal — not open to redesign

| Requirement | Source section | Status |
|---|---|---|
| Exactly four classes: Healthy Foot/Nail, Nail Fungal Infection, Foot Wound/Injury, Foot Ulcer | Abstract; Methods; 5.4 | `config.CLASS_NAMES` |
| MobileNetV2 as the **primary** model, chosen for computational efficiency | Methods; 2.5.5 | `config.PRIMARY_MODEL` |
| ResNet50 as the **comparative** model | Methods; WBS 4 | `config.COMPARISON_MODEL` |
| Transfer learning from ImageNet pre-training, not training from scratch | Methods; 2.5.5 | Phase 3-4 |
| Python, TensorFlow, Keras, OpenCV, Jupyter | 5.5 Resources Needed | `requirements.txt` |
| Publicly available datasets only | Methods; 5.6; 5.7 | `src/download_data.py` |
| Preprocessing: resizing, normalisation, cleaning, augmentation | Methods; WBS 3.4-3.7 | Phase 2 |
| Train / validation / test subsets | Methods; WBS 3 | Phase 2 |
| Metrics: accuracy, precision, recall, F1-score, confusion matrix | Methods; 5.4; WBS 5.1-5.5 | Phase 5 |
| Direct comparison of MobileNetV2 vs ResNet50 | 5.4; WBS 5.6 | Phase 5 |
| Prototype: upload an image, receive a predicted classification | Methods; WBS 6 | Phase 6 |
| Positioned as an assistive screening tool, never a diagnostic replacement | Abstract; 5.7 Ethical Aspects | Prototype disclaimer |

## Required by the stated contribution

Section 2.5.8 closes by stating this dissertation investigates multiple
architectures "using a consistent experimental framework" and that "in addition
to comparing classification performance, the study considers computational
efficiency and deployment suitability". Both are therefore deliverables, not
optional extras:

| Requirement | Source | Status |
|---|---|---|
| Both architectures trained under identical conditions — same splits, same hyper-parameters, same code path | 2.5.8 (gap: inconsistent comparative evaluation) | Shared `config.py`; one training path |
| Report model size, parameter count and inference time alongside accuracy | 2.5.8 (deployment suitability) | Phase 5 |

## Identified as research gaps — decision needed

Section 2.5.8 names these as limitations of existing work. The proposal does not
commit to addressing them, so they are flagged rather than assumed:

| Gap | Source | Decision |
|---|---|---|
| Model interpretability; Grad-CAM named explicitly | 2.5.8 | **Implemented.** Grad-CAM overlays for correct and misclassified test predictions (`src/evaluate.py`), and on every prediction in the prototype. |
| External validation on an independent dataset | 2.5.8; 2.5.5 | **Supported.** The prototype's batch tab scores independently captured photographs against the test split. Collecting them is the author's step. |
| Cross-validation | 2.5.5 | Out of scope; not required by the proposal. |

## Additions beyond the proposal

The proposal was submitted and graded before these were added, and the author
approved them explicitly. They are recorded here so that nothing in the
implementation is untraceable.

**Confidence calibration** (`src/calibrate.py`). The proposal does not mention
calibration; the word appears nowhere in it. It was added because the softmax
outputs a screening tool presents to a health worker are not probabilities in
any useful sense unless checked — modern networks are systematically
overconfident (Guo et al., "On Calibration of Modern Neural Networks", ICML
2017). Expected Calibration Error, a reliability diagram and temperature
scaling fitted on the validation split. Measured on MobileNetV2: mean
confidence 98.96% against 98.02% accuracy, corrected to 98.21% at temperature
1.428; validation ECE 0.0121 -> 0.0070.

**Selective prediction** (abstention threshold). Below a confidence threshold
the prototype declines to name a condition and recommends consulting a
clinician. The threshold is chosen on validation from the risk-coverage
trade-off, never on test. Measured on MobileNetV2: answering 97.0% of test
cases raises accuracy on those answered from 98.32% to 99.59%. This serves the
proposal's own positioning of the system as assistive screening rather than
diagnosis (5.7), and 2.5.8's criticism of work that optimises accuracy without
regard to clinical usability.

## Known deviation

**OpenCV is listed under 5.5 Resources Needed but is not used.** Image loading,
resizing and cleaning are done with Pillow and NumPy, which cover everything the
pipeline needs. OpenCV remains in `requirements.txt` but no module imports it.
Adding a contrived use to match the resource list would be worse than recording
the discrepancy; the write-up should either amend 5.5 or note that Pillow was
used in its place.

## Dataset sources

The proposal specifies "diabetic foot ulcer image datasets and nail disease
image datasets available through research publications and Kaggle repositories"
(Methods). The sources in use:

| Class | Source | Images |
|---|---|---|
| Healthy Foot/Nail | Mendeley `hsj38fwnvr` v3, `Normal/` | 2,757 |
| Foot Wound/Injury | Mendeley `hsj38fwnvr` v3, `wound_main/` | 2,686 |
| Foot Ulcer | FUSeg / AZH Chronic Wound (Wang et al., 2020) | 1,370 |
| Nail Fungal Infection | Figshare `5398573`, onychomycosis folders | 780 |

Excluded, with reasons:

- **Nail dystrophy (578 images, Figshare B1-D).** Not one of the four classes
  fixed by the proposal. Retained on disk, unused.
- **Montage thumbnail sheets (277 images, Figshare A1/A2).** Each file is a grid
  of several hundred tiny nail images tiled into one picture, so each would
  enter training as a single mislabelled example. Tiled into individual images
  by `src/extract_montages.py` before use; the sheets themselves never enter
  training.
- **Segmentation masks (Mendeley `wound_mask`, FUSeg `labels/`).** Ground-truth
  masks for a segmentation task, not photographs. Held outside `data/raw/`.

## Open issue: healthy nail images

"Healthy Foot/Nail" is a single class covering both. The healthy images
currently available are whole feet from Mendeley; the Figshare per-image folders
contain only diseased nails (onychomycosis and dystrophy). The model would
therefore never see a healthy nail close-up during training, and a user
photographing a healthy nail would most likely have it classified as fungal
infection — a false positive in exactly the use case the proposal describes.

Healthy nail images do exist in the A1 montage sheets, whose filenames encode
the class (`normalnail`, `naildystrophy`, `-focus`). **Resolved:**
`src/extract_montages.py` tiles the `normalnail` sheets into individual healthy
nail images, which join the Healthy Foot/Nail class alongside the Mendeley
whole-foot photographs.

## Deliverable status

| WBS | Deliverable | Status |
|---|---|---|
| 3 | Dataset preparation | Complete — 8,374 images, reproducible split |
| 4.1-4.7 | MobileNetV2 training | Complete |
| 4.1-4.7 | ResNet50 training | **Outstanding** — must be retrained on the current split |
| 5.1-5.5 | Metrics, confusion matrix | Complete for MobileNetV2 |
| 5.6 | Comparative evaluation | **Outstanding** — requires both models |
| 6.1-6.5 | Prototype | Complete |
| — | Calibration and abstention | Complete for MobileNetV2 |

## Dataset limitations to state in the dissertation

These are properties of the source data, not of the implementation. Each should
appear in the limitations section and be ready to defend at viva.

**Healthy nail images come from ~20 photographic sessions.** Tiling the montage
sheets produced 18,096 images, but they were cut from roughly 20 sheets. They
therefore represent about 20 sessions, and a small number of individuals, rather
than 18,096 independent observations. The split groups tiles by sheet so none
straddles the train/test boundary, but no grouping can create diversity the
source does not contain. The contribution is capped at 1,000 images
(`config.MAX_IMAGES_PER_SOURCE`), sampled evenly across every sheet.

**Uncapped, the Healthy class would dominate.** 20,853 of 25,689 images, or 81%
— a model predicting "healthy" every time would score 81% accuracy — and 87% of
that class would be nails, drowning the healthy-foot signal. After capping the
imbalance is roughly 4.8:1, handled with class weighting.

**Resolution correlates with class among the nail images.** Healthy nails are
102x102 tiles upscaled to 224x224; onychomycosis images are several hundred
pixels, downscaled. Sharpness is therefore a cue the model could exploit instead
of pathology. Randomised blur augmentation
(`config.AUGMENTATION["blur_factor"]`) removes it as a reliable signal, and
Grad-CAM is used to check where the model actually attends.

**Class is partly confounded with source.** Each class is drawn predominantly
from one dataset, so acquisition conditions differ systematically between
classes. The informative comparisons are those *within* a source — healthy foot
vs wound (both Mendeley), and healthy nail vs onychomycosis (both Figshare) —
and these appear as blocks in the confusion matrix.

**The nail images are largely fingernails.** The proposal's class is "Nail
Fungal Infection" rather than toenail specifically, but onychomycosis in the
agricultural population this project targets is predominantly a toenail
condition. Transfer to toenails is not demonstrated by this data.
