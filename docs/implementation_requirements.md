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
| Model interpretability; Grad-CAM named explicitly | 2.5.8 | **In scope.** Grad-CAM saliency maps are produced for test predictions, both to address the gap and to check the model attends to the lesion rather than to dataset artefacts. |
| External validation on an independent dataset | 2.5.8; 2.5.5 | Out of scope. Listed as future work. |
| Cross-validation | 2.5.5 | Out of scope; not required by the proposal. |

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
