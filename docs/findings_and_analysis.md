# Findings and analysis

Everything measured during implementation, with the reasoning behind each
decision. Written to be quoted from directly when drafting the dissertation.

**Status of the numbers below.** All MobileNetV2 results are from the final
(class × source stratified) split and are current. ResNet50 has not yet been
retrained on that split; its earlier figures are recorded separately and marked
as superseded — they must not be reported alongside the current test set,
because that model was trained on images now in it.

---

## 1. Dataset

### Sources

| Source | Provides | Class | Images |
|---|---|---|---|
| Mendeley Data `hsj38fwnvr` v3 | `Normal/` | Healthy Foot/Nail | 2,757 |
| Mendeley Data `hsj38fwnvr` v3 | `wound_main/` | Foot Wound/Injury | 2,508 |
| FUSeg / AZH Chronic Wound (Wang et al., 2020) | foot ulcer photographs | Foot Ulcer | 1,330 |
| Figshare `5398573` | onychomycosis folders (B1, B2, C, D) | Nail Fungal Infection | 779 |
| Figshare `5398573` | montage tiles, `normalnail` sheets | Healthy Foot/Nail | 1,000 |
| | | **Total** | **8,374** |

Citation for the ulcer data: C. Wang, D.M. Anisuzzaman, V. Williamson, M.K.
Dhar, B. Rostami, J. Niezgoda, S. Gopalakrishnan, Z. Yu, "Fully Automatic Wound
Segmentation with Deep Convolutional Neural Networks", *Scientific Reports*
10:21897, 2020. https://doi.org/10.1038/s41598-020-78799-w

### Preparation pipeline, with counts

| Step | Result |
|---|---|
| Files scanned under `data/raw/` | 26,544 |
| Mapped to one of the four classes | 25,689 (855 excluded) |
| Excluded: nail dystrophy | 578 — a real condition, but not one of the four classes fixed by the proposal |
| Excluded: montage sheets (A1, A2) | 277 — contact sheets, tiled separately |
| Duplicates removed | **1,758** (178 wound, 40 ulcer, 1 fungal, ~1,539 tiles) |
| Montage tiles capped | 18,096 → **1,000**, sampled evenly across all 20 sheets |
| **Final dataset** | **8,374** |
| Split | train 5,865 / val 1,261 / test 1,248 |
| Imbalance ratio | 4.8 : 1 |

Deduplication was not a formality: 1,758 images would otherwise have appeared on
both sides of the train/test boundary and inflated the reported accuracy.

### Test split composition

By class: healthy 556, foot_wound 376, foot_ulcer 199, nail_fungal 117.

By source: `mendeley_foot` 789, `ulcer_fuseg` 199, `figshare_nail_tiles` 143,
`figshare_nail` 117.

`mendeley_foot` is the only source contributing two classes (413 healthy feet,
376 wounds), which is what makes its accuracy the within-source test.

---

## 2. Results — MobileNetV2 (current split)

### Training

Two-stage transfer learning, 84.8 minutes on a Tesla T4.

| | Best val loss | Best val accuracy |
|---|---|---|
| Stage 1 (head, backbone frozen) | 0.1217 (epoch 14) | 0.9516 |
| Stage 2 (fine-tuning, top 19 layers) | **0.0604** (epoch 34 overall) | **0.9802** |

Fine-tuning improved substantially on head training, which is the evidence that
justifies the two-stage design rather than convention.

Class weights applied: healthy 0.56, nail_fungal 2.69, foot_wound 0.83,
foot_ulcer 1.57.

### Test set (1,248 images, evaluated once)

| Metric | Value |
|---|---|
| Accuracy | 0.9832 |
| Macro F1 | 0.9781 |
| Balanced accuracy | 0.9799 |
| Cohen's kappa | 0.9751 |
| Macro AUC (OvR) | 0.9994 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Healthy Foot/Nail | 0.9982 | 0.9946 | 0.9964 | 556 |
| Nail Fungal Infection | 0.9667 | 0.9915 | 0.9789 | 117 |
| Foot Wound/Injury | 0.9761 | 0.9787 | 0.9774 | 376 |
| Foot Ulcer | 0.9645 | **0.9548** | 0.9596 | 199 |

Confusion matrix (rows true, columns predicted):

```
                healthy  nail_fungal  foot_wound  foot_ulcer
healthy             553            3           0           0
nail_fungal           1          116           0           0
foot_wound            0            1         368           7
foot_ulcer            0            0           9         190
```

**Foot Ulcer recall (0.9548) is the weakest figure and the one that matters
most** — for a screening tool a missed ulcer is the costly error. Nine ulcers
were classified as wounds.

The dominant error mode is ulcer↔wound confusion (9 + 7 cases), which is
clinically coherent: both are foot lesions, and either being confused with
healthy would be far more serious.

All three healthy-class errors were **healthy nails misread as fungal
infection** — arithmetic confirms it (143 tiles × 2.1% error = 3, matching the
confusion matrix row exactly).

### Efficiency

| | |
|---|---|
| Parameters | 2,263,108 |
| Model size on disk | 21.78 MB |
| Inference, CPU | 215.0 ms/image |

CPU is the figure that speaks to phone deployment. See §5 for why the GPU
measurement points the other way.

---

## 3. Calibration and selective prediction

Modern networks are systematically overconfident (Guo et al., "On Calibration of
Modern Neural Networks", ICML 2017). For a tool aimed at a health worker with no
specialist to consult, a confidently wrong answer is worse than an admission of
uncertainty, so the trustworthiness of the number matters alongside accuracy.

### Calibration

| | Before | After |
|---|---|---|
| Mean confidence | 0.9896 | 0.9821 |
| Actual accuracy | 0.9802 | 0.9802 |
| Validation ECE | 0.0121 | **0.0070** |
| Test ECE | 0.0072 | 0.0068 |

Fitted temperature **1.4278** — above 1, so it softened confidence, as the
literature predicts. Mean confidence was ~0.9 points above actual accuracy;
afterwards within 0.2. Predictions were verified unchanged: temperature rescales
confidence without reordering classes.

Be honest in the write-up that the *test* ECE gain is marginal (0.0072 → 0.0068).
The model was only mildly miscalibrated. The finding is that calibration was
assessed, found mild, and corrected.

The temperature was fitted on the **validation** split only; test is used to
report the outcome, never to choose anything.

### Selective prediction

Abstention threshold **0.787**, chosen on validation from the risk-coverage
trade-off (target 0.99 accuracy, minimum 0.70 coverage — target met).

| | Coverage | Accuracy on answered |
|---|---|---|
| Validation | 97.2% | 0.9902 |
| **Test** | **97.0%** | **0.9959** |

The model makes roughly 21 errors on 1,248 images. Abstaining on ~37
low-confidence cases removes about 16 of them. **Refuse 3 in 100 and refer them
to a clinician, and be right 99.6% of the time on the rest.** This is the most
defensible clinical sentence available from these results.

---

## 4. Is the model reading pathology or dataset artefacts?

Each class is drawn largely from one source, so acquisition conditions differ
systematically between classes and a model could score well by recognising which
dataset an image came from. Three independent tests were run.

### 4.1 Within-source discrimination

`mendeley_foot` contributes 789 test images containing **both** Healthy (413) and
Foot Wound (376) — same clinic, same cameras, same framing.

**Accuracy: 0.9899.**

Dataset provenance offers no help on this comparison, so this is direct evidence
of genuine discrimination.

### 4.2 Grad-CAM

Attention localises to **toes and nail plates**, with image margins cold.
Qualitative, but consistent across the examples inspected, and produced for both
correct and misclassified predictions rather than successes only.

### 4.3 Border ablation (controlled occlusion)

Border brightness was found to differ systematically by class:

| Class / source | Border brightness | % near-white |
|---|---|---|
| nail_fungal / figshare_nail | 107.6 | 0.3% |
| foot_ulcer / ulcer_fuseg | 112.7 | 5.4% |
| foot_wound / mendeley_foot | 127.2 | 10.1% |
| healthy / figshare_nail_tiles | 166.7 | 32.9% |
| healthy / mendeley_foot | 170.1 | 7.0% |

Healthy images are markedly brighter at the edges, and the gap holds *within* one
source (Mendeley healthy 170.1 vs Mendeley wound 127.2). Probable cause is
framing — wound and ulcer photographs are close-ups filling the frame, healthy
feet are shot further back — but a shortcut need not be deliberate.

Results at 20px border width, on 1,248 test images:

| Arm | Accuracy | Drop |
|---|---|---|
| Unmodified | 0.9832 | — |
| Border masked | 0.9671 | 0.0160 |
| Interior control (equal area) | 0.8598 | 0.1234 |
| Border only | 0.4952 | (majority floor 0.4455) |

**Removing the border costs 1.6 points; removing an equal number of interior
pixels costs 12.3 — roughly eight times more.** Same pixel count, same
unfamiliar grey occlusion, differing only in location.

With only the border visible, per-class recall collapses to healthy 1.0000,
nail_fungal 0.0000, foot_wound 0.1516, foot_ulcer 0.0251 — a degenerate
classifier defaulting to the majority class. The +4.97 point margin over the
floor is the residue of that collapse.

**Conclusion:** the border carries a weak signal toward the healthy class,
insufficient to account for the model's performance. Do not write "the border
carries no information" — the honest claim is the weaker, defensible one.

---

## 5. The efficiency comparison points different ways on CPU and GPU

MobileNetV2's depthwise separable convolutions cut parameters and FLOPs but
underuse a GPU's dense-matrix hardware. On the earlier split, measured on a T4,
ResNet50 was *faster* (82.36 ms vs 99.69 ms) despite being ten times larger. On
CPU the ordering reverses, which is the relevant device for a phone.

Report both, and explain the reversal. Reporting only the GPU figure would
undercut the deployment argument the model size supports.

---

## 6. Limitations to state

**Healthy nail images come from ~20 photographic sessions.** Tiling produced
18,096 images from roughly 20 montage sheets, so they represent about 20 sessions
and a small number of individuals, not 18,096 independent observations. The split
groups tiles by sheet so none straddles the train/test boundary, but no grouping
creates diversity the source lacks. Contribution capped at 1,000.

**Resolution correlates with class among nail images.** Healthy nails are 102×102
tiles upscaled to 224×224; onychomycosis images are downscaled from several
hundred pixels. Randomised blur augmentation was added to prevent sharpness
acting as a class cue.

**Class is partly confounded with source.** Addressed by the three tests in §4,
but the design limitation remains and should be stated.

**The nail images are largely fingernails.** Onychomycosis in the agricultural
population this project targets is predominantly a toenail condition. Transfer to
toenails is not demonstrated by this data.

**No rural or field data.** All three sources are clinical. "For rural
healthcare" is the *motivation*; nothing here demonstrates performance in that
setting. Phrase accordingly — "motivated by", not "demonstrated for".

**OpenCV is listed in §5.5 but unused.** Pillow and NumPy cover the pipeline.
Amend the resource list or note the substitution.

---

## 7. Methodological decisions worth defending

| Decision | Reason |
|---|---|
| Two-stage transfer learning | A randomly initialised head would push large gradients through the pretrained filters and destroy the transferred features. Freeze, train the head, then fine-tune at 100× lower LR. |
| BatchNorm frozen during fine-tuning | Its statistics come from ImageNet; re-estimating them from small medical batches destabilises training and opens a train/validation gap. |
| Per-model `preprocess_input` inside the graph | MobileNetV2 expects [-1,1], ResNet50 caffe mean-subtraction. Verified numerically: pooled activations from raw input match externally preprocessed input to 2×10⁻⁵. |
| Class weights | A missed ulcer costs far more than a false alarm. |
| Aspect-preserving resize + centre crop | A stretch distorts the 153 extreme-aspect images; pad-to-square would reintroduce the black border removed from ulcers. |
| Ulcer padding cropped | FUSeg/Medetec images were 25–32% pure black; no other class carries that marking. Verified reduced to 0.000. |
| Split stratified by class **and source** | Stratifying on class alone put every montage tile in training, leaving healthy-nail performance unmeasured. |
| Test split read once | Every selection decision used validation. |

---

## 8. Superseded results (do not report with the current test set)

From the earlier class-only split. Retained because the ResNet50 training
figures indicate what to expect, and the MobileNetV2 comparison shows the split
fix cost almost nothing.

| | MobileNetV2 | ResNet50 |
|---|---|---|
| Test accuracy | 0.9833 | 0.9912 |
| Macro F1 | 0.9801 | 0.9887 |
| **Foot ulcer recall** | **0.9397** | **0.9899** |
| Parameters | 2,263,108 | 23,595,908 |
| Size | 21.78 MB | 210.55 MB |
| GPU inference | 99.69 ms | 82.36 ms |

**ResNet50 missed far fewer ulcers** — 0.9899 against 0.9397. If that holds on
the current split, the comparison is not "ResNet50 is 0.8 points more accurate"
but "ResNet50 misses substantially fewer ulcers, at ten times the size", which is
a more interesting trade-off and may not favour the small model.

Split-fix comparison for MobileNetV2: old split val 0.9817 / loss 0.0641; new
split val 0.9802 / loss 0.0604. Accuracy fell 0.15 points while loss improved —
the earlier figure was not meaningfully inflated, it simply was not measuring the
nail case at all.

---

## 9. Outstanding

1. **Retrain ResNet50 on the current split** — blocked on GPU quota. Required by
   5.4 and WBS 5.6, and it is the stated contribution in 2.5.8.
2. **Re-run `evaluate.py`, `calibrate.py` and `ablate_border.py` with both
   models** so every reported figure comes from one consistent run.
3. **External validation** — 20–50 photographs taken independently, scored
   through the prototype's batch tab. Optional but the highest-value addition
   remaining; check whether ethics approval is required first.
