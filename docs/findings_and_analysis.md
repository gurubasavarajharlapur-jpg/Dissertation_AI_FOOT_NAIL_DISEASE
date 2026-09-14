# Findings and analysis

Everything measured during implementation, with the reasoning behind each
decision. Written to be quoted from directly when drafting the dissertation.

**Status of the numbers below.** Both architectures are now trained on the
final (class × source stratified) split and evaluated together in a single run,
so every figure here comes from one consistent result set. The earlier
class-only-split figures are kept in §9, marked superseded, and must not be
mixed with these.

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

## 2. Results — both models on the current split

### Training

Identical code path, splits, hyper-parameters, augmentation and class weights
for both; only the backbone and its `preprocess_input` differ. That is what
makes the comparison a controlled experiment rather than two separate runs.

| | MobileNetV2 | ResNet50 |
|---|---|---|
| Stage 1 best val loss | 0.1217 (epoch 14) | 0.0390 (epoch 14) |
| Stage 2 best val loss | **0.0604** | **0.0272** (epoch 18 overall) |
| Best val accuracy | 0.9802 | 0.9905 |
| Backbone layers unfrozen | top 19 | top 21 of 30 |
| Trainable params at stage 2 | — | 14,436,868 of 23,595,908 |
| Training time (Tesla T4) | 84.8 min | 53.9 min |

Fine-tuning improved substantially on head training in both cases — the
evidence that justifies the two-stage design rather than convention.

ResNet50 trained *faster in wall-clock* despite being ten times larger, because
it converged in 26 epochs (early stopping at stage-2 epoch 11, best at epoch 3)
against MobileNetV2's full schedule. Worth a sentence: training cost and
inference cost do not move together, and it is inference cost that matters for
deployment.

Class weights applied to both: healthy 0.56, nail_fungal 2.69, foot_wound 0.83,
foot_ulcer 1.57.

### Test set (1,248 images, evaluated once)

| Metric | MobileNetV2 | ResNet50 |
|---|---|---|
| Accuracy | 0.9832 | **0.9896** |
| Macro F1 | 0.9781 | **0.9875** |
| Balanced accuracy | 0.9799 | **0.9896** |
| Cohen's kappa | 0.9751 | **0.9846** |
| Macro AUC (OvR) | 0.9994 | **0.9997** |
| Total errors | 21 | **13** |

Per-class recall — the figures to report, since accuracy hides the class that
matters:

| Class | MobileNetV2 | ResNet50 | Support |
|---|---|---|---|
| Healthy Foot/Nail | 0.9946 | 1.0000 | 556 |
| Nail Fungal Infection | 0.9915 | 1.0000 | 117 |
| Foot Wound/Injury | 0.9787 | 0.9734 | 376 |
| **Foot Ulcer** | **0.9548** | **0.9849** | 199 |

Confusion matrices (rows true, columns predicted):

```
MobileNetV2                                   ResNet50
                h   nf   fw   fu                            h   nf   fw   fu
healthy       553    3    0    0              healthy     556    0    0    0
nail_fungal     1  116    0    0              nail_fungal   0  117    0    0
foot_wound      0    1  368    7              foot_wound    1    1  366    8
foot_ulcer      0    0    9  190              foot_ulcer    0    0    3  196
```

**The headline clinical difference is missed ulcers: nine against three.**
For a screening tool that is the costly error, and it is a more defensible way
to state the comparison than "0.64 accuracy points".

ResNet50 is perfect on both nail classes and on healthy feet; every one of its
13 errors is a wound/ulcer confusion. The dominant error mode is the same for
both models and is clinically coherent — both are foot lesions, and either
being confused with *healthy* would be far more serious. Neither model ever
made that mistake in the ulcer or wound rows.

### Is the difference real?

+0.64 accuracy points on 1,248 images is not self-evidently a result. The two
models are scored on the *same* images and make correlated errors, so the
comparison must be made image by image — McNemar's exact test on the paired
correctness vectors, which `evaluate.py` now computes and prints, alongside
`results/metrics/<model>_predictions.csv`.

This matters more than it looks. From the error counts alone (21 vs 13) the
p-value is anywhere between 0.008 and 0.15 depending on how far the two error
sets overlap, so the difference is either clearly real or clearly unestablished
and the totals cannot tell you which. **Quote the p-value from the next
`evaluate.py` run; do not assert the difference without it.**

### Where the difference comes from

Errors by source, which is where the comparison gets interesting:

| Source | n | MobileNetV2 errors | ResNet50 errors | Change |
|---|---|---|---|---|
| figshare_nail (fungal only) | 117 | 1 | 0 | −1 |
| figshare_nail_tiles (healthy only) | 143 | 3 | 0 | −3 |
| ulcer_fuseg (ulcer only) | 199 | 9 | 3 | −6 |
| **mendeley_foot (healthy + wound)** | **789** | **8** | **10** | **+2** |
| Total | 1,248 | 21 | 13 | −8 |

Every one of ResNet50's ten gains is on a source contributing a **single**
class, where recognising the dataset is as good as recognising the pathology.
On `mendeley_foot` — the only source holding two classes, and therefore the only
comparison where provenance cannot help — ResNet50 is two images *worse*.

Two images out of 789 is noise and must be reported as such. The pattern is
still worth stating plainly, because it is the honest reading of §4 as well:
ResNet50's advantage is concentrated exactly where a provenance shortcut is
available to it.

### Efficiency

| | MobileNetV2 | ResNet50 | Ratio |
|---|---|---|---|
| Parameters | 2,263,108 | 23,595,908 | 10.4× |
| Size on disk | 21.78 MB | 210.55 MB | 9.7× |
| Inference, CPU | **253.80 ms** | 443.99 ms | 1.7× |
| Inference, GPU | **152.64 ms** | 227.79 ms | 1.5× |

Batch size 1, median of 50 runs. CPU is the figure that speaks to phone
deployment.

---

## 3. Calibration and selective prediction

Modern networks are systematically overconfident (Guo et al., "On Calibration of
Modern Neural Networks", ICML 2017). For a tool aimed at a health worker with no
specialist to consult, a confidently wrong answer is worse than an admission of
uncertainty, so the trustworthiness of the number matters alongside accuracy.

### Calibration

| | MobileNetV2 | ResNet50 |
|---|---|---|
| Fitted temperature | **1.4278** (softening) | **0.9495** (sharpening) |
| Validation ECE | 0.0121 → **0.0070** | 0.0063 → **0.0045** |
| Test ECE | 0.0072 → 0.0068 | 0.0039 → 0.0040 |
| Mean confidence | 0.9896 → 0.9821 (acc 0.9802) | 0.9903 → 0.9910 (acc 0.9905) |

The two models needed opposite corrections, which is the interesting finding
here and not one the literature would predict. MobileNetV2 behaved as Guo et al.
describe — overconfident, T > 1. **ResNet50 was mildly *under*confident** (mean
confidence 0.9903 against 0.9905 accuracy), so the fitted temperature came in
below 1 and sharpened its predictions instead.

Report that honestly rather than claiming the expected result. It is evidence
that calibration was *measured* rather than assumed, which is the methodological
point: the correction was fitted from data, and the data said something
different for each architecture.

Be equally honest that the *test* ECE gains are marginal — MobileNetV2 0.0072 →
0.0068, and ResNet50 0.0039 → **0.0040, very slightly worse**. Neither model was
badly miscalibrated to begin with. The finding is that calibration was assessed,
found mild, corrected on validation, and reported on test without adjustment.

Temperature was fitted on the **validation** split only for both models, and
verified not to reorder predictions — it rescales confidence, nothing else.

### Selective prediction

| | MobileNetV2 | ResNet50 |
|---|---|---|
| Abstention threshold | **0.787** | **0.000** |
| Validation coverage / accuracy | 97.2% / 0.9902 | 100% / 0.9905 |
| Test coverage / accuracy | **97.0% / 0.9959** | 100% / 0.9896 |

For MobileNetV2 the trade-off is the most defensible clinical sentence
available from these results: **refuse about 3 in 100 and refer them to a
clinician, and be right 99.6% of the time on the rest.** Abstaining on ~37
low-confidence cases removes about 16 of the 21 errors.

ResNet50 got no threshold at all. It already met the 0.99 validation target
answering every case, so the procedure correctly selected no abstention. But
note what follows: at 100% coverage its *test* accuracy is 0.9896 — marginally
**below** the 0.99 the threshold was chosen to guarantee. The target was met on
validation and missed by 0.04 points on test.

That is not a failure of the method, it is the method working as intended and
being honest about it: a threshold chosen on validation carries no guarantee on
unseen data, and 0.9896 against a 0.99 target on 1,248 images is well inside
sampling noise. It does mean **the selective-prediction argument is stronger for
MobileNetV2 than for ResNet50**, which is a convenient result for a dissertation
arguing the small model's case, and therefore one to state carefully rather than
lean on.

---

## 4. Is the model reading pathology or dataset artefacts?

Each class is drawn largely from one source, so acquisition conditions differ
systematically between classes and a model could score well by recognising which
dataset an image came from. Three independent tests were run, on both models.

### 4.1 Within-source discrimination

`mendeley_foot` contributes 789 test images containing **both** Healthy (413) and
Foot Wound (376) — same clinic, same cameras, same framing. Dataset provenance
offers no help on this comparison.

| | Accuracy | Errors |
|---|---|---|
| MobileNetV2 | **0.9899** | 8 |
| ResNet50 | 0.9873 | 10 |

Both are direct evidence of genuine discrimination, which is the primary claim
and it holds for both architectures.

The secondary observation — that the smaller model is marginally *better* here,
while being 0.64 points worse overall — is a 2-image difference on 789 and
nothing should be built on it alone. It is worth reporting only because §2 and
§4.3 point the same way: ResNet50's advantage lives in the single-source
classes.

### 4.2 Grad-CAM

Attention localises to **toes and nail plates**, with image margins cold.
Qualitative, but consistent across the examples inspected, and produced for both
correct and misclassified predictions rather than successes only. Figures exist
for both models (`results/figures/<model>_gradcam.png`).

### 4.3 Border ablation (controlled occlusion)

Border brightness differs systematically by class:

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

| Arm | MobileNetV2 | ResNet50 |
|---|---|---|
| Unmodified | 0.9832 | 0.9896 |
| Border masked | 0.9671 (−0.0160) | 0.9704 (−0.0192) |
| Interior control, equal area | 0.8598 (−0.1234) | 0.9030 (−0.0865) |
| Border only (centre masked) | 0.4952 | 0.5994 |
| Majority-class floor | 0.4455 | 0.4455 |

**The occlusion test passes for both.** Removing the border costs 1.6 points for
MobileNetV2 against 12.3 for an equal number of interior pixels — roughly eight
times more; for ResNet50 it is 1.9 against 8.7, about 4.5 times. Same pixel
count, same unfamiliar grey occlusion, differing only in location. Neither model
*depends* on the border.

**The border-only test separates them, and this is the finding.** With the
interior masked and only a 20px frame visible:

| Recall, border only | MobileNetV2 | ResNet50 |
|---|---|---|
| healthy | 1.0000 (556/556) | 0.9982 (555/556) |
| nail_fungal | 0.0000 (0/117) | 0.0000 (0/117) |
| foot_wound | 0.1516 (57/376) | **0.3511 (132/376)** |
| foot_ulcer | 0.0251 (5/199) | **0.3065 (61/199)** |
| Accuracy vs floor | 0.4952 vs 0.4455 (+0.0497) | 0.5994 vs 0.4455 (+0.1538) |

MobileNetV2 collapses to a degenerate majority-class classifier: 62 non-healthy
images correct out of 692, which is the residue of that collapse rather than a
signal. **ResNet50 recovers 193 — three times as many — from the border alone**,
and the script flags it: *"the border alone is predictive of the class."*

The honest conclusion, which is stronger than a clean pass would have been:

> The border carries a real signal toward class, weak but not absent. The larger
> model extracts substantially more of it. Neither model's decision *depends* on
> the border — the equal-area occlusion control establishes that for both — but
> the capacity to absorb dataset artefact scales with model capacity, and
> ResNet50 demonstrably absorbed more of it.

This lines up with §2's error breakdown and §4.1: ResNet50's accuracy advantage
is concentrated in single-source classes, and it reads more provenance from the
frame. **None of that makes its lower ulcer miss rate false** — three missed
ulcers against nine is the most important number in this project, and no test
here contradicts it. What it means is that the +0.64 points should not be
presented as a pure pathology-discrimination gain, and the viva answer to "why
recommend the smaller model?" now has three strands rather than one: comparable
accuracy where confound cannot help, less artefact absorption, and a tenth of
the size at 1.7× the speed on the deployment device.

---

## 5. Efficiency: the CPU/GPU ordering is a measurement, not a rule

MobileNetV2's depthwise separable convolutions cut parameters and FLOPs but
underuse a GPU's dense-matrix hardware, so the small model's advantage is
expected to be largest on CPU and may disappear on GPU.

**On the earlier split that reversal actually happened** — ResNet50 was faster
on a T4 (82.36 ms vs 99.69 ms) despite being ten times the size. **On the
current run it did not**: MobileNetV2 is faster on both devices (CPU 253.80 vs
443.99 ms, GPU 152.64 vs 227.79 ms), with its margin narrower on GPU (1.5×)
than on CPU (1.7×), which is the expected effect in weaker form.

Same code, same hardware class, opposite orderings between runs. Colab T4s are
shared and contended, and these are batch-size-1 latencies where scheduling
noise is large relative to the measurement, so treat the GPU figures as
indicative rather than precise.

Two things follow for the write-up. Report the CPU figure as the deployment
number, since it is the device a phone resembles and it ranks the models the
same way in both runs. And do not present the GPU reversal as a law — it is a
plausible architectural effect that appeared in one run and not the other, which
is exactly how it should be described. `evaluate.py` now branches on the
measured numbers rather than asserting the reversal, after it printed the
"ordering differs" claim on a run whose own table showed otherwise.

---

## 6. Which model to recommend, and how to phrase it

The proposal frames this as MobileNetV2 (deployment candidate) against ResNet50
(accuracy reference). The results support a recommendation, with the trade-off
stated rather than hidden:

**ResNet50 is the more accurate model** — 0.9896 against 0.9832, and more
importantly three missed ulcers against nine. If a missed ulcer is the costly
error, and it is, that difference points at ResNet50 on clinical grounds alone.

**MobileNetV2 is the deployable model** — a tenth of the parameters, 21.78 MB
against 210.55 MB, 1.7× faster on CPU, and it carries a working abstention
threshold where ResNet50's procedure returned none.

**The honest summary**: on the one comparison free of provenance confound they
are equivalent (0.9899 vs 0.9873, a 2-image difference), and ResNet50's measured
advantage sits entirely in classes where a dataset shortcut is available to it —
a shortcut §4.3 shows it picks up three times more of. That does not refute its
ulcer result, but it does mean the accuracy gap is not cleanly attributable to
better pathology discrimination.

The defensible recommendation is therefore **MobileNetV2 for phone-based
screening, with abstention and referral**, noting that ResNet50 remains the
better choice wherever a server is available and the ulcer miss rate dominates
every other consideration. Do not write "MobileNetV2 is as accurate as
ResNet50" — write that the accuracy gap is small, partly confound-assisted, and
bought at ten times the size.

The genuinely missing evidence is external validation: nothing here shows which
model degrades more gracefully on a phone photograph from outside all three
datasets, and that is the question the recommendation actually turns on.

---

## 7. Limitations to state

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

**One split, one seed, one run per architecture.** Every comparison here rests
on a single train/val/test partition and a single training run of each model
at `RANDOM_SEED = 42`. Several differences being discussed are 2–8 images wide,
which is smaller than the variation repeated runs would show. The paired
McNemar test (§2) addresses this for the head-to-head accuracy comparison, but
nothing here bounds seed-to-seed variance. Say "on this split" rather than
implying the ordering is stable; k-fold or repeated seeds would fix it and were
not affordable in the available GPU time.

**The abstention threshold is model-specific and was not transferable.**
MobileNetV2 got 0.787, ResNet50 got 0.000 because it met the target answering
everything on validation. A threshold chosen this way carries no guarantee on
unseen data, and in fact ResNet50's test accuracy at full coverage (0.9896) came
in just under the 0.99 it was selected to achieve.

**Model capacity and artefact absorption move together.** §4.3 shows ResNet50
recovering three times as much class information from a 20px border as
MobileNetV2. Choosing the larger model for accuracy also buys more sensitivity
to whatever is incidental in the data — relevant to any future work scaling up
the backbone.

---

## 8. Methodological decisions worth defending

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
| Equal-area interior control in the ablation | Masking anything removes information, so a small drop from border-masking proves nothing on its own. The control is what makes the arm interpretable, and a centre-masked arm catches the floor case where both arms collapse to chance. |
| McNemar's exact test, not two independent accuracies | Both models are scored on the same 1,248 images and make correlated errors. Comparing the accuracies as independent samples is the wrong test; the exact form is used because the discordant count is ~20, where the chi-square approximation is weakest. |
| Temperature fitted on validation, reported on test | The correction must not see the data it is judged on. It was also verified not to reorder predictions. |

---

## 9. Superseded results (do not report with the current test set)

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

**ResNet50 missed far fewer ulcers** — 0.9899 against 0.9397. That prediction
was made before the current split was trained, and it **held**: 0.9849 against
0.9548, three missed ulcers against nine (§2). The comparison is therefore not
"ResNet50 is 0.6 points more accurate" but "ResNet50 misses substantially fewer
ulcers, at ten times the size" — and §2 and §4.3 then complicate even that, by
locating the advantage in the classes where provenance can substitute for
pathology.

Note also that the ulcer recall gap *narrowed* between the splits — 0.9397 →
0.9548 for MobileNetV2 against 0.9899 → 0.9849 for ResNet50 — which is the
opposite of what a superficial reading of the old numbers would suggest and
another reason to quote only the current run.

Split-fix comparison for MobileNetV2: old split val 0.9817 / loss 0.0641; new
split val 0.9802 / loss 0.0604. Accuracy fell 0.15 points while loss improved —
the earlier figure was not meaningfully inflated, it simply was not measuring the
nail case at all.

---

## 10. Outstanding

Done: ResNet50 is trained on the current split (5.4, WBS 5.6), and
`evaluate.py`, `calibrate.py` and `ablate_border.py` have all been re-run with
both models, so every figure above comes from one consistent result set.

1. **Re-run `evaluate.py` once more for the paired significance test.** The
   McNemar test and the per-image prediction CSVs were added after the current
   results were produced, so the p-value on the +0.64 point gap is not yet
   measured. This is a few minutes of inference, no retraining, and it is the
   first thing a supervisor will ask about the comparison.
2. **External validation** — 20–50 photographs taken independently, scored
   through the prototype's batch tab. The highest-value addition remaining, and
   the only evidence that would settle which model to recommend (§6); check
   whether ethics approval is required first.
3. **Optional, if GPU time allows: a second seed for each architecture.** Would
   bound the seed-to-seed variance the single-run limitation in §7 concedes.
   Not required by the proposal.

---

## 11. Reproducibility notes

Failure modes that cost real time and are worth a paragraph in the methodology
or reflection, because each produced a *silent* wrong result rather than an
error.

| Symptom | Cause | Fix |
|---|---|---|
| Three Figshare zips downloaded as 0 bytes, reporting success | A JSON `Accept` header, then a browser `User-Agent`, on the download request. The default `requests` UA returned the correct 1.8 GB payload. | Send no custom headers on downloads; verify byte size and zip readability before treating a fetch as done. The notebook now also carries a `wget -c` fallback by file ID. |
| Mendeley zips reported missing from a Drive that had them | Colab's Drive FUSE mount is case-sensitive; the export ships `Normal.zip`, the code looked for `normal.zip`. | Resolve targets against the lower-cased names actually present. |
| An entire session's outputs lost | Drive FUSE buffers writes; the runtime was recycled before they flushed. | `src/colab_sync.py` writes single tar archives and verifies them, rather than syncing directories. |
| Test set contained zero healthy nails, unnoticed | Splitting stratified on class alone sent all 20 montage-tile groups to training. | Stratify on class × source; `preprocessing.py --verify` now fails if any source is absent from a split. |
| Best fine-tuning weights overwritten by worse ones | Stage 2's `ModelCheckpoint` started from no baseline, so its first epoch was "best" even when stage 1 had done better. | Pass stage 1's best `val_loss` as `initial_value_threshold`. |
| Border ablation "proved" robustness on data where it could not | Masking anything removes information, so a small drop means nothing on its own; and both arms hit chance on a border-only fixture. | Score an equal-area interior control and a centre-masked arm alongside, and detect the floor case explicitly. |

The pattern is consistent: every one of these passed silently. Each fix is
therefore a *verification* rather than a correction — a size check, a split
check, a control arm — which is the point worth making in the write-up.
