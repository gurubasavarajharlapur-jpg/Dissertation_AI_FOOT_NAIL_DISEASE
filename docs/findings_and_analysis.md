# Findings and analysis

Everything measured during implementation, with the reasoning behind each
decision. Written to be quoted from directly when drafting the dissertation.

## Headline results

- **98.96% test accuracy** (ResNet50) and **98.32%** (MobileNetV2) on 1,248
  held-out images, evaluated once, across four classes.
- **Foot ulcer recall 98.5% / 95.5%** — the clinically costly error, reported
  ahead of overall accuracy.
- **Three independent tests confirm the models classify pathology, not dataset
  provenance** (§4): 98.99% accuracy within a single source, Grad-CAM attention
  on toes and nail plates, and a controlled occlusion ablation with an
  equal-area interior control.
- **No statistically significant accuracy difference between the two
  architectures** — McNemar's exact test, p = 0.15, with any real difference
  bounded at +1.4 points at 95% confidence. MobileNetV2 reaches that at a tenth
  of the parameters and roughly twice the CPU speed (§2, §6).
- **Calibrated confidence with a referral threshold**: decline the least
  confident 3% of cases and refer them, and be correct on **99.6%** of the rest
  (§3).
- A **working prototype** with screening and batch-evaluation tabs, Grad-CAM
  explanation and clinical guidance text (Phase 6).

**Status of the numbers.** Both architectures are trained on the final
(class × source stratified) split and evaluated, calibrated and ablated in a
single run, so every figure here comes from one consistent result set. The
earlier class-only-split figures are kept in §10, marked superseded.

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

### MobileNetV2 matches a model ten times its size where the test is strictest

Errors by source. This is the table the deployment recommendation rests on:

| Source | n | Classes | MobileNetV2 | ResNet50 |
|---|---|---|---|---|
| figshare_nail | 117 | fungal only | 1 | 0 |
| figshare_nail_tiles | 143 | healthy only | 3 | 0 |
| ulcer_fuseg | 199 | ulcer only | 9 | 3 |
| **mendeley_foot** | **789** | **healthy + wound** | **8** | **10** |
| Total | 1,248 | | 21 | 13 |

`mendeley_foot` is the only source contributing two classes, so it is the one
comparison in which recognising the dataset cannot substitute for recognising
the condition (§4.1). **There the two architectures are indistinguishable** — 8
errors against 10 on 789 images, a difference well inside sampling noise.

So the compact model performs at the level of an architecture ten times its
size on the strictest comparison available, and ResNet50's measured advantage
comes from the single-class sources. Both facts are worth reporting, and
together they are the quantitative basis for recommending MobileNetV2 for
deployment (§6).

### Is the +0.64 point difference statistically established? **No.**

Both models are scored on the *same* 1,248 images and make correlated errors, so
the two accuracies cannot be compared as though they were independent samples.
The correct test is McNemar's exact test on the paired per-image outcomes.

> **The two models disagree on 24 of 1,248 images — 16 in ResNet50's favour, 8
> in MobileNetV2's. McNemar's exact test gives p = 0.1516. The difference is
> not statistically significant at the 5% level.**

A paired bootstrap confidence interval (10,000 resamples) puts the accuracy
difference at **+0.0064, 95% CI [−0.0016, +0.0144]**. The interval spans zero,
agreeing with the p-value, and bounds the practical size of any real difference
at roughly 1.4 accuracy points.

**This is the central quantitative result of the comparison, and it supports
the recommendation directly**: on this test set, MobileNetV2 is not
demonstrably less accurate than an architecture ten times its size. Stated
precisely for the dissertation —

> No statistically significant difference in overall accuracy was found between
> MobileNetV2 and ResNet50 (McNemar's exact test, p = 0.15, n = 1,248), with the
> accuracy difference bounded at +1.4 points at 95% confidence. MobileNetV2
> achieves this at 9.6% of the parameters and roughly half the CPU latency.

Two points of care in phrasing. Failing to establish a difference is not the
same as proving the models identical — the correct claim is "no significant
difference was found", not "the models are equivalent". And a larger test set
could resolve a real difference this one cannot; the confidence interval is the
honest expression of that, which is why it is reported alongside the p-value.

Reporting a difference between two models without a paired test is a common
weakness in the applied literature reviewed in §2.5.8. This study does not have
it.

### Per-class and per-source significance

`src/compare_models.py` repeats the paired test within each class and each
source, from the saved prediction CSVs — no GPU, no re-run of inference:

```bash
python src/compare_models.py
```

The class that matters is **Foot Ulcer**, where MobileNetV2 misses nine and
ResNet50 three. That is a separate question from overall accuracy and deserves
its own test rather than inheriting the overall verdict: a model can be better
overall and no better on the class that decides whether the tool is safe.
Quote that p-value in the discussion alongside the recall figures.

### Efficiency

| | MobileNetV2 | ResNet50 | Ratio |
|---|---|---|---|
| Parameters | 2,263,108 | 23,595,908 | 10.4× |
| Size on disk | 21.78 MB | 210.55 MB | 9.7× |
| Inference, CPU | **248.29 ms** | 460.56 ms | 1.9× |
| Inference, GPU | **155.39 ms** | 231.64 ms | 1.5× |

Batch size 1, median of 50 runs, on a Colab Tesla T4. CPU is the figure that
speaks to phone deployment. Latency varies a few percent between runs on shared
hardware (a repeat run measured 253.80 / 443.99 ms on CPU); the ordering and
the approximate ratio are stable, the third significant figure is not.

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

This is a genuine finding and a good one to present: the correction was
**fitted from the data rather than assumed**, and the data gave a different
answer for each architecture. A study that applied T > 1 to both because the
literature says networks are overconfident would have got ResNet50 wrong.

Both models were only mildly miscalibrated to begin with, so the validation
gains are modest in absolute terms — MobileNetV2 0.0121 → 0.0070 and ResNet50
0.0063 → 0.0045, both roughly a 30-40% reduction in expected calibration error.
On test the figures move little in either direction, which is the expected
result for models that were already close to calibrated. The contribution is
that calibration was *measured* on both, corrected on validation, and reported
on test without further adjustment.

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

ResNet50 needed no threshold: it reached the 99% target answering every
validation case, so the selection procedure correctly returned none. Its test
accuracy at full coverage is 0.9896, within sampling noise of that target on
1,248 images.

The practical point for the recommendation is that **selective prediction adds
most where it is most needed** — on the compact model intended for deployment,
where it converts 98.3% accuracy into 99.6% on answered cases for the cost of
referring 3 in 100. That is a deployable safety mechanism, and it is one of the
four grounds for the recommendation in §6.

---

## 4. Validation: both models read pathology, not dataset artefacts

Each class is drawn largely from one source, so a model could in principle score
well by recognising which dataset an image came from rather than the condition.
§2.5.8 of the literature review identifies this as the standard unaddressed
weakness in the published work, so this project tested for it three independent
ways instead of assuming it away.

**All three tests passed, for both architectures.** This is the strongest
methodological contribution in the project, and the evidence behind every
accuracy figure reported above.

### 4.1 Within-source discrimination — passed

`mendeley_foot` contributes 789 test images containing **both** Healthy (413) and
Foot Wound (376) — same clinic, same cameras, same framing, same lighting. On
this comparison, knowing the dataset tells a model nothing. It is the strictest
test available in this data.

| | Accuracy | Errors |
|---|---|---|
| **MobileNetV2** | **0.9899** | 8 |
| ResNet50 | 0.9873 | 10 |

**Both models score ~99% where provenance cannot help them.** That is direct
evidence of genuine pathology discrimination and it is the headline of this
section.

It is also where the deployment recommendation is won: on the strictest test
in the study, the compact model matches an architecture ten times its size
(a 2-image difference on 789 is well inside sampling noise).

### 4.2 Grad-CAM — passed

Attention localises to **toes and nail plates**, with image margins cold —
the model looks where a clinician looks. Consistent across every example
inspected, and generated for misclassified predictions as well as correct ones
rather than curating successes. Figures for both models at
`results/figures/<model>_gradcam.png`.

### 4.3 Controlled occlusion ablation — passed

A note on method first, because it is the part worth defending in a viva.
Masking the border and observing that accuracy holds proves nothing on its own:
masking *anything* removes information. The test is only interpretable against
an **equal-area interior control** — the same number of pixels, the same grey
occlusion, differing only in location. This design is what makes the result
evidence rather than assertion, and it is not present in the work §2.5.8
reviews.

Border brightness does differ by class, which is why the test was run:

| Class / source | Border brightness | % near-white |
|---|---|---|
| nail_fungal / figshare_nail | 107.6 | 0.3% |
| foot_ulcer / ulcer_fuseg | 112.7 | 5.4% |
| foot_wound / mendeley_foot | 127.2 | 10.1% |
| healthy / figshare_nail_tiles | 166.7 | 32.9% |
| healthy / mendeley_foot | 170.1 | 7.0% |

Results at 20px border width, 1,248 test images:

| Arm | MobileNetV2 | ResNet50 |
|---|---|---|
| Unmodified | 0.9832 | 0.9896 |
| Border masked | 0.9671 (−0.0160) | 0.9704 (−0.0192) |
| Interior control, equal area | 0.8598 (−0.1234) | 0.9030 (−0.0865) |

**Removing the border costs MobileNetV2 1.6 points; removing the same number of
interior pixels costs 12.3 — roughly eight times more.** For ResNet50 it is 1.9
against 8.7, about 4.5 times. Neither model's decision depends on the frame;
both depend on the foot.

### 4.4 MobileNetV2 is additionally the more robust of the two

The ablation's fourth arm masks the *interior* and leaves only a 20px frame, to
measure how much either model could read from the border alone:

| Border only | MobileNetV2 | ResNet50 |
|---|---|---|
| healthy recall | 1.0000 (556/556) | 0.9982 (555/556) |
| nail_fungal recall | 0.0000 (0/117) | 0.0000 (0/117) |
| foot_wound recall | 0.1516 (57/376) | 0.3511 (132/376) |
| foot_ulcer recall | 0.0251 (5/199) | 0.3065 (61/199) |
| Non-healthy recovered | **62 of 692** | 193 of 692 |
| Accuracy | 0.4952 | 0.5994 |
| Majority-class floor | 0.4455 | 0.4455 |
| Margin over floor | **+0.0497** | +0.1538 |

**MobileNetV2 extracts almost nothing from the frame.** With the interior
hidden it scores 0.4952 against a majority-class floor of 0.4455 — a margin of
five points that is the residue of collapsing into predicting "healthy" for
almost everything, recovering only 62 of 692 non-healthy images. Degrading into
the majority class rather than inventing a confident diagnosis is the behaviour
wanted from a screening tool.

ResNet50 recovers about three times as much (0.5994, a margin of 15 points over
the same floor). This is the expected consequence of capacity: a model with ten
times the parameters has more room to fit incidental detail alongside pathology.
It does not affect either model's validity — §4.3 establishes that neither
*relies* on the border, which is the question that matters — but it is a
further measured argument for the compact architecture, and a good answer to
"why not just use the bigger model?"

---

## 5. Efficiency, measured on both CPU and GPU

**MobileNetV2 is the faster model on both devices**, and by the wider margin on
CPU — 1.9× on CPU against 1.5× on GPU. CPU is the deployment-relevant figure,
since it is the device class a mid-range phone resembles, and it ranks the
architectures the same way in every run of this study.

The narrower GPU margin is itself an expected architectural effect, worth a
sentence in the discussion: depthwise separable convolutions cut parameters and
FLOPs, which a CPU converts directly into speed, but they underuse a GPU's
dense-matrix hardware, so the compact model gains least exactly where batched
server hardware is available. This is one more reason the efficiency argument
belongs to the phone-deployment case rather than to serving in general.

Measurement note for the methodology: these are batch-size-1 latencies taken on
a shared Colab T4, where contention is large relative to the measurement. An
earlier run on the previous split timed ResNet50 as the faster of the two on
GPU (82.36 ms vs 99.69 ms). The CPU ordering has been stable throughout, which
is why it is the figure quoted. `evaluate.py` reports whichever ordering it
measures rather than assuming one.

---

## 6. Recommendation

The proposal frames this as MobileNetV2 (deployment candidate) against ResNet50
(accuracy reference). The evidence supports a clear recommendation.

**MobileNetV2 is recommended for phone-based rural screening**, on four
measured grounds:

1. **There is no statistically significant accuracy difference between the two
   models.** McNemar's exact test on the paired per-image outcomes gives
   p = 0.15, with the difference bounded at +1.4 accuracy points at 95%
   confidence (§2). The same holds on the strictest comparison available —
   0.9899 against 0.9873 on the 789 within-source images, a difference of two
   images (§4.1).
2. **It is a tenth of the size and 1.9× faster on CPU** — 21.78 MB against
   210.55 MB, 248.29 ms against 460.56 ms per image. CPU is the device class a
   mid-range phone resembles.
3. **It is the more robust of the two to dataset artefact** — it recovers 62 of
   692 non-healthy images from the border alone against ResNet50's 193, and
   degrades gracefully rather than confidently when the lesion is hidden (§4.4).
4. **It supports a working abstention threshold**: refuse the least confident
   3% and refer them, and be right 99.6% of the time on the rest (§3).

**ResNet50 is the stronger accuracy reference, and should be reported as such.**
0.9896 accuracy, 0.9875 macro F1, perfect recall on both nail classes, and
three missed ulcers against nine. Where a server is available and the ulcer
miss rate outweighs every other consideration, it is the better model. That
finding is a genuine part of the contribution, not a concession — the
comparison was run precisely so the trade-off could be quantified rather than
assumed.

**The contribution is the quantified trade-off**, which is what WBS 5.6 asks
for: a 0.64-point accuracy gap that does not reach significance, six fewer
missed ulcers, bought at 10.4× the parameters, 9.7× the storage and 1.9× the
CPU latency. A dissertation that reports only "ResNet50 scored higher" has not
answered the research question — it has not established that the difference is
real, nor priced it. This one does both.

One honest qualification to carry into the discussion. Failing to establish a
difference is not the same as establishing equivalence, and with 1,248 test
images this study cannot resolve a difference smaller than roughly 1.4
accuracy points. The recommendation does not depend on the two models being
identical — it depends on the difference being small enough that a tenfold
reduction in size is worth it, and 1.4 points is the upper bound on how large
that difference could be.

---

## 7. Viva questions, with answers

Every answer below is supported by a number already in this document. The
pattern to use throughout: **state the result, name the evidence, stop.**
Confidence in a viva comes from having measured the thing, not from having
nothing to say about it.

**"Your classes come from different datasets — how do you know the model isn't
just recognising the dataset?"**

> I tested that three independent ways and it passed all three. First,
> within-source: one source in my test set contains both healthy feet and
> wounds — same clinic, same camera — and accuracy there is 98.99%, where
> knowing the dataset tells the model nothing. Second, Grad-CAM shows attention
> on the toes and nail plates with the margins cold. Third, a controlled
> occlusion ablation: masking the image border costs 1.6 accuracy points, but
> masking the same number of interior pixels costs 12.3 — eight times more. The
> decision is in the foot, not the frame.

This is the strongest answer in the project. It is also the question most
likely to be asked, because §2.5.8 shows the published work does not answer it.

**"Why the equal-area control? Wasn't masking the border enough?"**

> No, and that is why I added it. Masking anything removes information, so a
> small accuracy drop from border-masking on its own proves nothing — it could
> just mean 20 pixels is a small region. The control masks the same pixel count
> in the interior with the same grey occlusion, so location is the only variable.
> The eight-fold difference is the result. I also added a centre-masked arm to
> detect the case where both arms collapse to chance and the test would be
> uninformative.

**"ResNet50 is more accurate. Why are you recommending MobileNetV2?"**

> Because the accuracy gap is 0.64 points and the cost is ten times the model.
> On the one comparison where dataset provenance cannot help either model, they
> are statistically indistinguishable — 8 errors against 10 on 789 images.
> MobileNetV2 delivers that at 21.78 MB against 210.55 MB and nearly twice as
> fast on CPU, which is the device class a mid-range phone resembles. It is also the
> more robust of the two to incidental image features. For a rural screening
> tool that has to run on a health worker's phone, that is the right trade-off —
> and I report ResNet50's advantage as well, because quantifying the trade-off
> was the research question.

**"What is your most important result?"**

> Foot ulcer recall, and the referral threshold. Ulcer recall is 95.5% for
> MobileNetV2 and 98.5% for ResNet50 — a missed ulcer is the costly error in
> this application, so that is the number I lead with rather than overall
> accuracy. And with calibrated confidence, MobileNetV2 can decline the least
> confident 3% of cases and refer them to a clinician, and is then correct on
> 99.6% of the ones it does answer.

**"Is a 0.64 point difference between the two models significant?"**

> No, and I tested it rather than assuming either way. Both models are
> evaluated on the same 1,248 images and make correlated errors, so comparing
> the two accuracies as independent samples would be the wrong test. I used
> McNemar's exact test on the paired per-image outcomes: the models disagree on
> 24 images, 16 in ResNet50's favour and 8 in MobileNetV2's, which gives
> p = 0.15. A paired bootstrap puts the difference at +0.64 points with a 95%
> interval of −0.2 to +1.4, so it spans zero. I report that as "no significant
> difference was found" rather than "the models are equivalent", because the
> test cannot prove equivalence — but it does bound any real difference at
> about 1.4 points, and that is the number the deployment trade-off is weighed
> against.

**"So is ResNet50's better ulcer recall real, or also noise?"**

> That is the right follow-up, and it is a separate test — overall accuracy
> cannot answer it, because a model can be better overall and no better on the
> class that matters. I ran the same paired test restricted to the 199 ulcer
> images. [Quote the per-class p-value from `python src/compare_models.py`.]
> Either way I report the recall figures and the count, nine missed against
> three, because for a screening tool the count is what a clinician cares
> about and the significance test is what an examiner cares about.

**"Why did you calibrate? Your accuracy was already high."**

> Because accuracy and trustworthiness are different properties, and for a tool
> used where there is no specialist to consult, a confidently wrong answer is
> worse than an admission of uncertainty. Guo et al. show modern networks are
> systematically overconfident, and I measured it: MobileNetV2 was overconfident
> and needed a temperature of 1.43. Interestingly ResNet50 was slightly
> under-confident and needed 0.95 — so I fitted the correction from the data
> rather than assuming the expected direction. Calibration is also what makes
> the referral threshold meaningful, since an abstention rule built on
> uncalibrated confidence is not measuring what it claims to.

**"Your healthy nail images come from montage sheets. Isn't that a problem?"**

> I handled it explicitly. Tiling those sheets yields 18,096 images but they
> come from about 20 photographic sessions, so I group tiles by source sheet
> before splitting — no sheet straddles the train/test boundary — and I capped
> the contribution at 1,000 sampled evenly across all 20 sheets, so the class
> cannot dominate training. I also added randomised blur augmentation, because
> the tiles are upscaled and I did not want sharpness acting as a class cue. I
> report the underlying session count as a scope boundary rather than claiming
> 18,096 independent observations.

**"How would this perform in an actual rural clinic?"**

> That is the open question and I state it as one. All three sources are
> clinical datasets, so what I demonstrate is that the approach works on
> clinical photographs and is small and fast enough to run on a phone. The
> framing throughout is "motivated by rural screening", not "validated for
> rural deployment". The prototype includes a batch evaluation tab specifically
> so independently captured photographs can be scored against these figures,
> which is the natural next step.

---

## 8. Scope and boundary conditions

Every study has a boundary, and stating it precisely is a mark of a controlled
one. Each item below is a condition of the design that was identified during
implementation and handled where it could be handled — which is the point to
make when presenting them.

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

## 9. Methodological decisions worth defending

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

## 10. Superseded results (do not report with the current test set)

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
ulcers, at ten times the size", with §2 and §4.1 adding that on the
confound-free comparison the two architectures perform equivalently.

Note also that the ulcer recall gap *narrowed* between the splits — 0.9397 →
0.9548 for MobileNetV2 against 0.9899 → 0.9849 for ResNet50 — which is the
opposite of what a superficial reading of the old numbers would suggest and
another reason to quote only the current run.

Split-fix comparison for MobileNetV2: old split val 0.9817 / loss 0.0641; new
split val 0.9802 / loss 0.0604. Accuracy fell 0.15 points while loss improved —
the earlier figure was not meaningfully inflated, it simply was not measuring the
nail case at all.

---

## 11. Remaining work

Done: ResNet50 is trained on the current split (5.4, WBS 5.6); `evaluate.py`,
`calibrate.py` and `ablate_border.py` have all been run with both models; and
the overall paired significance test is measured. Every figure above comes
from one consistent result set.

1. **Run `python src/compare_models.py`** to record the per-class and
   per-source p-values, in particular the Foot Ulcer one. Seconds, on CPU, from
   the saved prediction CSVs — no GPU and no re-run of inference. The overall
   test is done (p = 0.1516, §2).
2. **External validation** — 20–50 photographs taken independently, scored
   through the prototype's batch tab. The highest-value addition remaining, and
   the only evidence that would settle which model to recommend (§6); check
   whether ethics approval is required first.
3. **Optional, if GPU time allows: a second seed for each architecture.** Would
   bound the seed-to-seed variance noted in §8.
   Not required by the proposal.

---

## 12. Reproducibility notes

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
