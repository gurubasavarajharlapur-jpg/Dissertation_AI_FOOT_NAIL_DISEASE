# Findings and analysis

Everything measured during implementation, with the reasoning behind each
decision. Written to be quoted from directly when drafting the dissertation.

## Headline results

- **98.96% test accuracy** (ResNet50) and **98.32%** (MobileNetV2) on 1,248
  held-out images, evaluated once, across four classes.
- **Foot ulcer recall 98.5% / 95.5%** — the clinically costly error, reported
  ahead of overall accuracy.
- **Three independent tests confirm the models classify pathology, not dataset
  provenance** (§5): 98.99% accuracy within a single source, Grad-CAM attention
  on toes and nail plates, and a controlled occlusion ablation with an
  equal-area interior control.
- **No statistically significant accuracy difference between the two
  architectures** — McNemar's exact test, p = 0.15, with any real difference
  bounded at +1.4 points at 95% confidence. On the confound-free comparison
  they are indistinguishable (p = 0.77). MobileNetV2 reaches that at a tenth of
  the parameters and roughly twice the CPU speed (§2, §7).
- **Calibrated confidence with a referral threshold**: decline the least
  confident 3% of cases and refer them, and be correct on **99.6%** of the rest
  (§3).
- **A safety-first referral policy, measured rather than asserted** (§4):
  escalation restores the correct urgency band to **7 of the 9 ulcers the
  classifier misreads** — none of which would reach it otherwise — while
  referring **0 of 556 healthy** and **0 of 117 fungal** cases for urgent or
  prompt care. The result holds at every escalation threshold tried.
- A **working prototype** with screening and batch-evaluation tabs, Grad-CAM
  explanation and clinical guidance text (Phase 6).

**Status of the numbers.** Both architectures are trained on the final
(class × source stratified) split and evaluated, calibrated and ablated in a
single run, so every figure here comes from one consistent result set. The
earlier class-only-split figures are kept in §11, marked superseded.

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

Full per-class precision, recall and F1 for both models, for the results table:

| Class | MNv2 P | MNv2 R | MNv2 F1 | RN50 P | RN50 R | RN50 F1 |
|---|---|---|---|---|---|---|
| Healthy Foot/Nail | 0.9982 | 0.9946 | 0.9964 | 0.9982 | 1.0000 | 0.9991 |
| Nail Fungal Infection | 0.9667 | 0.9915 | 0.9789 | 0.9915 | 1.0000 | 0.9957 |
| Foot Wound/Injury | 0.9761 | 0.9787 | 0.9774 | 0.9919 | 0.9734 | 0.9826 |
| Foot Ulcer | 0.9645 | 0.9548 | 0.9596 | 0.9608 | 0.9849 | 0.9727 |

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
the condition (§5.1). **There the two architectures are indistinguishable** — 8
errors against 10 on 789 images, a difference well inside sampling noise.

So the compact model performs at the level of an architecture ten times its
size on the strictest comparison available, and ResNet50's measured advantage
comes from the single-class sources. Both facts are worth reporting, and
together they are the quantitative basis for recommending MobileNetV2 for
deployment (§7).

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
source, from the saved prediction CSVs — seconds, on CPU, no re-run of
inference. All results at 1,248 images:

| Comparison | n | MobileNetV2 | ResNet50 | Disagree | p |
|---|---|---|---|---|---|
| **Overall** | 1,248 | 21 errors | 13 errors | 24 (16/8) | **0.1516** |
| Healthy Foot/Nail | 556 | 3 | 0 | 3 (3/0) | 0.2500 |
| Nail Fungal Infection | 117 | 1 | 0 | 1 (1/0) | 1.0000 |
| Foot Wound/Injury | 376 | 8 | 10 | 12 (5/7) | 0.7744 |
| **Foot Ulcer** | 199 | 9 | 3 | 8 (7/1) | **0.0703** |
| figshare_nail | 117 | 1 | 0 | 1 (1/0) | 1.0000 |
| figshare_nail_tiles | 143 | 3 | 0 | 3 (3/0) | 0.2500 |
| **mendeley_foot** *(two classes)* | 789 | 8 | 10 | 12 (5/7) | **0.7744** |
| ulcer_fuseg | 199 | 9 | 3 | 8 (7/1) | 0.0703 |

Disagreements are shown as (to ResNet50 / to MobileNetV2). **No comparison in
the study reaches significance at the 5% level.** Two rows carry the argument.

**`mendeley_foot`, p = 0.7744 — the models are indistinguishable where it
counts.** This is the confound-free comparison: 789 images, both healthy feet
and wounds, one clinic, one camera, so recognising the dataset cannot
substitute for recognising the condition. The models disagree on 12 images, 7
of them in MobileNetV2's favour, and the test is nowhere near significance.

A detail worth quoting: **both models classified all 413 healthy feet in this
source correctly.** Every error and every disagreement on `mendeley_foot` was
on a wound. Where the comparison is cleanest, the compact model is not behind —
it is fractionally ahead, on a difference far too small to claim.

**Foot Ulcer, p = 0.0703 — the one comparison that approaches significance, and
it favours ResNet50.** Nine missed ulcers against three, with 7 of the 8
disagreements in ResNet50's favour. It does not cross the conventional 5%
threshold on 199 ulcer images, so it cannot be reported as an established
difference. But it is the largest effect in the study and the direction is
consistent, so the honest description is a **trend, on the clinically most
important class, that a larger ulcer test set could confirm.**

Do not overstate this in either direction. It is not a proven deficiency in
MobileNetV2, and p = 0.07 is not "nearly significant" in a way that licenses
treating it as significant. It is a signal worth acting on in design rather
than dismissing — which is exactly what the referral threshold in §3 does, and
why that mechanism matters more for the deployment candidate than for the
reference model.

### What the significance testing establishes

1. **The headline accuracy gap is not established** (p = 0.15), and any real
   difference is bounded at +1.4 points at 95% confidence.
2. **Where dataset provenance cannot help either model, they are
   indistinguishable** (p = 0.77), with MobileNetV2 fractionally ahead.
3. **The one comparison that trends is Foot Ulcer** (p = 0.07), favouring
   ResNet50 — reported as a trend, and addressed by abstention rather than by
   changing the recommended architecture.

That is a more complete answer than "ResNet50 scored 0.64 points higher", and
it is the answer WBS 5.6 asks for.

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
four grounds for the recommendation in §7.

---

## 4. Safety-first triage, and what it costs

The classifier reports appearance; the prototype has to recommend an action.
Those are different problems, and the second one is asymmetric. A false
referral costs an appointment. A missed ulcer can cost a limb: diabetic foot
ulceration precedes roughly 80% of non-traumatic lower-limb amputations, and
five-year mortality after a first diabetic foot ulcer is comparable to several
common cancers (Armstrong et al., *NEJM* 2017).

So the referral policy (`src/triage.py`) is deliberately biased toward
referral, in three rules:

1. **A serious condition holding real probability escalates the band even when
   it is not the top class.** `P(foot_ulcer) ≥ 0.20` forces the urgent band;
   `P(ulcer) + P(wound) ≥ 0.20` forces prompt.
2. **Abstention never reassures.** Below the referral threshold the tool names
   no condition and routes to a clinician — an inconclusive screening is not a
   negative result.
3. **Every result carries the override text**, a clean one included, naming
   what should send someone to care regardless of what the tool said.

The thresholds are a **design decision, not a model output** — the network
never saw severity, infection, depth, perfusion or patient history — and they
live in `config.py`. `config.validate()` refuses to start if a class has no
band or if `foot_ulcer` maps to anything but urgent.

### Measured on the test set (MobileNetV2, 1,248 images)

A policy biased toward referral is only defensible if the bias has been priced.
`src/audit_triage.py` measures it from the saved predictions.

**Escalation fires on 26 of 1,248 cases — 2.1% [1.4%, 3.0%].** Twenty-one move
prompt → urgent, five move self-care → routine. By true class: 7 ulcers, 14
wounds, 4 healthy, 1 fungal.

**The nine ulcers the classifier misreads:**

| Outcome | Count |
|---|---|
| Escalated to **urgent** — correct urgency restored | **7 of 9** |
| Referred at *prompt* — correct referral, slower than an ulcer needs | 2 of 9 |
| Not referred at all | **0 of 9** |

All nine are still routed to care. Seven have their urgency corrected *by
escalation*; the two others were called Foot Wound at 99.3% and 99.9%
confidence, where no probability-based rule can fire — they are referred only
because wound already maps to prompt. Report those two honestly: referred, but
told "within a few days" when an ulcer warrants 24 hours.

**The cost, on this test set, is close to zero:**

| Cost | Rate |
|---|---|
| Healthy referred (urgent/prompt) | **0 of 556** — 0.0% [0.0%, 0.7%] |
| Nail fungal referred (urgent/prompt) | **0 of 117** — 0.0% [0.0%, 3.2%] |
| True wounds raised to urgent | 14 of 376 — over-caution, not harm |

Not one healthy or fungal case was sent for urgent or prompt care.

### What escalation contributes, stated precisely

| Policy | Sensitivity | Specificity |
|---|---|---|
| Refer on predicted class | 574/575 — 99.8% | 673/673 — 100% |
| Refer on triage band | 574/575 — 99.8% | 673/673 — 100% |

(*Serious* = ulcer or wound; the question is "does this person need to be
seen?")

**Referral coverage is unchanged.** That is not a failure of the policy, it is
the correct reading of it: the class ulcers are misread as — wound — already
triggers referral, so there was little coverage left to add. The single serious
case neither policy refers is a wound confidently misread as fungal.

**Escalation's contribution is urgency, not coverage**, and that is the claim
to make:

> The referral policy escalated 2.1% of cases [1.4–3.0%]. It restored the
> correct urgency band to seven of the nine ulcers the classifier misread,
> while referring no healthy or fungal case for urgent or prompt care. Overall
> referral sensitivity was unchanged (574/575 under either policy), because the
> class ulcers are misread as already triggers referral; the policy's
> contribution is to the *timeframe* given, not to whether care is sought.

That is a safety mechanism that measurably improves the handling of the
clinically critical class at no measured cost — a stronger and more specific
claim than "the prototype gives clinical guidance".

### Tuning the thresholds on validation

The escalation thresholds were set by judgement, so `src/sweep_thresholds.py`
tunes them against the **validation** split — never test, which by this point
has been read and cannot be tuned against without spending it. It evaluates
candidates by calling the real `triage.assess` with the value injected, so what
is tuned is what ships.

Validation: 1,261 images, 200 ulcers of which 11 are misclassified, 685 benign.

**`TRIAGE_URGENT_PROB` — P(ulcer) forcing the urgent band:**

| Threshold | Misclassified ulcers reaching urgent | Benign sent urgently |
|---|---|---|
| **0.02** | **10 of 11 — 90.9%** | 2 of 685 — 0.3% |
| 0.04 | 8 of 11 — 72.7% | 1 of 685 — 0.1% |
| 0.06 | 7 of 11 — 63.6% | 1 of 685 — 0.1% |
| 0.08 | 6 of 11 — 54.5% | 1 of 685 — 0.1% |
| **0.10 (adopted)** | **6 of 11 — 54.5%** | **0 of 685 — 0.0%** |
| 0.14 | 4 of 11 — 36.4% | 0 of 685 — 0.0% |
| 0.18 | 3 of 11 — 27.3% | 0 of 685 — 0.0% |
| **0.20 (original)** | **3 of 11 — 27.3%** | 0 of 685 — 0.0% |
| 0.24 | 2 of 11 — 18.2% | 0 of 685 — 0.0% |
| 0.32 | 1 of 11 — 9.1% | 0 of 685 — 0.0% |
| 0.38+ | 0 of 11 — 0.0% | 0 of 685 — 0.0% |

**The configured 0.20 is too conservative, and this is the sweep's clear
finding.** It escalates 3 of the 11 misclassified ulcers where 0.02 escalates
10, and the extra cost of getting there is two benign people receiving an
urgent referral they did not need — about **3.5 ulcers correctly escalated per
unnecessary appointment.** Clinically that trade is not close.

**`TRIAGE_REVIEW_PROB` — P(ulcer) + P(wound) forcing prompt:** referral of
serious cases holds at 100% from 0.02 all the way to 0.30, and the configured
0.20 and the selected 0.30 are **identical on validation** — both refer 576/576
serious and 3/685 benign. There is no measurable reason to change it.

### Two things the sweep could not settle

**The cost budget never binds.** At 2% of benign cases the budget allows 13.7
urgent referrals; the most aggressive threshold on the grid spends 2. The
constraint is inactive across the entire range, which means the selection rule
degenerated to "maximise benefit" and **validation never exercised the
cost side of the trade-off at all.** Report that honestly: the sweep bounded
the benefit, it did not price the cost, because on in-distribution data this
model puts almost no probability mass on ulcer for a benign foot.

That is also the reason not to read 0.02 as simply optimal. The cost of a low
threshold is paid under *distribution shift* — on a phone photograph in poor
light, where the probability vector is flatter and 2% ulcer mass may mean
nothing — and validation, drawn from the same clinical sources as training,
cannot measure that. The external validation photographs (§12) are the evidence
that would settle it.

**One ulcer is unreachable.** Even at 0.02 the figure is 10 of 11: one
misclassified ulcer carries less than 2% ulcer probability, a confident error
no probability-based rule can catch at any threshold. It is the same limitation
the test-set audit found in its two confident misreads, and it needs a better
classifier rather than a better policy.

### The decision, and why it is not the rule's answer

**`TRIAGE_URGENT_PROB` is set to 0.10.** It reaches 6 of the 11 misclassified
validation ulcers against 0.20's 3, and sends **no benign case at all** for
urgent care — 0 of 685. Doubling the benefit at zero measured cost is not a
close call.

**0.02, which the stated rule selected, was not adopted**, and the reasoning
belongs in the write-up rather than buried in a config comment:

- The rule's cost budget never bound, so the rule was not doing the job it was
  designed for. It was meant to trade benefit against cost; with no cost
  pressure anywhere on the grid it reduced to "maximise benefit", and its answer
  is the end of the grid rather than an optimum.
- 2% of the probability mass in a four-class softmax sits close to the noise
  floor. Validation is drawn from the same clinical sources as training, so it
  cannot show what a flatter, less confident vector from a phone photograph
  would do at that threshold. 0.10 keeps a five-fold margin.

Departing from a pre-stated selection rule needs justifying, which is why both
numbers are reported. The defensible claim is that the sweep established the
configured 0.20 was too conservative — that part is unambiguous — and that the
operating point within the affordable range was then chosen on robustness
grounds validation could not measure.

**`TRIAGE_REVIEW_PROB` stays at 0.20.** The sweep preferred 0.30, but the two
are identical on validation (576/576 serious referred, 3/685 benign), so there
is nothing to gain.

### Re-audited at the adopted 0.10 threshold

Moving the threshold from 0.20 to 0.10 was expected to reach more of the
misclassified ulcers, because that is what validation showed. **On test it
reached exactly the same number**, and this is the more interesting result:

| Measure (test, 1,248 images) | at 0.20 | at 0.10 |
|---|---|---|
| Escalation fired | 26 — 2.1% | 32 — 2.6% |
| Misclassified ulcers referred | 9 of 9 | 9 of 9 |
| **Misclassified ulcers reaching urgent** | **7 of 9** | **7 of 9** |
| Healthy referred | 0 of 556 | 0 of 556 |
| Nail fungal referred | 0 of 117 | 0 of 117 |
| True wounds raised to urgent | 14 of 376 — 3.7% | 27 of 376 — 7.2% |
| Sensitivity / specificity | 574/575, 673/673 | 574/575, 673/673 |

The benefit validation predicted did not appear, and the cost did: wound cases
raised to urgent nearly doubled while the ulcer outcome did not move at all.

**Why.** No misclassified test ulcer had P(ulcer) between 0.10 and 0.20, so
lowering the threshold across that band changed nothing. The two ulcers still
at *prompt* were called Foot Wound at 99.3% and 99.9% confidence, putting
P(ulcer) below roughly 0.007 and 0.001 — no threshold worth setting reaches
them. On validation three misreads did fall in that band; on test none did.
With 11 misclassified ulcers in validation and 9 in test, neither split can
resolve a difference of this size, and the disagreement between them is
sampling noise rather than a contradiction.

**The threshold is kept at 0.10.** Reverting because test did not replicate the
gain would be tuning on test, which is exactly the thing this project has
avoided throughout — the rule was fixed in advance as "tune on validation", and
honouring it when the answer is inconvenient is what makes it a rule. The cost
of being wrong here is bounded and benign: the additional cases are **true
wounds**, who need care regardless and are told to seek it within 24 hours
rather than a few days. The number that would represent real cost — well people
sent for urgent care — did not move at all, and stayed at zero for both healthy
and fungal cases.

### The conclusion the two audits support together

Running the audit at both thresholds gives a cleaner result than either alone:

> Without escalation, none of the nine misclassified ulcers reaches the urgent
> band — all map to *prompt* through the wound class. With escalation, seven do,
> at **any** threshold between roughly 0.01 and 0.20. Whether escalation exists
> is worth seven ulcers; where the threshold sits inside that range is worth
> none.

That is the claim to make. It is more robust than a figure tied to one
threshold, it does not depend on the validation/test disagreement resolving
either way, and it puts the emphasis where the evidence actually is — on the
policy rather than on its tuning.

### Limitations of this measurement

- **The headline test figures are identical at both thresholds tried.** 7 of 9
  misclassified ulcers reach urgent at 0.20 and at 0.10; only the number of
  true wounds raised to urgent changes (14 vs 27). Label triage figures with
  the threshold that produced them anyway, since the wound figure does differ.
- **Zero false referrals is a test-set result, not a guarantee.** The upper
  confidence bound is 0.7% for healthy cases, so up to about four in every 556
  is consistent with this evidence.
- **Two of nine ulcers still receive the slower timeframe.** No
  probability-based escalation can fix a confident misclassification; that
  needs a better classifier, not a better policy.

---

## 5. Validation: both models read pathology, not dataset artefacts

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

**Both models also classified all 413 healthy feet in this source correctly** —
every error and every disagreement here was on a wound. A model separating
healthy from diseased by recognising the dataset could not do that, because
both classes come from the same dataset.

It is also where the deployment recommendation is won: on the strictest test in
the study the two architectures are **statistically indistinguishable**
(McNemar's exact test, p = 0.77 — they disagree on 12 of 789 images, 7 of them
in MobileNetV2's favour). The compact model matches an architecture ten times
its size exactly where the comparison is cleanest.

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
It does not affect either model's validity — §5.3 establishes that neither
*relies* on the border, which is the question that matters — but it is a
further measured argument for the compact architecture, and a good answer to
"why not just use the bigger model?"

---

## 6. Efficiency, measured on both CPU and GPU

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

## 7. Recommendation

The proposal frames this as MobileNetV2 (deployment candidate) against ResNet50
(accuracy reference). The evidence supports a clear recommendation.

**MobileNetV2 is recommended for phone-based rural screening**, on four
measured grounds:

1. **There is no statistically significant accuracy difference between the two
   models.** McNemar's exact test on the paired per-image outcomes gives
   p = 0.15, with the difference bounded at +1.4 accuracy points at 95%
   confidence (§2). On the confound-free comparison the two are
   indistinguishable — 0.9899 against 0.9873 on 789 within-source images,
   p = 0.77, with MobileNetV2 fractionally ahead (§5.1).
2. **It is a tenth of the size and 1.9× faster on CPU** — 21.78 MB against
   210.55 MB, 248.29 ms against 460.56 ms per image. CPU is the device class a
   mid-range phone resembles.
3. **It is the more robust of the two to dataset artefact** — it recovers 62 of
   692 non-healthy images from the border alone against ResNet50's 193, and
   degrades gracefully rather than confidently when the lesion is hidden (§5.4).
4. **It supports a working abstention threshold**: refuse the least confident
   3% and refer them, and be right 99.6% of the time on the rest (§3).

**ResNet50 is the stronger accuracy reference, and should be reported as such.**
0.9896 accuracy, 0.9875 macro F1, perfect recall on both nail classes, and
three missed ulcers against nine — the one comparison in the study that
approaches significance (p = 0.07, §2). Where a server is available and the
ulcer miss rate outweighs every other consideration, it is the better choice.
That finding is a genuine part of the contribution, not a concession: the
comparison was run precisely so the trade-off could be quantified rather than
assumed, and the ulcer trend is the most useful thing it surfaced.

**The ulcer trend is addressed by design, not by ignoring it.** MobileNetV2 is
the model that carries a working referral threshold (§3) and the escalation
policy measured in §4, and abstention is
precisely the mechanism for a class the model is least certain about: decline
the least confident cases and route them to a clinician. An examiner asking
"but ResNet50 misses fewer ulcers" should get the trade-off *and* the
mitigation, not a dismissal of the number.

**The contribution is the quantified trade-off**, which is what WBS 5.6 asks
for: a 0.64-point accuracy gap that does not reach significance, six fewer
missed ulcers, bought at 10.4× the parameters, 9.7× the storage and 1.9× the
CPU latency. A dissertation that reports only "ResNet50 scored higher" has not
answered the research question — it has not established that the difference is
real, nor priced it. This one does both.

One honest qualification to carry into the discussion. Failing to establish a
difference is not the same as establishing equivalence, and with 1,248 test
images — 199 of them ulcers — this study cannot resolve a difference smaller
than roughly 1.4 accuracy points overall, nor settle the ulcer trend either
way. The recommendation does not depend on the two models being identical. It
depends on the difference being small enough that a tenfold reduction in size
is worth it, on that difference being bounded, and on the residual ulcer risk
having a mitigation. All three hold.

---

## 8. Viva questions, with answers

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
> against. I also ran the test within each class and each source, because an
> overall figure can hide a difference on one class.

**"So is ResNet50's better ulcer recall real, or also noise?"**

> That is the right follow-up, and it needs its own test — overall accuracy
> cannot answer it, because a model can be better overall and no better on the
> class that matters. I ran the same paired test restricted to the 199 ulcer
> images: they disagree on 8, seven in ResNet50's favour, p = 0.07. So it does
> not reach significance, but it is the largest effect in my study and the only
> comparison that comes close, and the direction is consistent. I report it as
> a trend on the clinically most important class that a larger ulcer test set
> could confirm — not as an established difference, and not as nothing.

**"If ResNet50 might genuinely be better at ulcers, why is your recommendation
still MobileNetV2?"**

> Because the recommendation is for a phone-based screening tool, and it comes
> with a referral threshold. Abstention is exactly the right mechanism for the
> class a model is least certain about: MobileNetV2 declines the least
> confident 3% of cases and refers them to a clinician, and is correct on 99.6%
> of what it does answer. So the design response to a possible ulcer weakness
> is to refer, not to ship a model ten times the size onto a phone that cannot
> run it. If the deployment target were a clinic server rather than a handset,
> my recommendation would be ResNet50 — that is the trade-off my comparison was
> built to quantify.

**"Nothing in your study reached statistical significance. Isn't that a weak
result?"**

> It is a clear result, and it is the one my research question asked for. The
> question was whether a compact architecture is good enough for phone-based
> screening, not which of two models scores higher. Finding no significant
> difference between MobileNetV2 and ResNet50 — and bounding any real
> difference at 1.4 accuracy points — is direct evidence for the deployment
> case, because MobileNetV2 delivers it at a tenth of the parameters and
> roughly half the CPU latency. A significant gap in ResNet50's favour would
> have been the weaker result for this project.

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

**"Your model misses about one ulcer in twenty. Is that safe to deploy?"**

> That is the right question, and it is why the prototype does not just report
> the top class. The referral policy escalates when a serious condition holds
> meaningful probability even if it is not the most likely one, so a result of
> "healthy 55%, ulcer 40%" is reported as urgent, not as healthy. I measured
> what that does on the test set: escalation fires on 2.1% of cases, and of the
> nine ulcers the classifier misreads, seven have their urgency corrected to
> urgent and all nine are still routed to care. It referred no healthy or
> fungal case for urgent or prompt attention, so on this data the safety margin
> was free.
>
> I would not claim the policy fixes the classifier. Two of those nine were
> confident misreads — called wound at over 99% — where no probability rule can
> fire, and they get "within a few days" when an ulcer warrants 24 hours. That
> needs a better model, not a better policy, and I state it as a limitation.

**"Where did the escalation threshold come from?"**

> It started as my judgement from the cost asymmetry, and I then swept it on
> the validation split. The sweep found my original 0.20 too conservative — it
> escalated 3 of 11 misclassified validation ulcers where 0.10 escalated 6 at
> no measured cost — so I moved it to 0.10. I did not take the value the
> selection rule actually returned, 0.02, and I report that openly: the rule's
> cost budget never bound, so it had collapsed into maximising benefit with
> nothing pushing back, and 2% of the mass in a four-class softmax is close to
> the noise floor. I did not tune against test at any point.

**"Did lowering the threshold actually help?"**

> On test, no — and I report that rather than only the validation result. Seven
> of the nine misclassified ulcers reach the urgent band at both 0.20 and 0.10;
> what changed is that true wounds raised to urgent went from 14 to 27. No test
> misread happened to fall in the band I lowered through, whereas three
> validation misreads did. With 9 and 11 misclassified ulcers respectively,
> neither split can resolve a difference that small.
>
> I kept 0.10 anyway, because reverting on test evidence would be tuning on
> test, and the rule was fixed in advance. The cost of being wrong is bounded:
> the extra escalations are true wounds, who need care regardless and are simply
> told to seek it sooner. Referral of well people stayed at zero throughout.
>
> Running it at both thresholds gave me a better claim than either alone.
> Without escalation none of those nine ulcers reaches urgent; with it, seven
> do, at any threshold between about 0.01 and 0.20. Whether escalation exists
> is worth seven ulcers — where exactly the threshold sits is worth none.

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

## 9. Scope and boundary conditions

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

**Class is partly confounded with source.** Addressed by the three tests in §5,
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

**Model capacity and artefact absorption move together.** §5.3 shows ResNet50
recovering three times as much class information from a 20px border as
MobileNetV2. Choosing the larger model for accuracy also buys more sensitivity
to whatever is incidental in the data — relevant to any future work scaling up
the backbone.

---

## 10. Methodological decisions worth defending

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

## 11. Superseded results (do not report with the current test set)

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
ulcers, at ten times the size", with §2 and §5.1 adding that on the
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

## 12. Remaining work

Done: ResNet50 is trained on the current split (5.4, WBS 5.6); `evaluate.py`,
`calibrate.py` and `ablate_border.py` have all been run with both models; the
paired significance testing is complete, overall and within every class and
source; and the triage thresholds have been swept on validation and the policy
audited on test at both the old and the adopted threshold. Every figure above
comes from one consistent result set.

1. **External validation** — 20–50 photographs taken independently, scored
   through the prototype's batch tab. The highest-value addition remaining, and
   the only evidence that would settle which model to recommend (§7); check
   whether ethics approval is required first.
2. **Optional, if GPU time allows: a second seed for each architecture.** Would
   bound the seed-to-seed variance noted in §9. Not required by the proposal.
3. **Optional: more ulcer test images.** The one trend in the study (p = 0.07)
   sits on 199 ulcers. More would settle it either way, and settling it is the
   single most useful additional measurement available. Not required by the
   proposal.

---

## 13. Reproducibility notes

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
