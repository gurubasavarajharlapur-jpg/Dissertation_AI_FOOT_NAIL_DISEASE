# Results

Everything below is measured. `docs/findings_and_analysis.md` holds the full
record with the reasoning behind each figure; this page is the summary.

Both architectures were trained on the same 8,374 images through the same code
path and evaluated once on the same 1,248 held-out images. The test split was
read once, at the end. Every development decision used the validation split.

## Test set, 1,248 images

| Metric | MobileNetV2 | ResNet50 |
|---|---|---|
| Accuracy | 0.9832 | **0.9896** |
| Macro F1 | 0.9781 | **0.9875** |
| Balanced accuracy | 0.9799 | **0.9896** |
| Cohen's kappa | 0.9751 | **0.9846** |
| Total errors | 21 | **13** |

Per-class recall matters more than accuracy here, because the classes are
imbalanced and a missed ulcer is the costly error.

| Class | MobileNetV2 | ResNet50 | Test images |
|---|---|---|---|
| Healthy Foot/Nail | 0.9946 | 1.0000 | 556 |
| Nail Fungal Infection | 0.9915 | 1.0000 | 117 |
| Foot Wound/Injury | 0.9787 | 0.9734 | 376 |
| **Foot Ulcer** | **0.9548** | **0.9849** | 199 |

## Is the difference real?

Both models see the same images, so their errors are correlated and the
comparison is paired rather than a contest between two accuracy figures.

| Comparison | Result |
|---|---|
| Overall, McNemar's exact test | p = 0.1516, not significant |
| Paired bootstrap, 10,000 resamples | ResNet50 ahead by at most 1.4 accuracy points, interval includes zero |
| Foot ulcer only | p = 0.0703, the largest effect in the study, still not significant |
| Within one source containing two classes | p = 0.7744, indistinguishable |

The last row is the strictest test available in this data. `mendeley_foot`
contributes both healthy feet and wounds photographed at the same clinic, so on
that comparison knowing the dataset tells a model nothing, and there the two
architectures perform the same.

## Cost

| | MobileNetV2 | ResNet50 | Ratio |
|---|---|---|---|
| Parameters | 2,263,108 | 23,595,908 | 10.4x |
| Size on disk | 21.78 MB | 210.55 MB | 9.7x |
| CPU inference, per image | 248.29 ms | 460.56 ms | 1.9x |

## Are the models reading the condition or the dataset?

Each class comes largely from one source, so this was tested three ways rather
than assumed.

| Test | Result |
|---|---|
| Accuracy within a single source holding two classes | 0.9899 and 0.9873, close to 99 percent where provenance cannot help |
| Grad-CAM attention | On the toes and nail plates, margins cold, including on misclassified images |
| Occlusion, 20px border masked | MobileNetV2 loses 1.6 accuracy points |
| Occlusion, equal area of interior masked | MobileNetV2 loses 12.3 points, roughly eight times more |

The interior control is what makes the occlusion test interpretable. Masking
anything removes information, so a small drop from masking the border proves
nothing on its own.

## Calibrated confidence and abstention

| | MobileNetV2 | ResNet50 |
|---|---|---|
| Temperature | 1.4278, softens | 0.9495, sharpens |
| Abstention threshold | 0.787 | none needed |
| Coverage on test | 97.0% | 100% |
| Accuracy on answered cases | **0.9959** | 0.9896 |

Declining the least confident 3 percent of cases and referring them removes 16
of the 21 errors.

## Where it breaks

Two measurements test performance outside the source datasets, and they are the
most important results here for anyone judging whether the approach is usable.

**Ten photographs taken on phones.** Five subjects, two backgrounds each, all
healthy feet. Six of ten classified correctly, against 98.32 percent on the test
split. The 95 percent interval runs from 31.3 to 83.2 percent, so ten images give
the direction of the drop and not its size. Every plain-background photograph was
correct and four of five patterned ones were wrong. All four errors were healthy
feet read as wounds, and because no subject had a condition, this set cannot
measure a missed diagnosis at all.

**578 images of a condition outside the four classes.** Nail dystrophy, excluded
during preprocessing and never seen by either model. Median confidence 0.9997
against 0.9998 on the test set, and 2.6 percent declined against 3.0 percent, so
confidence carries no signal that the input is unrepresentable. 97.8 percent were
labelled nail fungal infection, the nearest available class, and 98.4 percent
were still routed to a clinician because that class carries a referral.

Together these locate the abstention mechanism precisely: it responds to an
unfamiliar photograph of a familiar condition and not at all to a condition it
was never taught.

## Recommendation

MobileNetV2, for phone-based screening. No significant accuracy difference
against a model ten times its size, any real difference bounded at 1.4 points,
indistinguishable on the strictest comparison available, faster and smaller, and
it supports a working abstention threshold that ResNet50 did not need and
therefore does not have.

The one comparison favouring ResNet50, on ulcer recall, is a trend rather than an
established difference. The referral policy is the design response to it: it
brings seven of the nine ulcers MobileNetV2 misreads into the urgent band, none
of which would reach it otherwise.

## What this does not establish

No data from a rural or field setting was used at any stage. All three sources
are clinical collections, so this work is motivated by rural screening rather
than validated for it. Every result rests on one split and one training run per
architecture at a single seed. The system has no reject class, and the external
validation contains no disease.
