const { P, CH, H2, H3, cap, fcap, IMG, T, BREAK } = require('./lib');
const c = [];
c.push(...CH(5, 'Results'));

c.push(P('This chapter reports what the experiments produced. All figures come from a single run in which both architectures were trained on the same split and then evaluated, calibrated and ablated together, so nothing here mixes results from different runs. The test split was used once, at the end; every development decision used validation. Interpretation is held back for Chapter 6.'));

c.push(H2('5.1  Dataset and training'));
c.push(P('Preparation reduced 21,733 candidate images to the 8,374 used, and the split placed 1,248 of them in the test set. Table 5.1 gives the count at each stage.'));
//@TABLE 5.1
c.push(P('The two architectures were trained under identical conditions. Both improved substantially in the second stage, which is the expected behaviour of fine-tuning and confirms the staged procedure was worth its extra complexity.'));
//@TABLE 5.2

c.push(H2('5.2  Test set performance'));
c.push(P('Both models were evaluated once on the 1,248 held-out images.'));
//@TABLE 5.3
//@TABLE 5.4
c.push(IMG('confusion.png', 560, 267));
c.push(fcap('5.1', 'Confusion matrices on the test set. Counts are images; rows are the true class and columns the predicted class.'));
c.push(P('The clinically important difference is not the 0.64 accuracy points but the missed ulcers: nine for MobileNetV2 against three for ResNet50. ResNet50 is perfect on both nail classes and on healthy feet, and every one of its thirteen errors is a wound and ulcer confusion. The dominant error mode is the same for both models and is clinically coherent, since both are foot lesions; neither model ever classified a wound or an ulcer as healthy, which is the error that would matter most.'));

c.push(H2('5.3  Statistical comparison of the two models'));
c.push(P('Because both models are scored on the same images their errors are correlated, so the comparison is paired.'));
//@TABLE 5.6 AS 5.5
c.push(P('The models disagreed on 24 of the 1,248 images, sixteen favouring ResNet50 and eight favouring MobileNetV2, giving an exact p-value of 0.1516. The paired bootstrap bounds the difference at 1.4 accuracy points in ResNet50’s favour at 95 percent confidence, with the interval including zero. The difference is therefore not statistically established, and the interval says how large it could plausibly be, which is the figure a deployment decision needs.'));
c.push(P('Two per-class and per-source results matter. Foot ulcer is the largest effect in the study and the closest to significance at p = 0.0703, which is a trend rather than an established difference. And within mendeley_foot, the one source containing two classes and therefore the comparison in which dataset provenance cannot help either model, the two are statistically indistinguishable at p = 0.7744, with MobileNetV2 fractionally ahead.'));

c.push(H2('5.4  Computational cost'));
//@TABLE 5.7 AS 5.6
c.push(IMG('efficiency.png', 560, 200));
c.push(fcap('5.2', 'Model size and single-image inference latency. MobileNetV2 is a tenth of the size and roughly twice as fast on CPU.'));
c.push(P('MobileNetV2 is the faster model on both devices and by the wider margin on CPU, which is the deployment-relevant figure. The narrower GPU margin is an expected architectural effect, since depthwise separable convolutions cut operations in a way a CPU converts directly into speed but a GPU does not. These are batch-size-one latencies on shared hardware; the CPU ordering has been stable across every run of this study, which is why it is the figure quoted.'));

c.push(H2('5.5  Calibration and selective prediction'));
//@TABLE 5.8 AS 5.7
c.push(P('The two models needed opposite corrections. MobileNetV2 was overconfident and its temperature of 1.4278 softens its probabilities; ResNet50 was slightly underconfident and its temperature of 0.9495 sharpens them. Calibration error fell for MobileNetV2 and was already small for ResNet50. Neither temperature reordered any prediction, which was verified: accuracy is unchanged and only the confidence attached to each answer moves.'));
//@TABLE 5.9 AS 5.8
c.push(P('For MobileNetV2 this is the most defensible clinical statement available from these results: decline roughly three cases in a hundred and refer them, and be correct on 99.59 percent of the rest. Abstaining on about 37 low-confidence cases removes roughly 16 of the 21 errors. ResNet50 required no threshold, having met the 99 percent target while answering every validation case, so the selection procedure correctly returned none. Selective prediction therefore adds most where it is most needed, on the compact model intended for deployment.'));

c.push(BREAK());
c.push(H2('5.6  Testing for dataset shortcuts'));
c.push(P('All three tests described in Section 3.9 returned results consistent with the models classifying the condition rather than the source, for both architectures.'));
c.push(H3('5.6.1  Accuracy within a single source'));
//@TABLE 5.10 AS 5.9
c.push(P('This is the strictest test available in the data. Both models score close to 99 percent where knowing the dataset tells them nothing, and both classified all 413 healthy feet in this source correctly, so every error and every disagreement here was on a wound. A model separating healthy from diseased by recognising the dataset could not do that, because both classes come from the same dataset.'));
c.push(H3('5.6.2  Grad-CAM attention'));
c.push(P('Attention localises to the toes and nail plates with the image margins cold, consistently across every example inspected, and the overlays were generated for misclassified predictions as well as correct ones rather than curating successes. The evaluation script writes them for both architectures alongside the other results.'));
c.push(H3('5.6.3  Controlled occlusion'));
c.push(P('Border brightness does differ by class, which is why the test was run rather than assumed unnecessary: healthy images average 166.7 and 170.1 across their two sources against 107.6 for nail fungal and 112.7 for foot ulcer, and nearly a third of the healthy nail tiles have near-white borders.'));
//@TABLE 5.11 AS 5.10
c.push(IMG('ablation.png', 560, 385));
c.push(fcap('5.3', 'Controlled occlusion at 20-pixel border width. The interior control removes the same number of pixels as the border arm and differs only in location.'));
c.push(P('Masking the 20-pixel border takes MobileNetV2 from 0.9832 to 0.9671, a loss of 1.6 accuracy points, while masking the same number of interior pixels takes it to 0.8598, a loss of 12.3 points and roughly eight times as much. For ResNet50 the same arms give 0.9704 and 0.9030, losses of 1.9 and 8.7 points, a ratio of about four and a half. Neither model’s decision depends on the frame, and the equal-area control is what makes that statement evidence rather than assertion.'));
c.push(P('A fourth arm masks the interior and leaves only a 20-pixel frame, to measure how much either model can read from the border alone. MobileNetV2 scores 0.4952 against a majority-class floor of 0.4455, recovering only 62 of 692 non-healthy images; ResNet50 scores 0.5994, recovering 193. Per-class figures are in Appendix C. MobileNetV2 therefore extracts almost nothing from the frame and degrades into predicting the majority class rather than inventing a confident diagnosis, which is the behaviour wanted from a screening tool, while ResNet50 recovers about three times as much, the expected consequence of ten times the parameters.'));

c.push(H2('5.7  Referral policy'));
c.push(P('The escalation threshold was swept across the validation set, which contained 200 ulcers of which 11 were misclassified and 685 cases needing no urgent care; the full grid is in Appendix A. The threshold originally set by judgement, 0.20, escalated three of the eleven misclassified validation ulcers, and lowering it to 0.10 escalated six while still sending no benign case for urgent care. The 2 percent cost budget was never reached at any value on the grid, so the sweep bounded the benefit of escalation without ever pricing its cost. The threshold was set to 0.10 and the policy then audited on test at both values, so the effect of the change could be measured rather than assumed.'));
//@TABLE 5.15 AS 5.11
c.push(P('At the adopted threshold escalation fired on 32 of 1,248 test cases, 2.6 percent with a 95 percent interval of 1.8 to 3.6 percent. All nine ulcers MobileNetV2 misclassified were still routed to care and seven reached the urgent band; without escalation none would have, because all nine were classified as wounds and that class maps to the slower prompt band. The two that did not reach urgent were classified as wounds at 99.3 and 99.9 percent confidence, leaving almost no ulcer probability for any threshold to act on.'));
c.push(P('No healthy and no nail fungal image was referred for urgent or prompt care at either threshold. The measurable cost fell entirely on true wounds, of which 27 were raised from prompt to urgent at 0.10 against 14 at 0.20, so whether escalation exists is worth seven ulcers while where exactly the threshold sits is worth none.'));

c.push(BREAK());
c.push(H2('5.8  External validation on photographs taken outside the source datasets'));
c.push(P('All three source datasets are clinical collections, so everything above measures performance on that kind of image. Ten photographs were taken on mobile phones and scored through the prototype batch tab in the deployment configuration described in Section 3.12.'));
c.push(H3('5.8.1  What was collected'));
c.push(P('Five individuals were photographed, the author and four friends, each twice: once against a plain background and once against a patterned one, giving five matched pairs and ten images. Five different handsets were used across the five subjects, a Google Pixel, a Nothing Phone, a Samsung, an iPhone 14 and an iPhone 16 Pro, all indoors under ordinary tube lighting, so the set spans five cameras rather than one and the camera varies between subjects while the background comparison is made within one. All five subjects had healthy feet, so the true label for all ten is Healthy Foot/Nail. Every image is listed in Appendix B.'));
c.push(P('One correction should be recorded, because it changed a result rather than a label. The two photographs of subject 2 were uploaded with their background names transposed, and the figures below are corrected. Overall accuracy is unaffected, but the comparison between backgrounds moves from four out of five against two out of five to five out of five against one out of five. The error was detectable only because the batch tab records every filename.'));
c.push(H3('5.8.2  Accuracy'));
//@TABLE 5.17 AS 5.12
c.push(P('Six of the ten photographs were classified correctly, against 98.32 percent on the test set: a difference of 38.3 percentage points. The interval on ten images runs from 31.3 to 83.2 percent, so the measurement supports a direction rather than a magnitude. It supports the direction clearly, because 98.32 percent lies well outside that interval.'));
c.push(P('Macro F1 and macro recall should not be quoted for this set: they compute to 0.188 and 0.150 only because three of the four classes have no images here at all, so the meaningful figures are healthy recall, 6 of 10, and the false positive rate, 4 of 10. All four errors were the same error, a healthy foot classified as a foot wound. None was classified as an ulcer or as a nail condition, so all four run in the direction the referral policy was designed to tolerate. That is not evidence the system is safe on phone photographs: all five subjects had healthy feet, so this set contains no disease and cannot measure a missed diagnosis, which is the failure that matters clinically.'));
c.push(H3('5.8.3  The effect of the background'));
c.push(IMG('external.png', 560, 388));
c.push(fcap('5.4', 'Each subject photographed against both backgrounds. Lines join the two photographs of the same feet.'));
c.push(P('Every photograph taken against a plain background was classified correctly and four of the five taken against a patterned background were wrong. Because each subject appears under both conditions the appropriate test is McNemar’s exact test on the five pairs rather than a comparison of two proportions: four of the five pairs disagree and all four disagree in favour of the plain background, giving an exact two-sided p-value of 0.1250.'));
c.push(P('That is not significant, and it is worth being precise about why. With five pairs the smallest attainable two-sided p-value is 0.0625, which requires all five pairs to disagree in the same direction; four out of four gives 0.1250. The result is as one-directional as this many photographs allow and still cannot clear the conventional threshold, so what limits it is the size of the set rather than the strength of the effect. It is an observation with a plausible mechanism rather than a null result: a patterned background places edge and texture structure around the foot, exactly the kind of incidental detail Section 5.6.3 showed the models can read. Twenty to thirty pairs would settle it.'));
c.push(H3('5.8.4  What abstention did under these conditions'));
//@TABLE 5.18 AS 5.13
c.push(P('The threshold declined three of the four errors, raising accuracy on answered cases from 6 of 10 to 4 of 5 while declining half the set. On the test set the same threshold declined 3 percent of images; here it declined 50 percent, and that collapse is the mechanism working rather than failing: when the input is unlike the training data the probability vector flattens and the system stops answering instead of guessing. It also responded to the background, answering four of the five plain photographs and one of the five patterned. Mean confidence was 0.7789 against 0.9824 on test, and within the set confidence was lower on the four errors, 0.7373, than on the six correct, 0.8066. One error survived at 92.19 percent confidence, above any threshold that would leave the system usable.'));
c.push(H3('5.8.5  What the referral policy did with these ten'));
c.push(P('Four images were classified as wounds, which maps to the prompt band, and two correct healthy predictions fell below the abstention threshold, which the policy raises to routine. At least six of these ten healthy people are therefore routed to a clinician and at most four told no action is needed, these being bounds rather than exact counts because escalation can only raise a band. On the test set no healthy image at all was referred for urgent or prompt care.'));

c.push(H2('5.9  Behaviour on a condition outside the four classes'));
c.push(P('Section 5.8 changes how a photograph is taken while keeping the condition within the four classes. The opposite case was measured on the 578 nail dystrophy images excluded during preprocessing, which no model has seen at any stage.'));
//@TABLE 5.19 AS 5.14
c.push(P('Confidence on a condition the model cannot represent is indistinguishable from confidence on the data it was trained for: the means differ by 0.0017 and the medians by 0.0001, and slightly fewer of the unknown images were declined than of the known ones. No threshold placed on confidence can separate the two cases. Read alongside Section 5.8.4 this is sharper than either measurement alone: confidence fell steeply on unfamiliar photographs of familiar conditions and did not move at all on an unfamiliar condition.'));
//@TABLE 5.20 AS 5.15
c.push(P('Two results follow. Every one of the 578 images was assigned to a nail class and none to a foot class, so the model reliably identifies what part of the body it is looking at; what it cannot do is place a condition outside its vocabulary. And the label it assigns is the nearest available one, which shares the discoloration and thickening that characterise dystrophy. Because that class maps to the routine band, 569 of 578 were routed to a clinician, receiving the wrong name and the right action. Nine were told nothing was wrong when they had a real nail condition.'));

c.push(H2('5.10  Summary'));
c.push(P('MobileNetV2 reached 98.32 percent accuracy and ResNet50 98.96 percent. The difference is not statistically significant (p = 0.1516), a paired bootstrap bounds it at 1.4 accuracy points, and on the one comparison in which provenance cannot help either model the two are indistinguishable (p = 0.7744). ResNet50 missed three of the 199 test ulcers against nine, the largest effect in the study and also short of significance (p = 0.0703). MobileNetV2 achieved this with a tenth of the parameters and roughly half the CPU inference time, and supported an abstention threshold that raised accuracy on answered cases to 99.59 percent. All three tests for dataset shortcuts were consistent with the models classifying the condition rather than the source, and the referral policy brought seven of the nine misclassified ulcers into the urgent band while referring no healthy or nail fungal case for urgent or prompt care.'));
c.push(P('Two measurements test the boundaries of those figures. On ten phone photographs taken outside all three source datasets accuracy was 6 of 10, with every plain-background image correct and four of five patterned ones wrong, and the abstention threshold declined three of the four errors and half the set. On 578 images of a nail condition with no class in the system, confidence was indistinguishable from confidence on the test set and only 2.6 percent were declined, although 98.4 percent were still routed to a clinician. Together these locate the abstention mechanism precisely: it responds to an unfamiliar photograph of a familiar condition and not at all to a condition it was never taught. Chapter 6 interprets these results.'));

module.exports = c;
