const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
        Table, TableRow, TableCell, WidthType, ShadingType, PageBreak } = require('docx');
const fs = require('fs');
const ACCENT = '1F4E79', GREY = '595959';

const P = (t, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 160, line: 300 },
  alignment: o.just === false ? undefined : AlignmentType.JUSTIFIED,
  children: [new TextRun({ text: t, size: 22, italics: o.i, bold: o.b, color: o.c, font: 'Calibri' })] });

const TODO = t => new Paragraph({ spacing: { before: 120, after: 160, line: 300 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4E5' },
  children: [ new TextRun({ text: 'TO COMPLETE:  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
              new TextRun({ text: t, size: 20, italics: true, font: 'Calibri' }) ] });

const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 340, after: 150 },
  children: [new TextRun({ text: t, font: 'Calibri' })] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 260, after: 120 },
  children: [new TextRun({ text: t, font: 'Calibri' })] });

const cap = (n, t) => new Paragraph({ spacing: { before: 240, after: 90 },
  children: [ new TextRun({ text: `Table ${n}.  `, bold: true, size: 20, font: 'Calibri' }),
              new TextRun({ text: t, size: 20, font: 'Calibri' }) ] });
const fcap = (n, t) => new Paragraph({ spacing: { before: 90, after: 200 }, alignment: AlignmentType.CENTER,
  children: [ new TextRun({ text: `Figure ${n}.  `, bold: true, size: 20, font: 'Calibri' }),
              new TextRun({ text: t, size: 20, font: 'Calibri' }) ] });
const FIG = t => new Paragraph({ spacing: { before: 180, after: 40 }, alignment: AlignmentType.CENTER,
  shading: { type: ShadingType.CLEAR, fill: 'F2F2F2' },
  children: [ new TextRun({ text: `[ INSERT FIGURE — ${t} ]`, size: 20, color: GREY, italics: true, font: 'Calibri' }) ] });

const cell = (t, w, o = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  shading: o.head ? { type: ShadingType.CLEAR, fill: 'E8EEF4' } : undefined,
  margins: { top: 55, bottom: 55, left: 100, right: 100 },
  children: [new Paragraph({ spacing: { after: 0 }, alignment: o.right ? AlignmentType.RIGHT : undefined,
    children: [new TextRun({ text: t, bold: o.head, size: 18, font: 'Calibri' })] })] });
const T = (widths, rows, o = {}) => new Table({ columnWidths: widths,
  width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0,
    children: r.map((c, j) => cell(String(c), widths[j],
      { head: i === 0, right: o.rightFrom !== undefined && j >= o.rightFrom && i > 0 })) })) });

const c = [];
c.push(new Paragraph({ spacing: { after: 300 },
  children: [new TextRun({ text: 'Chapter 5', bold: true, size: 36, color: ACCENT, font: 'Calibri' })] }));
c.push(new Paragraph({ spacing: { after: 360 },
  children: [new TextRun({ text: 'Results', bold: true, size: 36, color: ACCENT, font: 'Calibri' })] }));

c.push(P('This chapter reports what the experiments produced. It covers the training of both models, their performance on the held-out test set, the statistical comparison between them, their computational cost, their calibration, the three tests used to check whether they classify disease or dataset, the behaviour of the referral policy, an external validation on photographs taken outside the source datasets, and what the system does with a condition it has no class for. Interpretation of these results, and what they mean for the research objectives, is held back for Chapter 6.'));
c.push(P('All figures reported here come from a single run in which both architectures were trained on the same data split and then evaluated, calibrated and ablated together. This matters because an earlier split produced different numbers, and mixing results from two runs would make the comparison meaningless. The test set was used once, at the end. Every decision made during development — when to stop training, which checkpoint to keep, what temperature to apply, where to set the referral thresholds — was made on the validation set.'));

// ---- 5.1 ----------------------------------------------------------------
c.push(H2('5.1  Dataset and split'));
c.push(P('The final dataset contained 8,374 images across the four classes, drawn from three public sources. Table 5.1 shows how the raw files were reduced to this figure.'));
c.push(cap('5.1', 'Data preparation pipeline, with counts at each stage.'));
c.push(T([4200, 4000], [
  ['Stage', 'Result'],
  ['Files scanned in data/raw/', '26,544'],
  ['Mapped to one of the four classes', '25,689  (855 excluded)'],
  ['Excluded: nail dystrophy', '578'],
  ['Excluded: montage contact sheets', '277  (tiled separately)'],
  ['Duplicates removed', '1,758'],
  ['Montage tiles capped', '18,096 → 1,000'],
  ['Final dataset', '8,374'],
  ['Train / validation / test', '5,865 / 1,261 / 1,248'],
  ['Class imbalance ratio', '4.8 : 1'],
]));
c.push(P('Removing the 1,758 duplicates was not a formality. Without that step the same images would have appeared on both sides of the train and test boundary, and the reported accuracy would have been inflated by the model recognising pictures it had already seen.'));
c.push(P('The test set held 1,248 images: 556 healthy, 376 foot wound, 199 foot ulcer and 117 nail fungal infection. Broken down by source rather than class, it held 789 images from the Mendeley foot dataset, 199 from the ulcer dataset, 143 montage tiles and 117 onychomycosis images. The Mendeley foot dataset is the only source that contributes two classes, 413 healthy feet and 376 wounds, which is what makes it useful for the confound test reported in Section 5.6.'));

// ---- 5.2 ----------------------------------------------------------------
c.push(H2('5.2  Training'));
c.push(P('Both models were trained using the same two-stage procedure described in Section 3.6. Table 5.2 reports the results of each stage.'));
c.push(cap('5.2', 'Two-stage transfer learning results for both architectures.'));
c.push(T([3000, 2600, 2600], [
  ['', 'MobileNetV2', 'ResNet50'],
  ['Stage 1 best validation loss', '0.1217  (epoch 14)', '0.0390  (epoch 14)'],
  ['Stage 2 best validation loss', '0.0604', '0.0272  (epoch 18)'],
  ['Best validation accuracy', '0.9802', '0.9905'],
  ['Backbone layers unfrozen', 'top 19', 'top 21 of 30'],
  ['Training time (Tesla T4)', '84.8 minutes', '53.9 minutes'],
]));
c.push(P('In both cases the second stage improved substantially on the first. MobileNetV2 fell from a best validation loss of 0.1217 after head training to 0.0604 after fine-tuning, and ResNet50 from 0.0390 to 0.0272. This is the evidence for the two-stage design: fine-tuning the top of the backbone was not a formality but produced most of the final performance.'));
c.push(P('ResNet50 finished training in less wall-clock time than MobileNetV2, 53.9 minutes against 84.8, despite having roughly ten times as many parameters. This happened because it converged sooner and early stopping ended its second stage at epoch 11 of 15. Training cost and inference cost are separate quantities, and it is inference cost that matters for the deployment case examined in Section 5.4.'));
c.push(FIG('results/figures/mobilenetv2_curves.png and resnet50_curves.png, side by side'));
c.push(fcap('5.1', 'Training and validation curves for both models across the two stages.'));

// ---- 5.3 ----------------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('5.3  Test set performance'));
c.push(P('Both models were evaluated once on the 1,248 held-out test images. Table 5.3 reports the overall metrics.'));
c.push(cap('5.3', 'Overall test set performance (1,248 images).'));
c.push(T([3000, 2600, 2600], [
  ['Metric', 'MobileNetV2', 'ResNet50'],
  ['Accuracy', '0.9832', '0.9896'],
  ['Macro F1', '0.9781', '0.9875'],
  ['Balanced accuracy', '0.9799', '0.9896'],
  ["Cohen's kappa", '0.9751', '0.9846'],
  ['Macro AUC (one-vs-rest)', '0.9994', '0.9997'],
  ['Total errors', '21', '13'],
], { rightFrom: 1 }));
c.push(P('ResNet50 scored higher on every overall metric. It made 13 errors against MobileNetV2’s 21, a difference of eight images out of 1,248. Whether that difference is large enough to be treated as real is tested in Section 5.4.'));
c.push(P('Overall accuracy is not the most informative figure here, because the four classes are unevenly represented and they do not carry equal clinical weight. Table 5.4 gives precision, recall and F1 for each class separately.'));
c.push(cap('5.4', 'Per-class precision (P), recall (R) and F1 for both models.'));
c.push(T([2100, 1000, 1000, 1000, 1000, 1000, 1000, 900], [
  ['Class', 'MNv2 P', 'MNv2 R', 'MNv2 F1', 'RN50 P', 'RN50 R', 'RN50 F1', 'n'],
  ['Healthy Foot/Nail', '0.9982', '0.9946', '0.9964', '0.9982', '1.0000', '0.9991', '556'],
  ['Nail Fungal Infection', '0.9667', '0.9915', '0.9789', '0.9915', '1.0000', '0.9957', '117'],
  ['Foot Wound/Injury', '0.9761', '0.9787', '0.9774', '0.9919', '0.9734', '0.9826', '376'],
  ['Foot Ulcer', '0.9645', '0.9548', '0.9596', '0.9608', '0.9849', '0.9727', '199'],
], { rightFrom: 1 }));
c.push(P('Foot ulcer recall is the figure that matters most in this application, because a missed ulcer is the error with the most serious consequences. MobileNetV2 reached 0.9548 on this class, meaning it failed to identify nine of the 199 ulcers in the test set. ResNet50 reached 0.9849, missing three. That gap of six images is a larger practical difference than the 0.64 percentage points separating the two overall accuracies.'));
c.push(P('ResNet50 achieved perfect recall on both the healthy and nail fungal classes, and all 13 of its errors were confusions between foot wound and foot ulcer. Table 5.5 shows the full confusion matrices.'));
c.push(cap('5.5', 'Confusion matrices. Rows are the true class, columns the predicted class.'));
c.push(P('MobileNetV2', { b: true, after: 70, just: false }));
c.push(T([2400, 1300, 1300, 1300, 1300], [
  ['', 'healthy', 'nail_fungal', 'foot_wound', 'foot_ulcer'],
  ['healthy', '553', '3', '0', '0'],
  ['nail_fungal', '1', '116', '0', '0'],
  ['foot_wound', '0', '1', '368', '7'],
  ['foot_ulcer', '0', '0', '9', '190'],
], { rightFrom: 1 }));
c.push(P('ResNet50', { b: true, after: 70, just: false }));
c.push(T([2400, 1300, 1300, 1300, 1300], [
  ['', 'healthy', 'nail_fungal', 'foot_wound', 'foot_ulcer'],
  ['healthy', '556', '0', '0', '0'],
  ['nail_fungal', '0', '117', '0', '0'],
  ['foot_wound', '1', '1', '366', '8'],
  ['foot_ulcer', '0', '0', '3', '196'],
], { rightFrom: 1 }));
c.push(P('The dominant error type is the same for both models: confusion between foot wound and foot ulcer, accounting for 16 of MobileNetV2’s 21 errors and 11 of ResNet50’s 13. Neither model ever classified a wound or an ulcer as healthy. MobileNetV2’s three healthy-class errors were all healthy nails read as fungal infection, and all three occurred among the montage tiles rather than the photographs of whole feet.'));
c.push(FIG('results/figures/mobilenetv2_confusion_matrix.png and resnet50_confusion_matrix.png'));
c.push(fcap('5.2', 'Normalised confusion matrices for both models on the test set.'));

// ---- 5.4 significance ----------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('5.4  Statistical comparison of the two models'));
c.push(P('The two models were evaluated on the same images, so their errors are correlated: an image that is difficult for one tends to be difficult for the other. Comparing the two accuracy figures as if they came from independent samples would therefore overstate the evidence. The comparison was made image by image instead, using McNemar’s exact test on the paired outcomes.'));
c.push(P('The models disagreed on 24 of the 1,248 test images. Sixteen of those disagreements favoured ResNet50 and eight favoured MobileNetV2, giving an exact p-value of 0.1516. The difference between the two models is therefore not statistically significant at the 5% level. A paired bootstrap with 10,000 resamples put the accuracy difference at +0.0064 with a 95% confidence interval of −0.0016 to +0.0144, an interval that includes zero.'));
c.push(P('The same test was repeated within each class and within each source, because an overall figure can hide a difference confined to one class. Table 5.6 reports all of these.'));
c.push(cap('5.6', "Paired comparison using McNemar's exact test, overall and by class and source. Disagreements are shown as (favouring ResNet50 / favouring MobileNetV2)."));
c.push(T([2600, 800, 1000, 1000, 1500, 900], [
  ['Comparison', 'n', 'MNv2 err', 'RN50 err', 'Disagreements', 'p'],
  ['Overall', '1,248', '21', '13', '24  (16 / 8)', '0.1516'],
  ['Healthy Foot/Nail', '556', '3', '0', '3  (3 / 0)', '0.2500'],
  ['Nail Fungal Infection', '117', '1', '0', '1  (1 / 0)', '1.0000'],
  ['Foot Wound/Injury', '376', '8', '10', '12  (5 / 7)', '0.7744'],
  ['Foot Ulcer', '199', '9', '3', '8  (7 / 1)', '0.0703'],
  ['figshare_nail', '117', '1', '0', '1  (1 / 0)', '1.0000'],
  ['figshare_nail_tiles', '143', '3', '0', '3  (3 / 0)', '0.2500'],
  ['mendeley_foot', '789', '8', '10', '12  (5 / 7)', '0.7744'],
  ['ulcer_fuseg', '199', '9', '3', '8  (7 / 1)', '0.0703'],
], { rightFrom: 1 }));
c.push(P('No comparison in the study reaches significance at the 5% level. Two rows are worth noting separately.'));
c.push(P('The foot ulcer comparison gives p = 0.0703, the closest any comparison comes to the conventional threshold. Seven of the eight disagreements on this class favour ResNet50, so the direction is consistent even though the result does not reach significance on 199 ulcer images.'));
c.push(P('The mendeley_foot comparison gives p = 0.7744. This is the only source that contains two classes, so it is the only comparison in which recognising the dataset cannot substitute for recognising the condition. The two models disagreed on 12 of its 789 images, seven of those in MobileNetV2’s favour.'));

// ---- 5.5 efficiency ------------------------------------------------------
c.push(H2('5.5  Computational cost'));
c.push(P('Model size and inference speed were measured because the deployment case in this project is a health worker’s phone rather than a server. Latency was recorded at batch size one, as the median of 50 runs, since the usage pattern is one photograph at a time.'));
c.push(cap('5.7', 'Model size and single-image inference latency.'));
c.push(T([2600, 2200, 2200, 1200], [
  ['', 'MobileNetV2', 'ResNet50', 'Ratio'],
  ['Parameters', '2,263,108', '23,595,908', '10.4×'],
  ['Size on disk', '21.78 MB', '210.55 MB', '9.7×'],
  ['Inference, CPU', '248.29 ms', '460.56 ms', '1.9×'],
  ['Inference, GPU', '155.39 ms', '231.64 ms', '1.5×'],
], { rightFrom: 1 }));
c.push(P('MobileNetV2 is roughly a tenth of the size of ResNet50 and takes about half as long per image on a CPU. Its advantage is smaller on a GPU, 1.5 times against 1.9 times, which is the expected consequence of its architecture: depthwise separable convolutions reduce the number of operations but make less efficient use of the dense matrix hardware a GPU provides. The CPU figure is the relevant one for this project, since it is the closer proxy for a mid-range phone.'));
c.push(P('These latency measurements were taken on a shared Colab Tesla T4 at batch size one, where timings vary by a few percent between runs. The ordering of the two models was stable across every run in this study, but the third significant figure should not be treated as precise.'));

// ---- 5.6 calibration -----------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('5.6  Calibration and selective prediction'));
c.push(P('Confidence values were calibrated using temperature scaling fitted on the validation set, and the expected calibration error was measured before and after. Table 5.8 reports the results.'));
c.push(cap('5.8', 'Temperature scaling and expected calibration error (ECE).'));
c.push(T([2800, 2700, 2700], [
  ['', 'MobileNetV2', 'ResNet50'],
  ['Fitted temperature', '1.4278  (softens)', '0.9495  (sharpens)'],
  ['Validation ECE', '0.0121 → 0.0070', '0.0063 → 0.0045'],
  ['Test ECE', '0.0072 → 0.0068', '0.0039 → 0.0040'],
  ['Mean confidence', '0.9896 → 0.9821', '0.9903 → 0.9910'],
]));
c.push(P('The two models required corrections in opposite directions. MobileNetV2 was overconfident, with a mean confidence of 0.9896 against an accuracy of 0.9802, and the fitted temperature of 1.4278 softened its predictions. ResNet50 was slightly under-confident, at 0.9903 confidence against 0.9905 accuracy, and its fitted temperature of 0.9495 sharpened them instead. Both temperatures were verified not to change any prediction, only the confidence attached to it.'));
c.push(P('Neither model was badly miscalibrated to begin with, so the improvements are modest in absolute terms, roughly a 30 to 40 percent reduction in validation calibration error for each. On the test set the values barely moved in either direction.'));
c.push(P('An abstention threshold was then chosen on the validation set, set at the lowest confidence value that achieved 99 percent accuracy among the cases the model still answered. Table 5.9 reports the outcome.'));
c.push(cap('5.9', 'Selective prediction: abstention threshold, coverage and accuracy on answered cases.'));
c.push(T([2800, 2700, 2700], [
  ['', 'MobileNetV2', 'ResNet50'],
  ['Abstention threshold', '0.787', '0.000  (none required)'],
  ['Validation coverage / accuracy', '97.2%  /  0.9902', '100%  /  0.9905'],
  ['Test coverage / accuracy', '97.0%  /  0.9959', '100%  /  0.9896'],
]));
c.push(P('For MobileNetV2 the procedure produced a working threshold. Declining the least confident 3.0 percent of test cases raised its accuracy on the remainder from 0.9832 to 0.9959. In practical terms, roughly 37 of the 1,248 test images were set aside for referral, and doing so removed about 16 of the 21 errors.'));
c.push(P('ResNet50 received no threshold. It already met the 99 percent target while answering every validation case, so the selection procedure correctly returned none. Its accuracy at full coverage on the test set was 0.9896.'));
c.push(FIG('results/figures/mobilenetv2_calibration.png — reliability diagram and risk-coverage curve'));
c.push(fcap('5.3', 'Reliability diagram before and after temperature scaling, and the risk-coverage curve used to select the abstention threshold, for MobileNetV2.'));

// ---- 5.7 confound --------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('5.7  Testing for dataset shortcuts'));
c.push(P('Each class in this dataset comes largely from one source, so the classes differ not only in the condition shown but in camera, lighting, framing and background. A model could therefore score well by learning to recognise which dataset an image came from rather than what is wrong with the foot. Section 2.5.8 identified this as a weakness that most published work in this area does not address. Three independent tests were run to check for it, and all three were applied to both models.'));

c.push(H3('5.7.1  Accuracy within a single source'));
c.push(P('The Mendeley foot dataset contributes 789 test images containing both healthy feet (413) and wounds (376). These images share a clinic, a camera and a style of framing, so knowing which dataset an image came from provides no information about its class. Accuracy on this subset is therefore a direct measure of whether the models separate the conditions themselves.'));
c.push(cap('5.10', 'Accuracy within mendeley_foot, the only source containing two classes.'));
c.push(T([3000, 1900, 1600, 1600], [
  ['Model', 'Accuracy', 'Errors', 'n'],
  ['MobileNetV2', '0.9899', '8', '789'],
  ['ResNet50', '0.9873', '10', '789'],
], { rightFrom: 1 }));
c.push(P('Both models scored close to 99 percent on this subset. Both also classified all 413 healthy feet in this source correctly: every error and every disagreement between the models on these images was on a wound. As reported in Section 5.4, the difference between the two models here is not significant (p = 0.7744).'));

c.push(H3('5.7.2  Grad-CAM attention'));
c.push(P('Grad-CAM overlays were generated for test predictions to show which region of each image most influenced the classification. Maps were produced for misclassified predictions as well as correct ones, rather than only for successes.'));
c.push(P('Across the examples inspected, attention concentrated on the toes and nail plates, with the margins of the image consistently cold. This is a qualitative result and is reported as such, but it is consistent across the sample and it agrees with the quantitative tests reported above and below.'));
c.push(FIG('results/figures/mobilenetv2_gradcam.png — grid of correct and misclassified examples'));
c.push(fcap('5.4', 'Grad-CAM overlays for MobileNetV2, including both correct and misclassified predictions.'));

c.push(H3('5.7.3  Controlled occlusion'));
c.push(P('Before the occlusion test was run, the brightness of the image borders was measured by class and source, to establish whether a border shortcut was even plausible. Table 5.11 shows that it was.'));
c.push(cap('5.11', 'Mean brightness of the outer 8-pixel border, by class and source.'));
c.push(T([3600, 2300, 2300], [
  ['Class / source', 'Border brightness', 'Near-white pixels'],
  ['nail_fungal / figshare_nail', '107.6', '0.3%'],
  ['foot_ulcer / ulcer_fuseg', '112.7', '5.4%'],
  ['foot_wound / mendeley_foot', '127.2', '10.1%'],
  ['healthy / figshare_nail_tiles', '166.7', '32.9%'],
  ['healthy / mendeley_foot', '170.1', '7.0%'],
], { rightFrom: 1 }));
c.push(P('Healthy images have markedly brighter borders than the others, and the gap holds within a single source: Mendeley healthy images average 170.1 against 127.2 for Mendeley wounds. A model could in principle use this.'));
c.push(P('The occlusion test masked a 20-pixel border and measured the change in accuracy. On its own this would prove nothing, because masking any part of an image removes information. The test therefore included a control arm that masked the same number of pixels from the interior instead, so that location was the only difference between the two. A fourth arm masked the centre and left only the border visible, to measure how much could be recovered from the border alone. Table 5.12 reports all four arms.'));
c.push(cap('5.12', 'Controlled occlusion at 20-pixel border width, 1,248 test images.'));
c.push(T([3400, 2400, 2400], [
  ['Arm', 'MobileNetV2', 'ResNet50'],
  ['Unmodified', '0.9832', '0.9896'],
  ['Border masked', '0.9671  (−0.0160)', '0.9704  (−0.0192)'],
  ['Interior control, equal area', '0.8598  (−0.1234)', '0.9030  (−0.0865)'],
  ['Border only (centre masked)', '0.4952', '0.5994'],
  ['Majority-class floor', '0.4455', '0.4455'],
], { rightFrom: 1 }));
c.push(P('Masking the border cost MobileNetV2 1.6 accuracy points. Masking the same number of pixels from the interior cost 12.3 points, about eight times more. For ResNet50 the equivalent figures were 1.9 and 8.7 points, a ratio of about four and a half. In both cases the border was worth considerably less than a comparable region elsewhere in the image.'));
c.push(P('The fourth arm separated the two models. Table 5.13 gives per-class recall when only the border was visible.'));
c.push(cap('5.13', 'Per-class recall with the interior masked and only a 20-pixel border visible.'));
c.push(T([2800, 2700, 2700], [
  ['Class', 'MobileNetV2', 'ResNet50'],
  ['Healthy Foot/Nail', '1.0000  (556/556)', '0.9982  (555/556)'],
  ['Nail Fungal Infection', '0.0000  (0/117)', '0.0000  (0/117)'],
  ['Foot Wound/Injury', '0.1516  (57/376)', '0.3511  (132/376)'],
  ['Foot Ulcer', '0.0251  (5/199)', '0.3065  (61/199)'],
  ['Non-healthy images recovered', '62 of 692', '193 of 692'],
]));
c.push(P('With the foot itself hidden, MobileNetV2 collapsed into predicting the majority class: it recovered 62 of the 692 non-healthy images, and its overall accuracy of 0.4952 sits only five points above the 0.4455 obtained by always guessing healthy. ResNet50 recovered 193 of the same 692, roughly three times as many, reaching 0.5994 against the same floor.'));

// ---- 5.8 triage ----------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('5.8  Referral policy'));
c.push(P('The referral policy described in Section 3.12 escalates a case when a serious condition holds meaningful probability, even when it is not the single most likely class. Its behaviour was measured on the test set, and its escalation threshold was tuned on the validation set.'));
c.push(H3('5.8.1  Threshold selection on validation'));
c.push(P('The threshold controlling escalation to the urgent band was swept across the validation set, which contained 200 ulcers of which 11 were misclassified, and 685 images of conditions requiring no urgent care. Table 5.14 reports the sweep.'));
c.push(cap('5.14', 'Validation sweep of the urgent escalation threshold (1,261 validation images).'));
c.push(T([2000, 3200, 2800], [
  ['Threshold', 'Misclassified ulcers reaching urgent', 'Benign cases sent urgently'],
  ['0.02', '10 of 11   (90.9%)', '2 of 685   (0.3%)'],
  ['0.04', '8 of 11   (72.7%)', '1 of 685   (0.1%)'],
  ['0.06', '7 of 11   (63.6%)', '1 of 685   (0.1%)'],
  ['0.08', '6 of 11   (54.5%)', '1 of 685   (0.1%)'],
  ['0.10  (adopted)', '6 of 11   (54.5%)', '0 of 685   (0.0%)'],
  ['0.14', '4 of 11   (36.4%)', '0 of 685   (0.0%)'],
  ['0.18', '3 of 11   (27.3%)', '0 of 685   (0.0%)'],
  ['0.20  (original)', '3 of 11   (27.3%)', '0 of 685   (0.0%)'],
  ['0.24', '2 of 11   (18.2%)', '0 of 685   (0.0%)'],
  ['0.32', '1 of 11   (9.1%)', '0 of 685   (0.0%)'],
  ['0.38 and above', '0 of 11   (0.0%)', '0 of 685   (0.0%)'],
]));
c.push(P('The threshold originally set by judgement, 0.20, escalated three of the 11 misclassified validation ulcers. Lowering it to 0.10 escalated six while still sending no benign case for urgent care. The cost budget defined for the sweep, which allowed up to 2 percent of benign cases to be escalated, was never reached at any threshold on the grid: the most aggressive value tested used 2 of the 13.7 cases the budget permitted. The threshold was set to 0.10 for the results reported below; the reasoning behind not taking the lowest value on the grid is given in Chapter 6.'));

c.push(H3('5.8.2  Behaviour on the test set'));
c.push(P('The policy was then audited on the test set at both thresholds, so that the effect of the change could be measured rather than assumed. Table 5.15 reports both.'));
c.push(cap('5.15', 'Referral policy audited on the test set at both escalation thresholds.'));
c.push(T([3600, 2300, 2300], [
  ['Measure (1,248 test images)', 'at 0.20', 'at 0.10  (adopted)'],
  ['Escalation fired', '26   (2.1%)', '32   (2.6%)'],
  ['Misclassified ulcers referred', '9 of 9', '9 of 9'],
  ['Misclassified ulcers reaching urgent', '7 of 9', '7 of 9'],
  ['Healthy cases referred', '0 of 556', '0 of 556'],
  ['Nail fungal cases referred', '0 of 117', '0 of 117'],
  ['True wounds raised to urgent', '14 of 376', '27 of 376'],
  ['Referral sensitivity', '574 / 575', '574 / 575'],
  ['Referral specificity', '673 / 673', '673 / 673'],
], { rightFrom: 1 }));
c.push(P('At the adopted threshold, escalation fired on 32 of the 1,248 test cases, or 2.6 percent, with a 95 percent confidence interval of 1.8 to 3.6 percent. All nine of the ulcers MobileNetV2 misclassified were still routed to care, and seven of the nine reached the urgent band. Without escalation none of the nine would have reached that band, because all nine were classified as wounds and the wound class maps to the slower prompt band.'));
c.push(P('The two ulcers that did not reach the urgent band were classified as wounds with confidences of 99.3 and 99.9 percent, leaving almost no probability on the ulcer class for any threshold to act on. They were still referred, but at the slower timeframe.'));
c.push(P('No healthy image and no nail fungal image was referred for urgent or prompt care at either threshold. The measurable cost of the policy fell entirely on true wounds, of which 27 were raised from the prompt band to the urgent band at the adopted threshold, against 14 at the original one.'));
c.push(P('Referral sensitivity and specificity were identical whether cases were referred on the predicted class alone or on the triage band: 574 of 575 and 673 of 673 in both cases. Lowering the threshold from 0.20 to 0.10 changed neither the number of misclassified ulcers reaching the urgent band nor any other measure in the table except the count of true wounds escalated.'));

// ---- 5.9 external --------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('5.9  External validation on photographs taken outside the source datasets'));
c.push(P('All three source datasets are clinical collections, photographed by people whose job is to photograph feet. Everything reported so far therefore measures performance on that kind of image. To test how far those results carry to a photograph taken the way this system is meant to be used, a small set of images was captured on a mobile phone and scored through the prototype’s batch evaluation tab.'));
c.push(H3('5.9.1  What was collected'));
c.push(P('Five individuals were photographed: the author and four friends. Each was photographed twice, once against a plain background and once against a patterned background, giving five matched pairs and ten images in total. The pairing is the point of the design. With a set this small, comparing two groups of five images as independent samples would be uninformative, whereas photographing the same feet under both conditions allows the background to be tested as a paired comparison, which is a stronger test on the same number of images.'));
c.push(P('One correction should be recorded, because it changed a result rather than a label. The two photographs of subject 2 were uploaded with their background names transposed, and the figures reported below are the corrected ones. Overall accuracy does not depend on those names and is unaffected, but the comparison between backgrounds does: it moves from four out of five against two out of five to five out of five against one out of five. The mistake was detectable only because the batch tab records the filename of every image, which is the argument for it doing so.'));
c.push(P('All five individuals had healthy feet, with no visible wound, ulcer or nail disease, so the true label for all ten images is Healthy Foot/Nail. Images were scored using MobileNetV2 with temperature scaling and the abstention threshold of 0.787 applied, which is the deployment configuration recommended in Section 5.6 rather than a more permissive one.'));
c.push(P('Five different handsets were used across the five subjects: a Google Pixel, a Nothing Phone, a Samsung, an iPhone 14 and an iPhone 16 Pro. All photographs were taken indoors under ordinary tube lighting. The set therefore spans five cameras rather than one, which widens the range of capture conditions it represents, and it also means the camera varies between subjects rather than being held constant. That does not weaken the background comparison, which is made within a subject and therefore within one handset.'));
c.push(cap('5.16', 'Every photograph in the external validation set, scored with MobileNetV2 (calibrated).'));
c.push(T([1800, 1600, 2000, 1300, 1600, 1700], [
  ['Image', 'True class', 'Predicted', 'Correct', 'Confidence', 'At 0.787'],
  ['s1_plain', 'Healthy', 'Healthy', 'yes', '0.9988', 'answered'],
  ['s1_patterned', 'Healthy', 'Foot Wound', 'no', '0.9219', 'answered'],
  ['s2_plain', 'Healthy', 'Healthy', 'yes', '0.9058', 'answered'],
  ['s2_patterned', 'Healthy', 'Foot Wound', 'no', '0.6376', 'abstained'],
  ['s3_plain', 'Healthy', 'Healthy', 'yes', '0.7917', 'answered'],
  ['s3_patterned', 'Healthy', 'Foot Wound', 'no', '0.7824', 'abstained'],
  ['s4_plain', 'Healthy', 'Healthy', 'yes', '0.8595', 'answered'],
  ['s4_patterned', 'Healthy', 'Healthy', 'yes', '0.5311', 'abstained'],
  ['s5_plain', 'Healthy', 'Healthy', 'yes', '0.7528', 'abstained'],
  ['s5_patterned', 'Healthy', 'Foot Wound', 'no', '0.6074', 'abstained'],
], { rightFrom: 3 }));
c.push(H3('5.9.2  Accuracy'));
c.push(P('Six of the ten photographs were classified correctly. Table 5.17 reports the figure overall and by background, with a Wilson confidence interval for each, because on ten images the interval is the honest number and the percentage on its own is not.'));
c.push(cap('5.17', 'External validation accuracy, overall and by background, with 95 percent Wilson intervals.'));
c.push(T([2600, 1600, 1700, 2600], [
  ['Set', 'Correct', 'Accuracy', '95% confidence interval'],
  ['All photographs', '6 of 10', '60.0%', '31.3%  to  83.2%'],
  ['Plain background', '5 of 5', '100.0%', '56.6%  to  100.0%'],
  ['Patterned background', '1 of 5', '20.0%', '3.6%  to  62.5%'],
  ['Test set, for comparison', '1,227 of 1,248', '98.32%', '97.4%  to  98.9%'],
], { rightFrom: 1 }));
c.push(P('Accuracy on these photographs was 60 percent, against 98.32 percent on the test set: a difference of 38.3 percentage points. The confidence interval on ten images runs from 31.3 to 83.2 percent, which is wide, and the claim this measurement supports is therefore directional rather than precise. What it does support is clear enough, because 98.32 percent lies well outside that interval: accuracy on phone photographs taken outside the source datasets is substantially lower than the test set figure. What it does not support is any particular value for the size of the drop.'));
c.push(P('Two averaged measures that appear elsewhere in this chapter should not be quoted for this set. Macro-averaged F1 computes to 0.188 and macro-averaged recall to 0.150, because three of the four classes have no images here at all and the average divides performance on one class by four. The meaningful figures are recall on the healthy class, which is 6 of 10, and the false positive rate, which is 4 of 10.'));
c.push(H3('5.9.3  Every error is the same error'));
c.push(P('All four misclassifications were the same one: a healthy foot classified as Foot Wound/Injury. No healthy foot was classified as an ulcer, and no image was classified as a nail condition. The errors therefore all run in the direction that the referral policy was designed to tolerate, which is the safer direction: a healthy person is sent to a clinician rather than a person with a wound being told nothing is wrong.'));
c.push(P('That should not be read as evidence that the system is safe on phone photographs, for one reason that has to be stated plainly. All five subjects had healthy feet, so this set contains no disease, and a set with no disease in it cannot measure a missed diagnosis. The failure that matters clinically, a real ulcer photographed on a phone and reported as healthy, is untested here and cannot be tested with these images. Section 7.3 lists photographs of genuine disease as the single most valuable extension to this work.'));
c.push(H3('5.9.4  The effect of the background'));
c.push(P('Every photograph taken against a plain background was classified correctly, and four of the five taken against a patterned background were wrong. Because each subject appears under both conditions, the appropriate test is McNemar’s exact test on the five pairs rather than a comparison of the two proportions. Four of the five pairs disagree, and all four disagree in favour of the plain background, which gives an exact two-sided p-value of 0.1250.'));
c.push(P('The difference is therefore not statistically significant, and it is worth being precise about why: with five pairs the smallest two-sided p-value attainable is 0.0625, which requires all five pairs to disagree in the same direction, and four out of four gives 0.1250. The result is as one-directional as this many photographs allow and the test still cannot clear the conventional threshold, so what limits it is the size of the set rather than the strength of the effect. The observation is reported as an observation with a plausible mechanism, not as a null result. A patterned background places edge and texture structure in the region around the foot, which is the kind of incidental detail that Section 5.7.3 showed the models can read to some degree. Establishing the effect would need roughly twenty to thirty pairs.'));
c.push(P('This design was chosen because of an earlier single observation made while the prototype was being tested, which is recorded here because it is what prompted the paired collection. The same pair of healthy feet, photographed twice within a few minutes, was classified as a foot wound with 98.6 percent confidence against a patterned red cloth at wide framing, and as healthy with 89.2 percent confidence against a plain wall at close framing. The Grad-CAM overlay for the first photograph showed attention extending beyond the feet into the surrounding fabric, while for the second it was concentrated on the toes. Two conditions differed between those two photographs, the background and the framing, so that pair could not separate them; the collection described above holds framing roughly constant and varies the background alone.'));
c.push(FIG('One subject photographed against both backgrounds, with the screening report and Grad-CAM overlay for each'));
c.push(fcap('5.5', 'The same feet against a plain and a patterned background, with the report each produced.'));
c.push(H3('5.9.5  What abstention did under these conditions'));
c.push(P('The abstention threshold behaved very differently here than on the test set, and this is the most useful result in the section.'));
c.push(cap('5.18', 'Selective prediction on the external photographs, against the same mechanism on the test set.'));
c.push(T([3000, 2100, 2100], [
  ['Measure', 'External photographs', 'Test set'],
  ['Coverage at threshold 0.787', '5 of 10   (50.0%)', '1,211 of 1,248   (97.0%)'],
  ['Accuracy on answered cases', '4 of 5   (80.0%)', '99.59%'],
  ['Errors declined', '3 of 4', '16 of 21'],
  ['Mean confidence', '0.7789', '0.9824'],
], { rightFrom: 1 }));
c.push(P('The threshold declined three of the four errors, which raised accuracy from 60 percent to 4 of 5 on the cases the model still answered, at the cost of declining half of them. On the test set the same threshold declined 3 percent of images; here it declined 50 percent. That collapse in coverage is the mechanism working rather than failing. When the input is unlike the data the model was fitted on, the probability vector flattens, confidence falls, and the system stops answering instead of guessing, which is precisely the behaviour a referral threshold exists to produce.'));
c.push(P('Mean confidence across these ten photographs was 0.7789, against 0.9824 on the test set, so the model was measurably less certain on phone photographs even where it was right. Within this set, confidence was lower on the four errors, at 0.7373 on average, than on the six correct predictions, at 0.8066, which is the ordering a calibrated model should show and the reason abstention was able to catch most of the errors.'));
c.push(P('The threshold also responded to the background. It answered four of the five plain photographs and one of the five patterned ones, which runs in the same direction as the accuracy difference in Section 5.9.4 and is a second piece of evidence that what the mechanism detects is a change in capture conditions.'));
c.push(P('One error survived the threshold. Image s1_patterned was classified as a foot wound with 92.19 percent confidence, which is above any threshold that would leave the system usable. Abstention therefore reduces the error rate under these conditions but does not eliminate it, and a threshold set high enough to catch that case would decline almost every image.'));
c.push(H3('5.9.6  What the referral policy did with these ten'));
c.push(P('Four images were classified as wounds, which maps to the prompt band, and two of the correct healthy predictions fell below the abstention threshold, which the policy raises to the routine band because an inconclusive screening is not a negative result. At least six of these ten healthy people are therefore routed to a clinician, and at most four are told that no action is needed on this result. These are bounds rather than exact counts, because escalation on wound and ulcer probability can only raise a band and never lower one.'));
c.push(P('The comparison worth recording is with the test set, where no healthy image at all was referred for urgent or prompt care. The cost of the referral policy, which measured as nothing on data from the source datasets, is substantial on photographs taken outside them. Chapter 6 takes up what that means for deployment.'));

// ---- 5.10 out-of-distribution -------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('5.10  Behaviour on a condition outside the four classes'));
c.push(P('Section 5.9 changes the conditions under which a photograph is taken while keeping the condition within the four classes. The opposite case also needs measuring: what the system does with a real condition it has no class for. Every image it receives is assigned to one of four categories, so this is not a hypothetical question.'));
c.push(P('The Figshare download contains 578 images of nail dystrophy, a genuine nail condition that was excluded during preprocessing because the four classes were fixed by the project proposal. The models have never seen these images in training, validation or test, which makes them a clean test on real medical photographs rather than on artificial inputs. They were scored with MobileNetV2, calibrated, using the same configuration as Section 5.9.'));
c.push(H3('5.10.1  Confidence gives no warning'));
c.push(cap('5.19', 'Confidence on 578 images of a condition with no class, against the 1,248 test images.'));
c.push(T([2800, 2200, 2200], [
  ['Measure', 'Nail dystrophy  (578)', 'Test set  (1,248)'],
  ['Mean confidence', '0.9807', '0.9824'],
  ['Median confidence', '0.9997', '0.9998'],
  ['Declined at threshold 0.787', '2.6%', '3.0%'],
], { rightFrom: 1 }));
c.push(P('Confidence on a condition the model cannot represent is indistinguishable from confidence on the data it was trained for: the means differ by 0.0017 and the medians by 0.0001. Slightly fewer of the unknown images were declined than of the known ones. Confidence therefore carries essentially no information about whether an input is something the model can represent, and no threshold placed on confidence can separate the two cases.'));
c.push(P('Read alongside Section 5.9.5 this is a more precise result than either measurement gives alone. Confidence fell sharply on unfamiliar photographs of familiar conditions and did not move at all on an unfamiliar condition. The two shifts are different and the mechanism responds to only one of them; Section 6.4 sets out why.'));
c.push(H3('5.10.2  What the system told those 578 people'));
c.push(cap('5.20', 'Labels and referral bands assigned to 578 images of nail dystrophy.'));
c.push(T([3000, 1900, 2200], [
  ['Outcome', 'Count', 'Share'],
  ['Labelled Nail Fungal Infection', '565', '97.8%'],
  ['Labelled Healthy Foot/Nail', '13', '2.2%'],
  ['Labelled Foot Wound/Injury', '0', '0.0%'],
  ['Labelled Foot Ulcer', '0', '0.0%'],
  ['Routed to a clinician', '569', '98.4%'],
  ['Told no action was needed', '9', '1.6%'],
], { rightFrom: 1 }));
c.push(P('Two results follow. Every one of the 578 images was assigned to a nail class and none to a foot class, so the model reliably identifies which part of the body it is looking at; what it cannot do is place a condition outside its vocabulary. And the label it assigns is the nearest available one, onychomycosis, which shares the discoloration and thickening that characterise dystrophy.'));
c.push(P('Because the nail fungal class maps to the routine referral band, 569 of the 578, or 98.4 percent, were routed to a clinician. They receive the wrong name for their condition and the right action, and the clinician they are sent to would arrive at the correct name. Nine people out of 578 were told that nothing was wrong when they had a real nail condition. Abstention earned a small amount here: of the 13 images labelled healthy, four fell below the threshold and were raised to a routine referral.'));
c.push(P('The outcome is therefore mostly benign, but Chapter 6 argues that this is an accident of the class structure rather than a safeguard the design earned, because the nearest available label happened to carry a referral.'));

// ---- 5.11 summary --------------------------------------------------------
c.push(H2('5.11  Summary'));
c.push(P('MobileNetV2 reached 98.32 percent accuracy on the test set and ResNet50 reached 98.96 percent. The difference between them is not statistically significant (p = 0.1516), and a paired bootstrap bounds it at 1.4 accuracy points at 95 percent confidence. On the one comparison in which dataset provenance cannot help either model, the two are indistinguishable (p = 0.7744). ResNet50 missed three of the 199 test ulcers against MobileNetV2’s nine, the largest effect observed in the study, although this too falls short of significance (p = 0.0703).'));
c.push(P('MobileNetV2 achieved its performance with a tenth of the parameters, 21.78 MB against 210.55 MB on disk, and roughly half the CPU inference time. It supported a working abstention threshold that raised accuracy on answered cases to 99.59 percent while declining 3 percent of them; ResNet50 required no threshold.'));
c.push(P('All three tests for dataset shortcuts returned results consistent with the models classifying the condition rather than the source. Both scored close to 99 percent within the single source containing two classes, Grad-CAM attention fell on the toes and nail plates, and masking the image border cost far less accuracy than masking an equal area of the interior. The border-only arm distinguished the two models, with ResNet50 recovering about three times as much class information from the border alone.'));
c.push(P('The referral policy escalated 2.6 percent of test cases and brought seven of the nine misclassified ulcers into the urgent band, none of which would have reached it otherwise, while referring no healthy or nail fungal case for urgent or prompt care.'));
c.push(P('Two measurements test the boundaries of those figures. On ten photographs taken on a phone outside all three source datasets, accuracy was 6 of 10, with a confidence interval from 31.3 to 83.2 percent; all four errors were healthy feet read as wounds, and the abstention threshold declined three of them, raising accuracy on answered cases to 4 of 5 while declining half the set. On 578 images of a nail condition with no class in the system, confidence was indistinguishable from confidence on the test set and only 2.6 percent were declined, although 98.4 percent were still routed to a clinician because the nearest available label carries a routine referral.'));
c.push(P('Taken together those two results locate the abstention mechanism precisely: it responds to an unfamiliar photograph of a familiar condition and not at all to a condition it has no class for.'));
c.push(P('Chapter 6 interprets these results, relates them to the research objectives and to the literature reviewed in Chapter 2, and states the limitations that qualify them.'));

const doc = new Document({
  styles: { default: {
    heading2: { run: { size: 28, bold: true, color: ACCENT, font: 'Calibri' },
                paragraph: { spacing: { before: 340, after: 160 } } },
    heading3: { run: { size: 24, bold: true, color: '2F5496', font: 'Calibri' },
                paragraph: { spacing: { before: 260, after: 120 } } },
  } },
  sections: [{ properties: { page: { margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } }, children: c }],
});
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(require('path').join(__dirname, 'Chapter5_Results_DRAFT.docx'), b);
  console.log('written', b.length, 'bytes');
});
