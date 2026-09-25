const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
        Table, TableRow, TableCell, WidthType, ShadingType, PageBreak } = require('docx');
const fs = require('fs');
const ACCENT = '1F4E79', GREY = '595959';

const P = (t, o = {}) => new Paragraph({ spacing: { after: o.after ?? 160, line: 300 },
  alignment: o.left ? undefined : AlignmentType.JUSTIFIED,
  children: [new TextRun({ text: t, size: 22, italics: o.i, bold: o.b, font: 'Calibri' })] });
const TODO = t => new Paragraph({ spacing: { before: 120, after: 160, line: 300 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4E5' },
  children: [ new TextRun({ text: 'CHECK:  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
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
const cell = (t, w, o = {}) => new TableCell({ width: { size: w, type: WidthType.DXA },
  shading: o.head ? { type: ShadingType.CLEAR, fill: 'E8EEF4' } : undefined,
  margins: { top: 55, bottom: 55, left: 100, right: 100 },
  children: [new Paragraph({ spacing: { after: 0 }, alignment: o.right ? AlignmentType.RIGHT : undefined,
    children: [new TextRun({ text: t, bold: o.head, size: 18, font: 'Calibri' })] })] });
const T = (widths, rows, o = {}) => new Table({ columnWidths: widths,
  width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  rows: rows.map((r, i) => new TableRow({ tableHeader: i === 0,
    children: r.map((cc, j) => cell(String(cc), widths[j],
      { head: i === 0, right: o.rightFrom !== undefined && j >= o.rightFrom && i > 0 })) })) });

const c = [];
c.push(new Paragraph({ spacing: { after: 280 },
  children: [new TextRun({ text: 'Chapter 3', bold: true, size: 36, color: ACCENT, font: 'Calibri' })] }));
c.push(new Paragraph({ spacing: { after: 360 },
  children: [new TextRun({ text: 'Methodology', bold: true, size: 36, color: ACCENT, font: 'Calibri' })] }));

// ---- 3.1 -----------------------------------------------------------------
c.push(H2('3.1  Research design'));
c.push(P('This study is experimental and quantitative. Two convolutional neural networks were trained to classify photographs of feet and nails into four categories, and their performance was compared on data neither had seen. The comparison is the point of the design. The research question is not whether deep learning can classify these conditions, which the literature reviewed in Chapter 2 already establishes, but what is lost by choosing a model small enough to run on a phone. That question only has an answer if the two models are trained under conditions that are identical in every respect except the architecture itself.'));
c.push(P('Everything in this chapter follows from that requirement. Both models share the same data, the same split, the same preprocessing, the same augmentation, the same class weights, the same learning rates and the same training procedure. The only planned differences are the backbone network and the input normalisation each backbone expects, which is a property of the pretrained weights rather than a choice.'));
c.push(P('A second design decision shapes the rest of the chapter. Because the four classes come largely from different sources, a model could score well by learning to recognise the dataset rather than the disease. Chapter 2 identified this as a weakness that most published work in the area does not address. Rather than assume it away, three separate tests were built into the evaluation to measure it, and these are described in Section 3.9.'));
c.push(P('The chapter continues with the datasets and how they were prepared, the splitting strategy, preprocessing and augmentation, the model architectures and training procedure, the handling of class imbalance, the evaluation metrics, the design of the confound tests, calibration and selective prediction, the referral policy, the statistical tests, the tools used, and the ethical position of the work.'));

// ---- 3.2 -----------------------------------------------------------------
c.push(H2('3.2  Datasets'));
c.push(P('Three publicly available image collections were used. No new clinical data was gathered, and no patient-identifiable information was handled at any point. Table 3.1 lists the sources and what each contributes.'));
c.push(cap('3.1', 'Source datasets and their contribution to each class.'));
c.push(T([3000, 2200, 2000, 1000], [
  ['Source', 'Folder used', 'Class', 'Images'],
  ['Mendeley Data hsj38fwnvr v3', 'Normal/', 'Healthy Foot/Nail', '2,757'],
  ['Mendeley Data hsj38fwnvr v3', 'wound_main/', 'Foot Wound/Injury', '2,508'],
  ['FUSeg / AZH Chronic Wound', 'images/', 'Foot Ulcer', '1,330'],
  ['Figshare 5398573', 'B1, B2, C, D', 'Nail Fungal Infection', '779'],
  ['Figshare 5398573', 'normalnail sheets', 'Healthy Foot/Nail', '1,000'],
  ['Total', '', '', '8,374'],
], { rightFrom: 3 }));
c.push(P('The foot ulcer images come from the dataset published by Wang et al. (2020) alongside their work on automatic wound segmentation. That dataset is distributed with segmentation masks as well as photographs. The masks were deliberately excluded and stored outside the training directory, because a binary mask counted as a training image is a labelled example of nothing and would corrupt the class it was filed under.'));
c.push(P('The mapping from source folders to the four classes was decided by a person after inspecting a report of what each folder actually contained, rather than inferred from folder names. Folder names in public datasets are frequently inconsistent or misleading, and an automatic mapping would have silently mislabelled several thousand images.'));
c.push(P('One category of image was excluded on clinical rather than technical grounds. The Figshare collection contains 578 images of nail dystrophy, which is a genuine condition but not one of the four classes fixed by the project proposal. Including them would have required either a fifth class, which the proposal does not allow, or filing them under an incorrect label.'));

// ---- 3.3 -----------------------------------------------------------------
c.push(H2('3.3  Data preparation'));
c.push(P('The raw files were reduced to the final dataset through a sequence of steps, each of which is described below and each of which is verified rather than assumed.'));
c.push(H3('3.3.1  Deduplication'));
c.push(P('Duplicate images were identified by hashing file contents and removed, leaving one copy of each. This removed 1,758 images. The step matters more than it appears: without it, identical images would have appeared in both the training and the test sets, and the model would have been tested partly on pictures it had already memorised. Reported accuracy would have been inflated by an unknown amount, and the inflation would have been invisible in every metric.'));
c.push(H3('3.3.2  Tiling the montage sheets'));
c.push(P('The Figshare collection stores its healthy nail images as montage contact sheets, each a grid of many small photographs in a single file. These 277 sheets could not be used directly, because a single file containing forty nails is not a single training example. A two-stage grid detection routine was written to cut them into individual tiles: it first looks for uniform separator bands between cells, and falls back to detecting periodicity in the edge profile of the image when no clean separators exist. Tiling produced 18,096 individual nail images.'));
c.push(H3('3.3.3  Capping the montage contribution'));
c.push(P('Those 18,096 tiles came from roughly twenty photographic sessions, so they represent far fewer independent observations than their count suggests. Left unchecked they would also have made healthy nails the overwhelming majority class. The contribution was therefore capped at 1,000 images, sampled evenly across all twenty source sheets so that no single session dominated. Table 3.2 shows the full pipeline with counts.'));
c.push(cap('3.2', 'Data preparation pipeline.'));
c.push(T([4400, 3800], [
  ['Stage', 'Result'],
  ['Files scanned', '26,544'],
  ['Mapped to one of the four classes', '25,689  (855 excluded)'],
  ['Excluded: nail dystrophy', '578'],
  ['Excluded: montage contact sheets', '277  (tiled separately)'],
  ['Duplicates removed', '1,758'],
  ['Montage tiles capped', '18,096 → 1,000'],
  ['Final dataset', '8,374'],
  ['Class imbalance ratio', '4.8 : 1'],
]));

// ---- 3.4 -----------------------------------------------------------------
c.push(H2('3.4  Splitting the data'));
c.push(P('The dataset was divided into training, validation and test sets in the proportions 0.70, 0.15 and 0.15, using a fixed random seed of 42 so that the split can be reproduced exactly. This produced 5,865 training images, 1,261 validation images and 1,248 test images.'));
c.push(P('The split is stratified by class and by source together, not by class alone. This is a correction to an earlier design and the reason is worth stating, because the failure it caused was silent. Stratifying on class alone kept the proportion of each class the same in every split, which appeared correct. But the healthy class draws from two very different sources, photographs of whole feet and montage tiles of nails, and the tiles are grouped by source sheet so that no sheet can straddle the boundary. With only class-level stratification, all twenty tile groups landed in the training set. The test set therefore contained no healthy nails at all, and healthy-nail performance was not being measured. Every metric looked reasonable and none of them revealed the problem.'));
c.push(P('Stratifying by class and source together fixes this, and the preprocessing script now refuses to complete if any source is missing from any split or if any group appears on both sides of a boundary. Table 3.3 gives the composition of the test set.'));
c.push(cap('3.3', 'Composition of the test set by class and by source.'));
c.push(T([2400, 1200, 2400, 1200], [
  ['Class', 'Images', 'Source', 'Images'],
  ['Healthy Foot/Nail', '556', 'mendeley_foot', '789'],
  ['Foot Wound/Injury', '376', 'ulcer_fuseg', '199'],
  ['Foot Ulcer', '199', 'figshare_nail_tiles', '143'],
  ['Nail Fungal Infection', '117', 'figshare_nail', '117'],
], { rightFrom: 1 }));
c.push(P('The mendeley_foot source is the only one contributing two classes, 413 healthy feet and 376 wounds. This makes it the one part of the test set in which knowing the source of an image tells a model nothing about its class, and Section 3.9 explains how that property is used.'));
c.push(P('One rule governed the use of these splits throughout. The test set was evaluated once, at the very end. Every decision taken during development, including when to stop training, which checkpoint to keep, what calibration temperature to apply and where to set the referral thresholds, was made using the validation set only.'));

// ---- 3.5 -----------------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('3.5  Preprocessing'));
c.push(P('All images were converted to RGB and resized to 224 by 224 pixels, the input size both pretrained backbones expect. The resize preserves the original aspect ratio and then takes a centre crop, rather than stretching the image to fit. A simple stretch would distort the 153 images in the dataset with extreme aspect ratios, changing the apparent shape of toes and nails, which is part of what the model needs to read. Padding to a square was rejected for a different reason: it would have reintroduced the uniform black borders that were removed from the ulcer images, as described below.'));
c.push(P('The ulcer photographs from the FUSeg and Medetec collections arrived with large regions of pure black padding, between 25 and 32 percent of each image. No other class carried this marking, so it was a perfect predictor of the ulcer class and entirely unrelated to the condition. The padding was cropped away before training, and the proportion of pure black pixels was verified to have fallen to zero.'));
c.push(P('Image normalisation is applied inside each model rather than baked into the saved files. This is a deliberate choice rather than an implementation detail. MobileNetV2 was pretrained on inputs scaled to the range minus one to one, while ResNet50 expects images in the caffe convention with per-channel mean subtraction. Writing either scheme into the processed image files would have handicapped the other architecture and made the comparison meaningless. Storing images in their natural form and applying the correct transformation as the first operation inside each model keeps both backbones on the terms they were trained under.'));

c.push(H2('3.6  Augmentation'));
c.push(P('Augmentation is applied to the training set only, as a set of preprocessing layers inside the model graph. Table 3.4 lists the transformations and their ranges. Each was chosen to reflect variation that genuinely occurs in photographs of feet, rather than to inflate the dataset arbitrarily.'));
c.push(cap('3.4', 'Augmentation applied during training.'));
c.push(T([2600, 1700, 4000], [
  ['Transformation', 'Range', 'Reason'],
  ['Horizontal flip', 'enabled', 'Left and right feet are mirror images, so this is a real symmetry. Vertical flipping is not, since an upside-down clinical photograph is unrealistic.'],
  ['Rotation', '±0.055 (~20°)', 'Photographs are taken at uncontrolled angles.'],
  ['Translation', '±0.15', 'The foot is not always centred in the frame.'],
  ['Zoom', '±0.15', 'Distance from the subject varies between photographs.'],
  ['Brightness', '±0.15', 'Lighting differs between clinics; kept small so that diagnostic colour is preserved.'],
  ['Contrast', '±0.15', 'As above.'],
  ['Blur', '0.25', 'Removes image sharpness as a class cue. See below.'],
]));
c.push(P('The blur augmentation deserves separate explanation because it addresses a specific weakness in this dataset rather than a general one. The healthy nail images are montage tiles of roughly 102 pixels, upscaled to 224, while the onychomycosis images are downscaled from several hundred pixels. Sharpness therefore correlates with class through the way the images were produced rather than through anything about the conditions. Without intervention a model could learn to separate the two nail classes on blur alone. Randomising the blur applied during training removes it as a reliable signal.'));

// ---- 3.7 -----------------------------------------------------------------
c.push(H2('3.7  Model architectures and transfer learning'));
c.push(P('Two architectures were compared. MobileNetV2 is the primary model and the deployment candidate, chosen for its small size and its use of depthwise separable convolutions, which reduce the number of operations required per image. ResNet50 is the comparison model and serves as an accuracy reference: it is a substantially larger and more conventional residual network, and the question the study asks is what is given up by preferring the smaller one.'));
c.push(P('Both were initialised with weights pretrained on ImageNet. Training from scratch was not considered: the dataset here contains 8,374 images, which is several orders of magnitude smaller than what would be needed to learn general visual features from nothing, and Chapter 2 establishes transfer learning as the standard approach in this domain for exactly this reason.'));
c.push(P('The classification head is identical for both models and consists of global average pooling, a dropout layer at rate 0.3, and a dense layer of four units with softmax activation and L2 regularisation at 1e-4.'));
c.push(H3('3.7.1  Two-stage training'));
c.push(P('Training proceeds in two stages. In the first, the pretrained backbone is frozen and only the new classification head is trained, for 15 epochs at a learning rate of 1e-3. In the second, the top thirty layers of the backbone are unfrozen and training continues at 1e-5, a hundred times lower.'));
c.push(P('The reason for separating these stages is specific. At the start of training the classification head is randomly initialised, so its outputs are meaningless and the gradients flowing back from them are large and uninformative. If the backbone were unfrozen at that moment, those gradients would pass straight through the pretrained filters and destroy the very features the transfer was meant to preserve. Freezing the backbone until the head has learned something sensible prevents this. The much lower learning rate in the second stage serves the same purpose in a gentler form, allowing the general ImageNet features to adapt to skin and nail texture without being overwritten.'));
c.push(P('The second stage was allowed up to 25 epochs. In practice the two models used that allowance differently, and the difference is worth recording because it is not symmetric. ResNet50 stopped after 11 fine-tuning epochs, triggered by the early stopping rule: its best validation loss occurred at epoch 18 of training overall, and training ended exactly eight epochs later, which is the configured patience. Its result is therefore determined by convergence rather than by the epoch allowance, and would have been identical under any allowance above eleven. MobileNetV2 ran the full 25 fine-tuning epochs without early stopping. Its best validation loss occurred at epoch 34 overall, six epochs before training ended, so it was clearly plateauing but had not met the convergence criterion when the allowance ran out.'));
c.push(P('This asymmetry is acknowledged rather than glossed over. ResNet50 trained to convergence by the project\u2019s own criterion; MobileNetV2 stopped because it reached its epoch limit. Any residual effect therefore runs against MobileNetV2, the model this study goes on to recommend, which makes the comparison conservative rather than favourable to its conclusion. It is restated among the limitations in Chapter 6.'));
c.push(H3('3.7.2  Batch normalisation'));
c.push(P('Batch normalisation layers remain frozen throughout both stages, including fine-tuning. These layers hold running estimates of the mean and variance of their inputs, learned from ImageNet over millions of images. Allowing them to re-estimate those statistics from batches of 32 medical photographs would replace stable estimates with noisy ones, which destabilises training and opens a gap between training and validation performance that is easily mistaken for overfitting.'));
c.push(H3('3.7.3  Training control'));
c.push(P('Both stages use the Adam optimiser and categorical cross-entropy loss, with a batch size of 32. Three callbacks control the process. Early stopping ends training when validation loss has not improved for eight epochs and restores the best weights. Learning rate reduction halves the rate after four epochs without improvement. Model checkpointing saves the weights whenever validation loss improves.'));
c.push(P('The checkpoint behaviour required a correction. As originally written, the second stage began with no record of what the first stage had achieved, so its first epoch was treated as the best result so far even when head training had produced a lower validation loss. In that situation the better weights were overwritten by worse ones. The fix was to seed the second stage checkpoint with the best validation loss from the first, so that it only saves weights that genuinely improve on everything seen before.'));
c.push(cap('3.5', 'Training configuration, identical for both architectures.'));
c.push(T([3400, 4800], [
  ['Setting', 'Value'],
  ['Input size', '224 × 224 × 3'],
  ['Batch size', '32'],
  ['Optimiser', 'Adam'],
  ['Loss', 'Categorical cross-entropy'],
  ['Stage 1 — epochs, learning rate', '15 at 1e-3, backbone frozen'],
  ['Stage 2 — epochs, learning rate', 'up to 25 at 1e-5, early stopping applied'],
  ['Backbone layers unfrozen in stage 2', 'top 30 (BatchNorm excluded)'],
  ['Dropout', '0.3'],
  ['L2 regularisation', '1e-4'],
  ['Early stopping patience', '8 epochs, best weights restored'],
  ['Learning rate reduction', '×0.5 after 4 epochs without improvement'],
  ['Random seed', '42'],
]));
c.push(FIG('Diagram of the two-stage transfer learning procedure'));
c.push(fcap('3.1', 'Two-stage transfer learning. The backbone is frozen while the head is trained, then the top layers are unfrozen at a much lower learning rate.'));

// ---- 3.8 -----------------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('3.8  Class imbalance'));
c.push(P('The four classes are unevenly represented, with a ratio of 4.8 to 1 between the largest and smallest. Left uncorrected, a model minimising overall error has an incentive to favour the common classes, and the rarest class here is also among the most serious.'));
c.push(P('Class weights were applied during training, inversely proportional to class frequency: healthy 0.56, nail fungal infection 2.69, foot wound 0.83 and foot ulcer 1.57. Weighting was preferred to resampling because oversampling a small class repeats the same images many times within an epoch, which encourages memorisation, and undersampling the large classes would have discarded usable data from a dataset that is already small.'));
c.push(P('The weights also carry a clinical meaning that happens to align with the statistical one. A missed ulcer has far worse consequences than a false alarm, so a training signal that penalises errors on the rarer serious classes more heavily is appropriate independently of the imbalance.'));

// ---- 3.9 -----------------------------------------------------------------
c.push(H2('3.9  Evaluation metrics'));
c.push(P('Performance is reported using accuracy, per-class precision, recall and F1, macro-averaged F1, balanced accuracy, Cohen’s kappa, macro one-versus-rest AUC, and the full confusion matrix.'));
c.push(P('Accuracy alone would be misleading here, for two reasons. The classes are imbalanced, so a model can achieve a respectable overall figure while performing poorly on a small class. And the classes are not equally important: a foot ulcer misread as healthy is a materially different failure from a healthy foot misread as a wound. Macro-averaged F1 gives each class equal weight regardless of size, and per-class recall makes visible what overall accuracy hides. Foot ulcer recall in particular is treated throughout as the headline clinical metric.'));
c.push(P('Computational cost is measured alongside accuracy, since the deployment argument depends on it. Parameter count and model size on disk are recorded, together with inference latency on both CPU and GPU. Latency is measured at batch size one, as the median of fifty runs after a warm-up, because the usage pattern being modelled is a single photograph taken on a phone rather than a batch processed on a server.'));

// ---- 3.10 ----------------------------------------------------------------
c.push(H2('3.10  Testing for dataset shortcuts'));
c.push(P('Because each class draws largely from one source, the classes differ in camera, lighting, framing and background as well as in the condition shown. A model could therefore achieve high accuracy by recognising the dataset. Section 2.5.8 identified this as a gap in the published literature. Three independent tests were designed to measure it, and all three were applied to both models.'));
c.push(H3('3.10.1  Accuracy within a single source'));
c.push(P('The first test is the most direct. The mendeley_foot source contributes both healthy feet and wounds, taken at the same clinic with the same equipment. Restricting evaluation to those images removes source as a usable signal entirely, because both classes share it. Accuracy on that subset therefore measures discrimination between the conditions themselves.'));
c.push(H3('3.10.2  Gradient-weighted class activation mapping'));
c.push(P('The second test uses Grad-CAM (Selvaraju et al., 2017) to produce a heat map showing which region of an image most influenced the prediction. If the models were relying on background or framing, attention would fall on the margins of the image rather than on the foot. Maps were generated for misclassified predictions as well as correct ones, since showing only successes would make the test decorative rather than diagnostic.'));
c.push(H3('3.10.3  Controlled occlusion'));
c.push(P('The third test masks part of the image and measures the resulting change in accuracy. The design of this test is where most of the methodological care sits, because the obvious version of it proves nothing.'));
c.push(P('Masking a border and observing that accuracy barely moves is not evidence that the border is unimportant, because masking any region removes information and a small region will generally cost little. The test is only interpretable against a control that removes the same amount of information from somewhere else. The ablation therefore has four arms: the unmodified image, a masked 20-pixel border, an interior region of exactly equal area masked in the same way, and a fourth arm masking the centre so that only the border remains visible. Comparing the border arm against the equal-area interior control isolates location as the only variable. The fourth arm measures how much class information the border carries on its own, and also detects the case where every arm collapses to chance, in which the test would be uninformative and should not be reported as a pass.'));
c.push(P('Before the ablation was run, mean border brightness was measured for each class and source, to establish whether a border shortcut was plausible in the first place rather than testing for something that could not exist.'));

// ---- 3.11 ---------------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('3.11  Calibration and selective prediction'));
c.push(P('A classifier reports a confidence alongside its prediction, and that number is only useful if it means something. Modern neural networks are known to be systematically overconfident (Guo et al., 2017). For a tool intended for use where no specialist is available to check its output, a confidently wrong answer is worse than an admission of uncertainty, so the trustworthiness of the confidence value matters alongside the accuracy of the prediction.'));
c.push(P('Calibration was measured using expected calibration error, computed over fifteen confidence bins. The correction applied is temperature scaling, a single-parameter method that divides the logits by a learned constant before the softmax. One parameter cannot meaningfully overfit, and the transformation is monotonic, so it rescales confidence without changing which class is predicted. This was verified rather than assumed. The temperature was fitted on the validation set only.'));
c.push(P('On top of calibrated confidence, a selective prediction rule was added. Below a threshold confidence the system declines to name a condition and recommends consulting a clinician instead. The threshold is chosen from the validation risk-coverage curve as the lowest value that achieves 99 percent accuracy among the cases still answered, subject to answering at least 70 percent of them. The coverage floor exists because a system that abstains on everything is perfectly safe and completely useless.'));

// ---- 3.12 ---------------------------------------------------------------
c.push(H2('3.12  Referral policy'));
c.push(P('Classifying an image and deciding what a person should do about it are different problems, and the second is not symmetric. Referring a healthy person to a clinic wastes an appointment. Failing to refer someone with a foot ulcer can cost them a limb: diabetic foot ulceration precedes roughly eighty percent of non-traumatic lower limb amputations. The referral policy was therefore designed to be deliberately cautious, and this section states how.'));
c.push(P('Each class maps to one of four action bands: urgent, meaning care should be sought within 24 hours; prompt, within a few days; routine, an ordinary appointment; and self-care, no appointment needed on this result alone. Three rules then modify that mapping.'));
c.push(P('First, a serious condition holding meaningful probability escalates the band even when it is not the most likely class. A prediction of healthy at 55 percent with foot ulcer at 40 percent is reported as urgent, not as healthy. This is the rule the policy exists for, because reporting that case as healthy would be the most dangerous output the system could produce.'));
c.push(P('Second, abstention never reassures. When confidence falls below the referral threshold the system names no condition and routes the person to a clinician, rather than falling back on the most likely class. An inconclusive screening is not a negative result.'));
c.push(P('Third, every result carries text stating what should send someone to seek care regardless of what the system reported, including results of healthy. The failure that matters most is somebody with a genuine ulcer reading that nothing is wrong and staying at home.'));
c.push(P('It must be stated clearly that these bands and thresholds are a human design decision and not an output of the model. The network was trained on four class labels and has never seen severity, infection, depth, circulation or patient history, so it has no basis on which to judge urgency. Presenting a model-derived urgency would be inventing a capability the data does not support. The escalation thresholds were subsequently tuned on the validation set, as described in Section 3.13.'));

// ---- 3.13 ---------------------------------------------------------------
c.push(H2('3.13  Statistical analysis'));
c.push(P('Comparing the two models required care, because they are evaluated on the same images. An image that is difficult for one architecture tends to be difficult for the other, so their errors are correlated and treating the two accuracy figures as independent samples would overstate the strength of any difference between them.'));
c.push(P('The comparison was therefore made pairwise, using McNemar’s exact test on the per-image outcomes. Only the images the two models disagree on carry information: under the null hypothesis that the models are equally accurate, each disagreement is equivalent to a fair coin toss, and the exact binomial test on the disagreement counts gives the p-value. The exact form was used in preference to the chi-square approximation because the number of disagreements here is around twenty, which is precisely the regime in which the approximation is least reliable.'));
c.push(P('A paired bootstrap confidence interval was computed alongside the test, resampling images rather than models so that each pair stays together, with 10,000 resamples. The interval answers a question the p-value does not: not merely whether a difference exists, but how large it could plausibly be. For a deployment decision that is the more useful quantity.'));
c.push(P('The same paired test was applied within each class and within each source, because an overall result can conceal a difference confined to a single class, and the class that matters most in this application is also one of the smallest.'));
c.push(P('Proportions measured on small sets are reported with a Wilson confidence interval rather than as a percentage alone. The Wilson form was chosen in preference to the ordinary normal approximation because the quantities of interest here are often close to zero or one, or measured on few images, and the normal approximation misbehaves in both of those cases: it can produce an interval extending below zero or above one, and it is too narrow when the count is small. Where a paired design was used on a small set, as in the external validation described in Section 3.14, the paired test was applied to the pairs rather than the two groups being compared as independent samples, for the same reason it was used to compare the two architectures.'));
c.push(P('Finally, the escalation threshold of the referral policy was tuned by sweeping it across the validation set and measuring, at each value, how many misclassified ulcers reached the urgent band and how many people needing no care were escalated. The test set was not used for this, and could not be: it had already been read, and tuning a parameter against it would have spent the one split the rest of the study depends on having been used once.'));

// ---- 3.14 ---------------------------------------------------------------
c.push(H2('3.14  Validation beyond the source datasets'));
c.push(P('Everything described so far is measured on images drawn from the three source datasets. Two further tests were designed to probe the two ways in which a real photograph can differ from those: it can show one of the four conditions but have been captured differently, or it can show a condition the system was never taught. These are separate problems and they were measured separately.'));
c.push(H3('3.14.1  Photographs captured outside the source datasets'));
c.push(P('A small set of photographs was collected on a mobile phone, outside all three source datasets, and scored through the prototype’s batch evaluation tab using the deployment configuration: MobileNetV2, temperature scaling applied, and the abstention threshold set at the value derived on the validation set. Using the deployment configuration rather than a more permissive one matters, because the question being asked is what the system as recommended would do with these photographs.'));
c.push(P('The collection was designed as a paired comparison rather than as a sample. Each subject was photographed twice, once against a plain background and once against a patterned one, holding framing and lighting roughly constant so that the background is what differs within a pair. The reason for that design is statistical: on a set of this size, comparing two independent groups of photographs would be uninformative, whereas photographing the same feet under both conditions allows the background to be tested with a paired test on the same number of images. The design was prompted by a single observation made while the prototype was being tested, in which the same healthy feet were classified differently against two different backgrounds.'));
c.push(P('Two limits of this design were accepted before the photographs were taken, and are stated here rather than presented later as caveats. The set is small, so it can establish the direction of any difference from the test set figures but not its size, which is why every figure from it is reported with a confidence interval. And the subjects available were people with healthy feet, so the set can measure how often a healthy foot is misread and cannot measure how often a real condition is missed. The second limit is the more serious one, and it is the reason this measurement is described as a first external validation rather than as a validation of the system.'));
c.push(H3('3.14.2  A condition outside the four classes'));
c.push(P('The four classes were fixed by the project proposal, and the Figshare source contains a fifth condition, nail dystrophy, which was therefore excluded during preprocessing. Those 578 images were kept aside and used as a test of what the system does with a condition it has no category for. Because the images were excluded before the split was made, the models have never seen them in training, validation or test, and because they are real clinical photographs of a real condition the test does not depend on artificial or adversarial inputs.'));
c.push(P('The measurements taken were the confidence assigned to these images compared with the confidence assigned to the test set, the proportion that the abstention threshold declined, the classes they were assigned to, and the referral bands they would have produced. The first two ask whether the system can tell that it is out of its depth; the last two ask what a person photographing this condition would actually be told.'));

// ---- 3.15 ---------------------------------------------------------------
c.push(H2('3.15  Tools and reproducibility'));
c.push(P('The system was implemented in Python using TensorFlow 2.16 with Keras 3, together with scikit-learn for metrics, SciPy for the statistical tests, Pillow and NumPy for image handling, Matplotlib for figures and Streamlit for the prototype interface. Training was carried out in Google Colab on an NVIDIA Tesla T4.'));
c.push(P('Every path, hyper-parameter, split ratio, class name and augmentation setting is held in a single configuration module which every script imports. Changing an experimental setting therefore means editing one file rather than hunting through several, and no script contains a hard-coded value that could drift out of step with the others. That module validates itself when it is imported, so an inconsistent configuration fails immediately rather than part way through a training run.'));
c.push(P('A fixed random seed of 42 governs the split and the training initialisation. All intermediate outputs, including per-image predictions with full probability vectors, are written to disk, which allows the statistical analysis and the referral policy to be re-examined without repeating inference.'));
c.push(P('The proposal listed OpenCV among the resources required. In the event it was not used, since Pillow and NumPy proved sufficient for every image operation the pipeline performs. This is recorded here rather than left as an unexplained discrepancy between the proposal and the implementation.'));

// ---- 3.16 ---------------------------------------------------------------
c.push(H2('3.16  Ethical considerations'));
c.push(P('All image data used for training and evaluation comes from publicly available research datasets, used in accordance with their stated terms. The datasets contain photographs of feet and nails only, with no faces and no accompanying identifying information, and no attempt was made to re-identify any individual. The source datasets are cited in full so that the work can be checked and reproduced.'));
c.push(P('The system is positioned throughout as an assistive screening aid and not as a diagnostic device or a replacement for professional judgement. This is not merely a statement of intent: the prototype displays a disclaimer to that effect with every result, declines to name a condition when its confidence is low, and always presents guidance that directs the user toward a healthcare professional. The referral policy described in Section 3.12 is built around the principle that the system should never be the reason somebody with a serious condition fails to seek care.'));
// ---- 3.17 ---------------------------------------------------------------
c.push(H2('3.17  Summary'));
c.push(P('This chapter has described an experimental design built around a controlled comparison: two architectures trained on identical data, with identical preprocessing, augmentation, class weights and training procedure, so that any difference between them can be attributed to the architecture rather than to the conditions of the experiment. It has described how three public datasets were combined, deduplicated and split so that no image and no photographic session appears on both sides of the train and test boundary, and why the test set is read exactly once.'));
c.push(P('It has also described four elements that go beyond straightforward classification: a set of tests designed to check whether the models classify disease or dataset, a calibration and abstention procedure that makes the reported confidence meaningful and allows the system to decline, a referral policy that converts a prediction into a recommended action while erring deliberately toward caution, and two tests of what happens outside the source datasets, one changing how the photograph was taken and one changing the condition in it.'));
c.push(P('Chapter 4 describes how this design was implemented, including the problems encountered during implementation and the verification steps introduced in response to them.'));

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
  fs.writeFileSync(require('path').join(__dirname, 'Chapter3_Methodology_DRAFT.docx'), b);
  console.log('written', b.length, 'bytes');
});
