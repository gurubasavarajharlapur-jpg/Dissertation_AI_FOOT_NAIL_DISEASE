const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
        TableOfContents, PageBreak, ShadingType, Footer, PageNumber } = require('docx');
const fs = require('fs');
const ACCENT = '1F4E79', GREY = '595959';

const ch1 = require('./ch1'), ch3 = require('./ch3_mod'), ch4 = require('./ch4_mod');
const ch5 = require('./ch5_mod'), ch6 = require('./ch6'), ch7 = require('./ch7');

const centre = (t, o = {}) => new Paragraph({ spacing: { after: o.after ?? 140 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: t, bold: o.b, size: o.size ?? 22, color: o.c, font: 'Calibri' })] });
const P = (t, o = {}) => new Paragraph({ spacing: { after: o.after ?? 160, line: 300 },
  alignment: o.left ? undefined : AlignmentType.JUSTIFIED,
  children: [new TextRun({ text: t, size: 22, italics: o.i, font: 'Calibri' })] });

const children = [];

// ------------------------------------------------------------- title page --
children.push(
  centre('MIDDLESEX UNIVERSITY', { size: 24, c: GREY, after: 1000 }),
  centre('AI-Based Early Detection and Classification of', { b: true, size: 34, after: 80 }),
  centre('Foot and Nail Conditions Using Transfer Learning', { b: true, size: 34, after: 80 }),
  centre('for Rural Healthcare', { b: true, size: 34, after: 900 }),
  centre('Gurubasavaraj Harlapur', { size: 26, after: 80 }),
  centre('M01093116', { size: 24, c: GREY, after: 700 }),
  centre('Supervisor: Dr Krishnadas Nanath', { size: 22, after: 700 }),
  centre('A dissertation submitted in partial fulfilment', { size: 21, c: GREY, after: 60 }),
  centre('of the requirements for the degree of', { size: 21, c: GREY, after: 60 }),
  centre('MSc Data Science and Artificial Intelligence', { size: 21, c: GREY, after: 600 }),
  new Paragraph({ children: [new PageBreak()] }),
);

// -------------------------------------------------------------- abstract --
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { after: 240 },
  children: [new TextRun({ text: 'Abstract', bold: true, size: 32, color: ACCENT, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { before: 100, after: 200, line: 300 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4E5' },
  children: [ new TextRun({ text: 'WRITE THIS LAST.  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'About 300 words, in one paragraph or two. Cover, in order: the problem and why it matters; what you built; the headline results; the recommendation; and the main limitation. Write it once the rest of the dissertation is finished, because an abstract written first describes what you intended rather than what you did. The draft below is a starting point to rewrite, not text to keep.', size: 20, italics: true, font: 'Calibri' }) ] }));
children.push(P('Foot and nail conditions are frequently overlooked in rural communities, where limited access to healthcare means that problems which would respond to early treatment are often first seen once they have become serious. This dissertation develops and evaluates a transfer learning based image classification system intended to support earlier detection, classifying photographs into four categories: healthy, nail fungal infection, foot wound and foot ulcer.'));
children.push(P('Two architectures were trained under identical conditions on 8,374 images assembled from three public datasets, and evaluated once on 1,248 held-out images. MobileNetV2, the deployment candidate, reached 98.32 percent accuracy; ResNet50, the accuracy reference, reached 98.96 percent. The difference between them is not statistically significant under a paired test, and a bootstrap interval bounds any real difference at 1.4 accuracy points, while MobileNetV2 requires a tenth of the parameters and roughly half the inference time on CPU.'));
children.push(P('Because each class draws largely from a different source, three independent tests examined whether the models classify the condition or the dataset, including a controlled occlusion ablation measured against an equal-area interior control. All three support the former. Confidence was calibrated and a referral threshold derived, allowing the compact model to decline the least confident three percent of cases and be correct on 99.6 percent of the remainder, and a triage policy converts a classification into a recommended action that escalates when a serious condition is plausible.'));
children.push(P('The system was then tested outside the data it was built on. On ten photographs taken on a mobile phone by the author, six of ten were classified correctly, against 98.32 percent on the held-out clinical images; all four errors were healthy feet read as wounds, and the referral threshold declined three of them. On 578 photographs of a nail condition outside the four classes, confidence was indistinguishable from confidence on the test set, although 98.4 percent were still routed to a clinician because the nearest available class carries a referral.'));
children.push(P('The principal limitation follows from that. All training data comes from clinical collections, and performance on photographs taken outside them is substantially lower, though ten images bound the direction of that drop rather than its size. The work demonstrates feasibility on clinical photographs and efficiency sufficient for phone deployment; it does not demonstrate performance in the rural setting that motivates it.'));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ------------------------------------------------ acknowledgements + decl --
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { after: 240 },
  children: [new TextRun({ text: 'Acknowledgements', bold: true, size: 32, color: ACCENT, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { before: 100, after: 200, line: 300 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4E5' },
  children: [ new TextRun({ text: 'WRITE THIS YOURSELF.  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'A short paragraph thanking your supervisor and anyone else who helped. This one should be in your own words.', size: 20, italics: true, font: 'Calibri' }) ] }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ------------------------------------------------------------ declaration --
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { after: 240 },
  children: [new TextRun({ text: 'Declaration', bold: true, size: 32, color: ACCENT, font: 'Calibri' })] }));
children.push(P('I declare that this dissertation is my own work and has been produced in accordance with the University\u2019s regulations on academic integrity. It has not been submitted, in whole or in part, for any other degree or qualification at this or any other institution. All sources of information have been acknowledged, and all material taken from the work of others has been properly cited and referenced.'));
children.push(P('I acknowledge the use of generative artificial intelligence in the preparation of this dissertation. It was used to assist with the development of the software described in Chapter 4, the implementation of the analysis and statistical procedures reported in Chapter 5, and the drafting of the written chapters from results that I produced and verified. The research design, the experimental work, the collection and preparation of the data, and all decisions and conclusions reported in this dissertation are my own.'));
children.push(new Paragraph({ spacing: { before: 500, after: 260 },
  children: [new TextRun({ text: 'Name:', size: 22, font: 'Calibri' }),
             new TextRun({ text: '\t\t' + '_'.repeat(46), size: 22, color: GREY, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 260 },
  children: [new TextRun({ text: 'Student number:', size: 22, font: 'Calibri' }),
             new TextRun({ text: '\t' + '_'.repeat(46), size: 22, color: GREY, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 260 },
  children: [new TextRun({ text: 'Date:', size: 22, font: 'Calibri' }),
             new TextRun({ text: '\t\t' + '_'.repeat(46), size: 22, color: GREY, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 260 },
  children: [new TextRun({ text: 'Signature:', size: 22, font: 'Calibri' }),
             new TextRun({ text: '\t' + '_'.repeat(46), size: 22, color: GREY, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { before: 400, after: 200, line: 300 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4E5' },
  children: [ new TextRun({ text: 'BEFORE SUBMITTING.  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'If your programme publishes its own declaration wording, replace the first paragraph above with theirs exactly rather than paraphrasing. Adjust the second paragraph so it describes accurately what you want on record, and delete it if your supervisor prefers the AI use recorded elsewhere. Then delete this box.', size: 20, italics: true, font: 'Calibri' }) ] }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ------------------------------------------------------------------ TOC ---
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { after: 240 },
  children: [new TextRun({ text: 'Table of Contents', bold: true, size: 32, color: ACCENT, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 200 },
  children: [new TextRun({ text: 'In Word: right-click the table below and choose "Update Field", then "Update entire table", to fill in the page numbers.', size: 19, italics: true, color: GREY, font: 'Calibri' })] }));
children.push(new TableOfContents('Contents', { hyperlink: true, headingStyleRange: '1-3' }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ------------------------------------------------------------- chapters ---
children.push(...ch1);

// Chapter 2 placeholder — their graded chapter, pasted in Word to keep its formatting
children.push(new Paragraph({ pageBreakBefore: true, spacing: { after: 280 },
  children: [new TextRun({ text: 'Chapter 2', bold: true, size: 36, color: ACCENT, font: 'Calibri' })] }));
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { after: 360 },
  children: [new TextRun({ text: 'Literature Review', bold: true, size: 36, color: ACCENT, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { before: 200, after: 200, line: 300 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4E5' },
  children: [ new TextRun({ text: 'PASTE YOUR SUBMITTED CHAPTER 2 HERE.  ', bold: true, size: 21, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'Open your graded Proposal and Literature Review document, select everything from the Chapter 2 heading to the end of section 2.5.9, copy it, and paste it here using Home > Paste > Keep Source Formatting. Pasting from the original preserves your tables (2.1 to 2.16), your figures and your citation formatting, all of which would be lost if the text were retyped. Delete this box afterwards, then update the Table of Contents.', size: 21, italics: true, font: 'Calibri' }) ] }));
children.push(P('Chapter 2 reviews the clinical background to foot and nail conditions, the development of convolutional neural networks and transfer learning for medical image classification, existing AI systems for this problem, and the research gaps in Section 2.5.8 that this dissertation addresses.', { i: true }));

children.push(...ch3, ...ch4, ...ch5, ...ch6, ...ch7);

// ---------------------------------------------------------- references ----
children.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, spacing: { after: 240 },
  children: [new TextRun({ text: 'References', bold: true, size: 32, color: ACCENT, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { before: 100, after: 200, line: 300 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4E5' },
  children: [ new TextRun({ text: 'MERGE THESE WITH YOUR CHAPTER 2 REFERENCES.  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'The four below are cited in Chapters 3 to 7 and must appear. Your literature review has its own list; combine them into one alphabetical sequence in your programme’s required style.', size: 20, italics: true, font: 'Calibri' }) ] }));
[
 'Armstrong, D.G., Boulton, A.J.M. and Bus, S.A. (2017) ‘Diabetic Foot Ulcers and Their Recurrence’, New England Journal of Medicine, 376(24), pp. 2367–2375.',
 'Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) ‘On Calibration of Modern Neural Networks’, Proceedings of the 34th International Conference on Machine Learning, pp. 1321–1330.',
 'Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2017) ‘Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization’, Proceedings of the IEEE International Conference on Computer Vision, pp. 618–626.',
 'Wang, C., Anisuzzaman, D.M., Williamson, V., Dhar, M.K., Rostami, B., Niezgoda, J., Gopalakrishnan, S. and Yu, Z. (2020) ‘Fully Automatic Wound Segmentation with Deep Convolutional Neural Networks’, Scientific Reports, 10, 21897.',
].forEach(r => children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: r, size: 21, font: 'Calibri' })] })));

// ---------------------------------------------------------- appendices ----
children.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, spacing: { after: 240 },
  children: [new TextRun({ text: 'Appendices', bold: true, size: 32, color: ACCENT, font: 'Calibri' })] }));
children.push(P('Suggested appendices, each referenced from the relevant chapter:', { left: true }));
[
 'Appendix A — Source code repository and structure.',
 'Appendix B — Full configuration file, showing every hyper-parameter used.',
 'Appendix C — Complete per-class results for both architectures.',
 'Appendix D — Additional Grad-CAM examples, including further misclassified cases.',
 'Appendix E — External validation photographs and the batch evaluation output.',
].forEach(a => children.push(new Paragraph({ spacing: { after: 120 },
  children: [new TextRun({ text: a, size: 21, font: 'Calibri' })] })));

// ---------------------------------------------------------------- build ---
const doc = new Document({
  styles: { default: {
    heading1: { run: { size: 32, bold: true, color: ACCENT, font: 'Calibri' },
                paragraph: { spacing: { before: 360, after: 200 } } },
    heading2: { run: { size: 28, bold: true, color: ACCENT, font: 'Calibri' },
                paragraph: { spacing: { before: 340, after: 160 } } },
    heading3: { run: { size: 24, bold: true, color: '2F5496', font: 'Calibri' },
                paragraph: { spacing: { before: 260, after: 120 } } },
  } },
  features: { updateFields: true },
  sections: [{
    properties: { page: { margin: { top: 1440, bottom: 1440, left: 1700, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 19, color: GREY, font: 'Calibri' })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(require('path').join(__dirname, 'Dissertation_FULL_DRAFT.docx'), b);
  console.log('written', (b.length/1024).toFixed(0), 'KB');
});
