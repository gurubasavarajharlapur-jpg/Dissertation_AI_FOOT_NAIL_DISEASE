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
  children: [ new TextRun({ text: 'READ THIS OVER BEFORE YOU SUBMIT.  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'Every figure below comes from your results. Change any wording that does not sound like you, and it runs to about 390 words, so trim the first and fifth paragraphs if your programme sets a 300-word limit. Delete this box afterwards.', size: 20, italics: true, font: 'Calibri' }) ] }));
children.push(P('Foot and nail problems are easy to ignore, and in rural areas they are also hard to get looked at. A condition that would have been simple to treat early is often first seen once it has become serious. Foot ulcers are the worst case, because they come before most amputations of the lower limb that are not caused by an accident.'));
children.push(P('Almost everyone now carries a phone with a camera, so this project asks a simple question: can a photograph taken on an ordinary phone tell someone that their foot needs to be seen by a professional?'));
children.push(P('To answer it I trained two deep learning models to sort photographs into four groups, a healthy foot or nail, a fungal nail infection, a foot wound and a foot ulcer. The 8,374 images came from three public datasets. Both models were trained the same way on the same data so that the comparison between them would be fair. MobileNetV2 is small enough to run on a phone; ResNet50 is about ten times larger and was used as a yardstick.'));
children.push(P('On 1,248 images kept back until the end, MobileNetV2 was right 98.32 percent of the time and ResNet50 98.96 percent. A statistical test on the same images shows that small gap could easily be chance, so the smaller model is the sensible choice for a phone. Three further tests checked that the models read the condition in the photograph rather than recognising which dataset it came from, and all three passed. The system also says how soon someone should seek help, and it leans towards sending them to a clinician: when it is not confident it refuses to name a condition instead of saying that nothing is wrong.'));
children.push(P('The system was then given photographs it was never built for. Ten photographs of healthy feet, taken on five ordinary phones indoors, were classified correctly only six times, and all four mistakes read a healthy foot as a wound. Shown 578 photographs of a nail condition it had never been taught, it confidently gave each one the closest label it knew.'));
children.push(P('So the method works on clinical photographs and is small and fast enough for a phone. What it has not yet been shown to do is work on photographs taken in the place it was designed for, and that is where this work should go next.'));
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
  children: [ new TextRun({ text: 'CHECK THE STYLE, AND FILL THE GAPS BELOW.  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'This list combines the bibliography from your submitted proposal, the works cited in Chapter 2, and the works cited in Chapters 3 to 7, in one alphabetical sequence. Check it against the referencing style your programme requires, and read the box at the end of the list: nine citations that appear in Chapter 2 have no entry here because their source could not be established, and each needs either a full reference or removal from the text.', size: 20, italics: true, font: 'Calibri' }) ] }));
[
 'Almufadi, N.F., Alhasson, H.F. and Alharbi, S.S. (2025) ‘E-DFu-Net: An efficient deep convolutional neural network model for diabetic foot ulcer classification’, Biomolecules and Biomedicine, 25(2), pp. 445–460.',
 'Armstrong, D.G., Boulton, A.J.M. and Bus, S.A. (2017) ‘Diabetic foot ulcers and their recurrence’, New England Journal of Medicine, 376(24), pp. 2367–2375.',
 'Cassidy, B., Kendrick, C., Reeves, N.D., Pappachan, J.M., O’Shea, C., Armstrong, D.G. and Yap, M.H. (2021) ‘Diabetic Foot Ulcer Grand Challenge 2021: evaluation and summary’, arXiv preprint.',
 'Deloitte (2024) Artificial Intelligence in Healthcare: Current Applications and Future Opportunities. Deloitte Insights.',
 'Esteva, A., Kuprel, B., Novoa, R.A., Ko, J., Swetter, S.M., Blau, H.M. and Thrun, S. (2017) ‘Dermatologist-level classification of skin cancer with deep neural networks’, Nature, 542(7639), pp. 115–118.',
 'Esteva, A., Robicquet, A., Ramsundar, B., Kuleshov, V., DePristo, M., Chou, K., Cui, C., Corrado, G., Thrun, S. and Dean, J. (2019) ‘A guide to deep learning in healthcare’, Nature Medicine, 25(1), pp. 24–29.',
 'Girmaw, D.W. and Taye, G.B. (2025) ‘MobileNetV2 model for detecting and grading diabetic foot ulcer’, Discover Applied Sciences, 7(268), pp. 1–20.',
 'Goodfellow, I., Bengio, Y. and Courville, A. (2016) Deep Learning. Cambridge, MA: MIT Press.',
 'Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) ‘On calibration of modern neural networks’, Proceedings of the 34th International Conference on Machine Learning, pp. 1321–1330.',
 'He, K., Zhang, X., Ren, S. and Sun, J. (2016) ‘Deep residual learning for image recognition’, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778.',
 'Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q.V. and Adam, H. (2019) ‘Searching for MobileNetV3’, Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1314–1324.',
 'Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M. and Adam, H. (2017) ‘MobileNets: efficient convolutional neural networks for mobile vision applications’, arXiv preprint arXiv:1704.04861.',
 'Howard, J. and Ruder, S. (2018) ‘Universal language model fine-tuning for text classification’, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, pp. 328–339.',
 'Huang, H.N., Zhang, T., Yang, C.T., Sheen, Y.J. and Chen, H.M. (2022) ‘Image segmentation using transfer learning and Fast R-CNN for diabetic foot wound treatments’, Frontiers in Public Health, 10, pp. 1–15.',
 'International Working Group on the Diabetic Foot (2023) IWGDF Guidelines on the Prevention and Management of Diabetic Foot Disease. Amsterdam: IWGDF.',
 'Krizhevsky, A., Sutskever, I. and Hinton, G.E. (2012) ‘ImageNet classification with deep convolutional neural networks’, Advances in Neural Information Processing Systems, 25, pp. 1097–1105.',
 'LeCun, Y., Bengio, Y. and Hinton, G. (2015) ‘Deep learning’, Nature, 521(7553), pp. 436–444.',
 'Litjens, G., Kooi, T., Bejnordi, B.E., Setio, A.A.A., Ciompi, F., Ghafoorian, M., van der Laak, J.A.W.M., van Ginneken, B. and Sánchez, C.I. (2017) ‘A survey on deep learning in medical image analysis’, Medical Image Analysis, 42, pp. 60–88.',
 'Oltu, B., Karaca, B.K., Erdem, H. and Özgür, A. (2021) ‘A systematic review of transfer learning based approaches for diabetic retinopathy detection’, arXiv preprint.',
 'Pan, S.J. and Yang, Q. (2010) ‘A survey on transfer learning’, IEEE Transactions on Knowledge and Data Engineering, 22(10), pp. 1345–1359.',
 'Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M.P. and Ng, A.Y. (2017) ‘CheXNet: radiologist-level pneumonia detection on chest X-rays with deep learning’, arXiv preprint arXiv:1711.05225.',
 'Sait, A.R.W. and Nagaraj, R. (2025) ‘Diabetic foot ulcers detection model using a hybrid convolutional neural networks–vision transformers approach’, Diagnostics, 15(6), pp. 1–24.',
 'Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.-C. (2018) ‘MobileNetV2: inverted residuals and linear bottlenecks’, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4510–4520.',
 'Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2017) ‘Grad-CAM: visual explanations from deep networks via gradient-based localization’, Proceedings of the IEEE International Conference on Computer Vision, pp. 618–626.',
 'Shorten, C. and Khoshgoftaar, T.M. (2019) ‘A survey on image data augmentation for deep learning’, Journal of Big Data, 6(60), pp. 1–48.',
 'Tan, M. and Le, Q.V. (2019) ‘EfficientNet: rethinking model scaling for convolutional neural networks’, Proceedings of the 36th International Conference on Machine Learning, pp. 6105–6114.',
 'Topol, E.J. (2019) ‘High-performance medicine: the convergence of human and artificial intelligence’, Nature Medicine, 25(1), pp. 44–56.',
 'Wang, C., Anisuzzaman, D.M., Williamson, V., Dhar, M.K., Rostami, B., Niezgoda, J., Gopalakrishnan, S. and Yu, Z. (2020) ‘Fully automatic wound segmentation with deep convolutional neural networks’, Scientific Reports, 10, 21897.',
 'World Health Organization (2021) Global Strategy on Digital Health 2020–2025. Geneva: World Health Organization.',
 'Yap, M.H., Cassidy, B., Pappachan, J.M., O’Shea, C., Gillespie, D. and Reeves, N. (2021) ‘Analysis towards classification of infection and ischaemia of diabetic foot ulcers’, arXiv preprint.',
].forEach(r => children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: r, size: 21, font: 'Calibri' })] })));

children.push(new Paragraph({ spacing: { before: 300, after: 200, line: 300 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4E5' },
  children: [ new TextRun({ text: 'TEN ENTRIES STILL NEEDED.  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'Eight of these are cited in the text of Chapter 2 but are not in the bibliography you submitted with the proposal, and their sources could not be identified with enough confidence to write an entry. Find each one and add it, or delete the citation from Chapter 2 — an in-text citation with no reference is a marked fault, and a reference to a work that does not exist is a more serious one, so check each against a database before keeping it. Two dataset citations are also needed: both the Mendeley and the figshare pages carry a ready-made citation you can copy.', size: 20, italics: true, font: 'Calibri' }) ] }));
[
 'Goyal et al. (2020) — cited in Chapter 2.',
 'Gupta et al. (2022) — cited in Chapter 2, section 2.5.2.',
 'Lipner and Scher (2019) — cited in Chapter 2, section 2.5.2.',
 'Lipsky et al. (2020) — cited in Chapter 2.',
 'Patel et al. (2017) — cited in Chapter 2.',
 'Wang et al. (2022) — cited in Chapter 2. Note this is a different Wang from the 2020 entry above.',
 'World Bank (2023) — cited in Chapter 2, section 2.4.',
 'Zhang et al. (2018) — cited in Chapter 2.',
 'Mendeley Data, DOI 10.17632/hsj38fwnvr.3 — the source of the healthy and wound images (Section 3.2). Copy the citation from data.mendeley.com.',
 'figshare, article 5398573, Model Onychomycosis Training Datasets (JPG thumbnails) and Validation Datasets (JPG images) — the source of the nail images (Section 3.2). Copy the citation from figshare.com.',
].forEach(r => children.push(new Paragraph({ spacing: { after: 120, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: r, size: 21, italics: true, font: 'Calibri' })] })));

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
