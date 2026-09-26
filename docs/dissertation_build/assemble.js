const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
        TableOfContents, PageBreak, ShadingType, Footer, PageNumber } = require('docx');
const fs = require('fs');
const ACCENT = '1F4E79', GREY = '595959';

const ch1 = require('./ch1_mod'), ch3 = require('./ch3_mod'), ch4 = require('./ch4_mod');
const ch5 = require('./ch5_mod'), ch6 = require('./ch6_mod'), ch7 = require('./ch7_mod');
const appendices = require('./appendices');

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
  centre('September 2026', { size: 21, c: GREY, after: 300 }),
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
  children: [ new TextRun({ text: 'MAKE THIS YOUR OWN.  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'A draft in your voice. Add or remove whoever you like, then delete this box.', size: 20, italics: true, font: 'Calibri' }) ] }));
children.push(P('I would like to thank my supervisor, Dr Krishnadas Nanath, for his guidance throughout this project. His feedback shaped the direction of the work at several points, and his questions were often the reason I looked more carefully at a result instead of accepting it.'));
children.push(P('I am grateful to the four friends who agreed to have their feet photographed for the external validation in Chapter 5. It is an odd thing to be asked, and the ten photographs they made possible turned out to be one of the more useful parts of this dissertation.'));
children.push(P('I would also like to acknowledge the researchers who made their image datasets publicly available. Work of this kind is only possible because other people chose to share their data, and this project would not exist without them.'));
children.push(P('Finally, I thank my family for their support and patience while I was working on this.'));
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

// ------------------------------------------- lists of tables and figures ---
// Built from the TC fields carried by every caption, so Word fills in real page
// numbers on the same field update that fills in the contents page.
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { after: 200 },
  children: [new TextRun({ text: 'List of Tables', bold: true, size: 32, color: ACCENT, font: 'Calibri' })] }));
children.push(new TableOfContents('Tables', { hyperlink: true, tcFieldIdentifier: 'T' }));
children.push(new Paragraph({ spacing: { before: 360 }, heading: HeadingLevel.HEADING_1,
  children: [new TextRun({ text: 'List of Figures', bold: true, size: 32, color: ACCENT, font: 'Calibri' })] }));
children.push(new TableOfContents('Figures', { hyperlink: true, tcFieldIdentifier: 'F' }));
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
  children: [ new TextRun({ text: 'CHECK THE STYLE AGAINST YOUR HANDBOOK.  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'One alphabetical list for the whole dissertation: the bibliography from your proposal, everything cited in Chapter 2, everything cited in Chapters 3 to 7, and the two image datasets. It is written in Harvard style, so check it against the style your programme requires. Four citations need a decision from you and are listed after the references. Delete this box afterwards.', size: 20, italics: true, font: 'Calibri' }) ] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Almufadi, N.F., Alhasson, H.F. and Alharbi, S.S. (2025) ‘E-DFu-Net: an efficient deep convolutional neural network model for diabetic foot ulcer classification’, Biomolecules and Biomedicine, 25(2), pp. 445–460.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Armstrong, D.G., Boulton, A.J.M. and Bus, S.A. (2017) ‘Diabetic foot ulcers and their recurrence’, New England Journal of Medicine, 376(24), pp. 2367–2375.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Cassidy, B., Kendrick, C., Reeves, N.D., Pappachan, J.M., O’Shea, C., Armstrong, D.G. and Yap, M.H. (2021) ‘Diabetic Foot Ulcer Grand Challenge 2021: evaluation and summary’, arXiv preprint.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Deloitte (2024) Artificial Intelligence in Healthcare: Current Applications and Future Opportunities. Deloitte Insights.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Esteva, A., Kuprel, B., Novoa, R.A., Ko, J., Swetter, S.M., Blau, H.M. and Thrun, S. (2017) ‘Dermatologist-level classification of skin cancer with deep neural networks’, Nature, 542(7639), pp. 115–118.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Esteva, A., Robicquet, A., Ramsundar, B., Kuleshov, V., DePristo, M., Chou, K., Cui, C., Corrado, G., Thrun, S. and Dean, J. (2019) ‘A guide to deep learning in healthcare’, Nature Medicine, 25(1), pp. 24–29.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Girmaw, D.W. and Taye, G.B. (2025) ‘MobileNetV2 model for detecting and grading diabetic foot ulcer’, Discover Applied Sciences, 7(268), pp. 1–20.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Goodfellow, I., Bengio, Y. and Courville, A. (2016) Deep Learning. Cambridge, MA: MIT Press.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Goyal, M., Reeves, N.D., Rajbhandari, S., Ahmad, N., Wang, C. and Yap, M.H. (2020) ‘Recognition of ischaemia and infection in diabetic foot ulcers: dataset and techniques’, Computers in Biology and Medicine, 117, 103616.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) ‘On calibration of modern neural networks’, Proceedings of the 34th International Conference on Machine Learning, pp. 1321–1330.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Gupta, A.K. et al. (2022) ‘Diagnosing onychomycosis: a step forward?’, Journal of Cosmetic Dermatology, 21(2), pp. 530–535.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Han, S.S. (2017) Model Onychomycosis Training Datasets (JPG thumbnails) and Validation Datasets (JPG images) [dataset]. figshare. Available at: https://figshare.com/articles/dataset/5398573.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Han, S.S., Park, G.H., Lim, W., Kim, M.S., Na, J.I., Park, I. and Chang, S.E. (2018) ‘Deep neural networks show an equivalent and often superior performance to dermatologists in onychomycosis diagnosis: automatic construction of onychomycosis datasets by region-based convolutional deep neural network’, PLOS ONE, 13(1), e0191493.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'He, K., Zhang, X., Ren, S. and Sun, J. (2016) ‘Deep residual learning for image recognition’, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q.V. and Adam, H. (2019) ‘Searching for MobileNetV3’, Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1314–1324.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M. and Adam, H. (2017) ‘MobileNets: efficient convolutional neural networks for mobile vision applications’, arXiv preprint arXiv:1704.04861.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Howard, J. and Ruder, S. (2018) ‘Universal language model fine-tuning for text classification’, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, pp. 328–339.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Huang, H.N., Zhang, T., Yang, C.T., Sheen, Y.J. and Chen, H.M. (2022) ‘Image segmentation using transfer learning and Fast R-CNN for diabetic foot wound treatments’, Frontiers in Public Health, 10, pp. 1–15.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'International Working Group on the Diabetic Foot (2023) IWGDF Guidelines on the Prevention and Management of Diabetic Foot Disease. Amsterdam: IWGDF.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Islam, M.M. (2026) Lower Limb and Feet Wound Image Dataset for Medical Analysis, version 3 [dataset]. Mendeley Data. DOI: 10.17632/hsj38fwnvr.3.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Krizhevsky, A., Sutskever, I. and Hinton, G.E. (2012) ‘ImageNet classification with deep convolutional neural networks’, Advances in Neural Information Processing Systems, 25, pp. 1097–1105.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'LeCun, Y., Bengio, Y. and Hinton, G. (2015) ‘Deep learning’, Nature, 521(7553), pp. 436–444.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Lipner, S.R. and Scher, R.K. (2019) ‘Onychomycosis: clinical overview and diagnosis’, Journal of the American Academy of Dermatology, 80(4), pp. 835–851.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Lipsky, B.A., Senneville, É., Abbas, Z.G., Aragón-Sánchez, J., Diggle, M., Embil, J.M., Kono, S., Lavery, L.A., Malone, M., van Asten, S.A., Urbančič-Rovan, V. and Peters, E.J.G. (2020) ‘Guidelines on the diagnosis and treatment of foot infection in persons with diabetes (IWGDF 2019 update)’, Diabetes/Metabolism Research and Reviews, 36(S1), e3280.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Litjens, G., Kooi, T., Bejnordi, B.E., Setio, A.A.A., Ciompi, F., Ghafoorian, M., van der Laak, J.A.W.M., van Ginneken, B. and Sánchez, C.I. (2017) ‘A survey on deep learning in medical image analysis’, Medical Image Analysis, 42, pp. 60–88.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Oltu, B., Karaca, B.K., Erdem, H. and Özgür, A. (2021) ‘A systematic review of transfer learning based approaches for diabetic retinopathy detection’, arXiv preprint.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Pan, S.J. and Yang, Q. (2010) ‘A survey on transfer learning’, IEEE Transactions on Knowledge and Data Engineering, 22(10), pp. 1345–1359.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M.P. and Ng, A.Y. (2017) ‘CheXNet: radiologist-level pneumonia detection on chest X-rays with deep learning’, arXiv preprint arXiv:1711.05225.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Sait, A.R.W. and Nagaraj, R. (2025) ‘Diabetic foot ulcers detection model using a hybrid convolutional neural networks–vision transformers approach’, Diagnostics, 15(6), pp. 1–24.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.-C. (2018) ‘MobileNetV2: inverted residuals and linear bottlenecks’, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4510–4520.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2017) ‘Grad-CAM: visual explanations from deep networks via gradient-based localization’, Proceedings of the IEEE International Conference on Computer Vision, pp. 618–626.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Shorten, C. and Khoshgoftaar, T.M. (2019) ‘A survey on image data augmentation for deep learning’, Journal of Big Data, 6(60), pp. 1–48.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Tan, M. and Le, Q.V. (2019) ‘EfficientNet: rethinking model scaling for convolutional neural networks’, Proceedings of the 36th International Conference on Machine Learning, pp. 6105–6114.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Topol, E.J. (2019) ‘High-performance medicine: the convergence of human and artificial intelligence’, Nature Medicine, 25(1), pp. 44–56.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Wang, C., Anisuzzaman, D.M., Williamson, V., Dhar, M.K., Rostami, B., Niezgoda, J., Gopalakrishnan, S. and Yu, Z. (2020) ‘Fully automatic wound segmentation with deep convolutional neural networks’, Scientific Reports, 10, 21897.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Wang, C., Mahbod, A., Ellinger, I., Galdran, A., Gopalakrishnan, S., Niezgoda, J. and Yu, Z. (2022) ‘FUSeg: the foot ulcer segmentation challenge’, arXiv preprint arXiv:2201.00414.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'World Health Organization (2021) Global Strategy on Digital Health 2020–2025. Geneva: World Health Organization.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'World Health Organization and World Bank (2023) Tracking Universal Health Coverage: 2023 Global Monitoring Report. Geneva: World Health Organization and World Bank.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Yap, M.H., Cassidy, B., Pappachan, J.M., O’Shea, C., Gillespie, D. and Reeves, N. (2021) ‘Analysis towards classification of infection and ischaemia of diabetic foot ulcers’, arXiv preprint.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 140, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Zhang, J., Xia, Y., Xie, Y., Fulham, M. and Feng, D.D. (2018) ‘Classification of medical images in the biomedical literature by jointly using deep and handcrafted visual features’, IEEE Journal of Biomedical and Health Informatics, 22(5), pp. 1521–1530.', size: 21, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { before: 320, after: 200, line: 300 },
  shading: { type: ShadingType.CLEAR, fill: 'FFF4E5' },
  children: [ new TextRun({ text: 'FOUR CITATIONS NEED A DECISION.  ', bold: true, size: 20, color: 'B26500', font: 'Calibri' }),
    new TextRun({ text: 'Every other citation in the dissertation now has an entry above. These four could not be matched to a source with confidence, so nothing has been guessed. A citation with no reference loses marks; a reference to a work that does not exist loses more, so settle each one before you submit and then delete this box.', size: 20, italics: true, font: 'Calibri' }) ] }));
children.push(new Paragraph({ spacing: { after: 120, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Patel et al. (2017), cited in section 2.3 — no source matching this citation could be found. Replace it with a work you have read, or delete the citation.', size: 21, italics: true, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 120, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Zhang et al. (2018), cited in section 2.3 — the entry in the list is the closest match found. Confirm it is the work you meant.', size: 21, italics: true, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 120, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Wang et al. (2022), cited in section 2.5.6 — the entry in the list is the closest match found, but it is about segmentation while the sentence cites it for classification accuracy. Confirm or re-point it.', size: 21, italics: true, font: 'Calibri' })] }));
children.push(new Paragraph({ spacing: { after: 120, line: 300 },
  indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: 'Gupta et al. (2022) is cited twice, in sections 2.5.2 and 2.5.6. The entry in the list fits the first, on laboratory diagnosis; it does not support the second, on transfer learning architectures. Re-point the second citation, and complete the author list from the journal page.', size: 21, italics: true, font: 'Calibri' })] }));

// ---------------------------------------------------------- appendices ----
children.push(new Paragraph({ pageBreakBefore: true, heading: HeadingLevel.HEADING_1, spacing: { after: 240 },
  children: [new TextRun({ text: 'Appendices', bold: true, size: 32, color: ACCENT, font: 'Calibri' })] }));
children.push(...appendices);

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
