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
  children: [new Paragraph({ spacing: { after: 0 },
    children: [new TextRun({ text: t, bold: o.head, size: 18, font: 'Calibri' })] })] });
const T = (widths, rows) => new Table({ columnWidths: widths,
  width: { size: widths.reduce((a,b)=>a+b,0), type: WidthType.DXA },
  rows: rows.map((r,i)=> new TableRow({ tableHeader: i===0,
    children: r.map((cc,j)=> cell(String(cc), widths[j], { head: i===0 })) })) });

const c = [];
c.push(new Paragraph({ spacing: { after: 280 },
  children: [new TextRun({ text: 'Chapter 4', bold: true, size: 36, color: ACCENT, font: 'Calibri' })] }));
c.push(new Paragraph({ spacing: { after: 360 },
  children: [new TextRun({ text: 'Design and Implementation', bold: true, size: 36, color: ACCENT, font: 'Calibri' })] }));

c.push(P('Chapter 3 described what was decided. This chapter describes how it was built: the structure of the software, how each stage of the pipeline was implemented, how the prototype works, and what went wrong during development. The last of these is given a section of its own, because several of the problems encountered produced incorrect results without producing any error message, and the checks introduced in response are part of why the results in Chapter 5 can be relied on.'));

// ---- 4.1 -----------------------------------------------------------------
c.push(H2('4.1  System architecture'));
c.push(P('The system is organised as a set of standalone command-line scripts sharing a single configuration module, with a web interface built on top. It is not structured as a library, because nothing here is imported by an external application: each script is an entry point that performs one stage of the work and writes its output to disk for the next stage to read.'));
c.push(P('The configuration module is the centre of the design. Every file path, hyper-parameter, split ratio, class name, augmentation setting, calibration parameter and triage threshold is defined there and imported by everything else. No script contains a hard-coded experimental value. This has two consequences that matter for a comparison study. Changing a setting means editing one file, so the two architectures cannot accidentally be trained under different conditions through an overlooked constant. And because the module validates itself when imported, an inconsistent configuration stops the program immediately instead of failing part way through an hour of training.'));
c.push(P('The validation performed at import is deliberately strict. It checks that every class has an entry in each of the lookup tables that depend on the class list, that the split ratios sum to one, that every model named has a fine-tuning depth defined, that the triage thresholds lie between zero and one, and that the foot ulcer class maps to the most urgent triage band. That last check exists because it encodes a clinical requirement rather than a programming one, and a future edit that broke it would otherwise be silent.'));
c.push(P('Each script begins by inserting the project root into the module search path before importing the configuration. This allows any script to run unchanged from the repository root, from inside the source directory, or from a notebook cell, which matters because the work was carried out partly on a local machine and partly in Google Colab.'));
c.push(cap('4.1', 'Software modules and their responsibilities.'));
c.push(T([2500, 5700], [
  ['Module', 'Responsibility'],
  ['config.py', 'Every setting used anywhere in the project; validates itself on import'],
  ['download_data.py', 'Retrieves the source datasets and verifies each download'],
  ['inspect_data.py', 'Audits the raw data and reports folder contents for human class mapping'],
  ['extract_montages.py', 'Cuts the montage contact sheets into individual nail images'],
  ['preprocessing.py', 'Cleans, deduplicates, resizes, splits and verifies the dataset'],
  ['train.py', 'Two-stage transfer learning, shared by both architectures'],
  ['evaluate.py', 'Test metrics, confusion matrices, Grad-CAM, efficiency, model comparison'],
  ['calibrate.py', 'Temperature scaling and selection of the abstention threshold'],
  ['ablate_border.py', 'Controlled occlusion test for dataset shortcuts'],
  ['compare_models.py', 'Paired significance testing from saved predictions'],
  ['audit_triage.py', 'Measures what the referral policy does to the test set'],
  ['sweep_thresholds.py', 'Tunes the escalation thresholds on the validation set'],
  ['stats.py', 'Shared statistical functions, free of any deep learning dependency'],
  ['triage.py', 'Converts a probability vector into a recommended action'],
  ['colab_sync.py', 'Saves and restores work between Colab sessions'],
  ['prototype/app.py', 'The two-tab screening interface'],
]));
c.push(FIG('System architecture diagram: configuration at the centre, scripts as stages, prototype on top'));
c.push(fcap('4.1', 'System architecture. Arrows show the flow of data between stages.'));

// ---- 4.2 -----------------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('4.2  Data pipeline'));
c.push(H3('4.2.1  Acquisition'));
c.push(P('The download script retrieves each dataset from its published location and checks the result before treating it as complete. The check compares the size of the downloaded file against the size the server reported, and confirms that each archive can be opened and read. This is stricter than it may appear necessary, for reasons described in Section 4.6.'));
c.push(P('The inspection script then walks the downloaded directories and reports what each folder contains: how many files, of what types, at what image sizes, with sample filenames. This report is what a person reads before deciding which folder maps to which class. The mapping is not inferred automatically, because folder names in public datasets are inconsistent and an automatic mapping would produce silently mislabelled data.'));
c.push(H3('4.2.2  Montage extraction'));
c.push(P('The healthy nail images arrive as contact sheets, each containing a grid of many small photographs. Cutting them into individual images required detecting the grid, which is harder than it sounds because the sheets vary in the number of cells and in whether they have visible separators.'));
c.push(P('The extraction routine works in two stages. It first looks for rows and columns of near-uniform pixels running the full width or height of the sheet, which indicates a separator band between cells. When a sheet has no clean separators, it falls back to analysing the edge profile of the image: it computes the density of detected edges along each axis and looks for a repeating period, which corresponds to the cell pitch. A dry-run mode reports the grid it would use without writing anything, and a preview mode writes a small sample so the detection can be checked by eye before committing to thousands of files. A manual cell size can be supplied when both methods fail.'));
c.push(H3('4.2.3  Preprocessing and verification'));
c.push(P('The preprocessing script performs the sequence described in Chapter 3: it maps folders to classes, excludes what does not belong, removes duplicates by content hash, crops the black padding from the ulcer images, resizes with aspect preservation and a centre crop, caps the montage contribution by sampling evenly across source sheets, and assigns each image to a split stratified by class and source.'));
c.push(P('It then verifies its own output, and refuses to finish if any of four conditions fails. No group may appear in more than one split, which prevents images from the same photographic session straddling the train and test boundary. Every split must contain every class. Every split must contain something from every source, because a source confined to one split leaves its sub-population unmeasured, and this is precisely the failure that produced a test set with no healthy nail images. And every written image must be the expected size on disk.'));
c.push(P('A dry-run mode reports exactly what would be written, with counts at each stage, without modifying anything. This was used before every real run.'));
c.push(P('The script writes three CSV files, one per split, each listing the relative path, class label, source and group for every image. Those files are the single definition of the split, and every later stage reads them rather than re-deriving anything.'));

// ---- 4.3 -----------------------------------------------------------------
c.push(H2('4.3  Training'));
c.push(P('A single training script handles both architectures. The backbone and its matching input normalisation are selected from a lookup table keyed by model name, and everything else — the data pipeline, the classification head, the optimiser, the callbacks, the class weights, the two-stage schedule — is shared code. This is what makes the comparison controlled in practice rather than only in intention: there is no separate code path in which the two models could diverge.'));
c.push(P('Input pipelines are built with the TensorFlow data API. Images are read from disk, decoded, batched and prefetched, with augmentation applied as preprocessing layers inside the model so that it runs on the accelerator and is automatically disabled at inference time. Keras 3, which ships with TensorFlow 2.16, removed the older image generator API, so this is the current idiom rather than a stylistic preference.'));
c.push(P('The normalisation each backbone expects is applied as the first operation inside the model graph. This was verified numerically rather than assumed: pooled activations produced by feeding raw images into the model were compared against activations from images normalised externally, and agreed to within 2 × 10⁻⁵.'));
c.push(P('Three callbacks control training. Early stopping monitors validation loss with a patience of eight epochs and restores the best weights when it fires. Learning rate reduction halves the rate after four epochs without improvement. Model checkpointing writes the weights whenever validation loss improves.'));
c.push(P('The checkpointing required a correction that is worth describing, because the bug was invisible in every output. The second training stage created a fresh checkpoint callback with no knowledge of what the first stage had achieved. Its internal record of the best result therefore began empty, so the first fine-tuning epoch was always treated as an improvement and its weights were saved, even when head training had already produced a lower validation loss. In that situation the better weights were silently overwritten by worse ones, and nothing in the logs indicated it. The fix was to pass the first stage best validation loss into the second stage checkpoint as its starting threshold. The behaviour was reproduced deliberately with a synthetic case before and after the change to confirm the fix worked.'));
c.push(P('On completion the script writes the trained model, the per-epoch history including the size of the validation split used, and a plot of the training curves. Recording the validation size in the history file allows a later stage to detect that a model was trained on a different split from the one currently on disk, which is checked before any evaluation runs.'));

// ---- 4.4 -----------------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('4.4  Evaluation and analysis'));
c.push(P('Evaluation is split across several scripts rather than concentrated in one, for a reason that turned out to matter more than expected. The scripts that require a trained model and a GPU are separated from those that only need the saved predictions.'));
c.push(P('The evaluation script loads each trained model, scores the test set once, and produces the metrics, confusion matrices, Grad-CAM figures, efficiency measurements and the comparison between models. Before it does any of this it checks that each model was trained on the split currently on disk, by comparing the validation size recorded in the training history against the current validation file, and stops with an explanatory message if they differ. This prevents the most damaging possible mistake in a study of this kind, which is reporting metrics from a model trained on a different split without noticing.'));
c.push(P('Grad-CAM is implemented by building a secondary model that exposes both the final convolutional feature maps and the class scores, computing the gradient of the score for the predicted class with respect to those feature maps, pooling those gradients to obtain a weight per channel, and forming a weighted sum. Examples are selected to include misclassified predictions as well as correct ones.'));
c.push(P('Efficiency is measured with a warm-up phase before timing, because the first calls to a model include graph tracing and compilation and would otherwise dominate the measurement. Latency is reported as the median of fifty timed runs at batch size one, on CPU and on GPU separately.'));
c.push(P('Crucially, the evaluation script writes the complete probability vector for every test image to a CSV file, not merely the predicted class. Every policy layered on top of the classifier — the abstention threshold, the triage escalation rules — is a function of that whole vector. Saving it means those policies can be audited, re-tuned and re-measured in seconds on a laptop, without a GPU, without TensorFlow and without repeating inference. Three of the analysis scripts depend on this and would otherwise each have required a full evaluation run.'));
c.push(P('The calibration script fits the temperature on validation logits by minimising negative log likelihood, measures expected calibration error over fifteen bins before and after, verifies that no prediction changed, computes the risk-coverage curve and selects the abstention threshold. It also saves the validation predictions in the same format, which is what allows the threshold sweep to run without a GPU.'));
c.push(P('The remaining analysis scripts read those saved predictions. The comparison script performs the paired significance tests and the bootstrap interval, and refuses to run if the two prediction files do not describe the same images — checking length, file paths and true labels — because two files from different splits would silently invalidate every paired test. The triage audit measures what the referral policy does to the test set. The threshold sweep tunes the escalation thresholds on validation, and is written so that it reads only the validation predictions and cannot accidentally touch the test set.'));
c.push(P('A design decision worth noting is that the sweep evaluates candidate thresholds by calling the same triage function the prototype uses, with the candidate value passed in as an argument, rather than reimplementing the rule. A second implementation could drift from the deployed one and would then be tuning something other than what ships.'));

// ---- 4.5 -----------------------------------------------------------------
c.push(H2('4.5  The prototype'));
c.push(P('What the proposal committed to at this stage was a prototype interface demonstrating the trained model, not a deployable mobile application, and that is what was built. The interface is implemented in Streamlit and runs in a browser, which keeps the demonstration in the same Python environment as the rest of the pipeline and allows it to load the trained model directly rather than requiring the model to be converted and packaged for a phone. Packaging for a phone is a deployment exercise that the efficiency measurements in Chapter 5 are intended to inform, and it is listed as future work rather than claimed here.'));
c.push(P('The interface has two tabs, serving two different purposes. The first demonstrates the system on a single image as an end user would experience it. The second scores a labelled collection of images and reports the same metrics used in Chapter 5, which is what makes the external validation in Section 5.9 possible without writing additional code.'));
c.push(H3('4.5.1  Screening tab'));
c.push(P('The user uploads a photograph, or selects one from the test split for demonstration. The image is preprocessed exactly as the training pipeline preprocesses images, scored by the selected model, and the resulting probabilities are temperature-scaled using the value fitted during calibration.'));
c.push(P('The result is presented as a structured report. The order of the report is a deliberate design decision: the recommended action and its timeframe appear first, in a coloured banner, before the predicted condition. Somebody reading this on a phone in a clinic may not read past the first line, so the first line is what they should do. The predicted class, the calibrated confidence, the reason the recommendation was reached, the guidance for that condition, and the text stating what should send them to seek care regardless all follow beneath.'));
c.push(P('Several details of this presentation are constrained deliberately. Confidence is never displayed as 100 percent, since a rounded 0.9997 shown as certainty overclaims, particularly in a dissertation that establishes the model needed calibration at all. When the model abstains, no condition is named at all rather than a low-confidence guess being shown with a caveat. And when a result has been escalated by the triage rules, the guidance shown is that of the escalating condition rather than the predicted class. That last point was a defect during development and is described in Section 4.6.'));
c.push(P('The report is deliberately not styled to resemble a clinician’s report. It carries no signature, no practitioner name and no letterhead, and the statement that the system is not a medical device appears inside the report rather than beneath it. A document that could be mistaken for a genuine medical record would be a worse outcome than an unattractive one.'));
c.push(P('A Grad-CAM overlay is shown below the report, so that the user can see which region of their photograph drove the result.'));
c.push(H3('4.5.2  Batch evaluation tab'));
c.push(P('The second tab scores either the project test split or a compressed archive of the user’s own photographs organised into folders by class. It reports accuracy, per-class precision, recall and F1, a confusion matrix, and the proportion of cases answered rather than abstained.'));
c.push(P('Two accommodations were added after testing the archive upload. Folder names are matched after normalising case and separators, so that "Foot Ulcer", "foot-ulcer" and "foot_ulcer" all resolve to the same class; the original exact match silently discarded every image in a folder named with a capital letter. And files in a recognised class folder whose format is not supported are counted and reported by extension rather than skipped in silence, because mobile phones commonly save images in a format the pipeline cannot read, and a user uploading fifty such files would otherwise see only an empty result with no explanation.'));
c.push(P('Because the sets scored here are small, the tab reports a confidence interval alongside the accuracy rather than the percentage alone, and warns when fewer than thirty images are supplied. On a set of that size the interval is the honest figure: nineteen correct out of twenty displays as 95 percent but is equally consistent with a true accuracy of 76 percent.'));
c.push(P('The tab also reports every image individually, carrying the filename through from the uploaded archive, and writes that table to a file alongside the aggregate metrics. Each row holds the filename, the true and predicted class, whether the prediction was correct, the confidence and the full calibrated probability vector. On a small set the aggregate accuracy is the least informative part of the result: which images failed, and how confidently, is what can be reasoned about, and preserving filenames is what allows a paired design such as the one in Section 5.9 to be analysed as pairs rather than as a pool of independent images.'));
c.push(FIG('Screenshot of the screening tab showing a completed report and Grad-CAM overlay'));
c.push(fcap('4.2', 'The screening tab, showing the automated report and the Grad-CAM overlay.'));
c.push(FIG('Screenshot of the batch evaluation tab showing metrics and confusion matrix'));
c.push(fcap('4.3', 'The batch evaluation tab, scoring a labelled set of photographs.'));

// ---- 4.6 -----------------------------------------------------------------
c.push(new Paragraph({ children: [new PageBreak()] }));
c.push(H2('4.6  Problems encountered and how they were resolved'));
c.push(P('This section describes six problems found during implementation. They are grouped together because they share a property that shaped the design of the whole system: every one of them produced an incorrect result without producing an error. Nothing crashed, no warning was printed, and in each case the output looked entirely reasonable. The response in every case was to add a verification step rather than simply to correct the immediate fault, and those verification steps are now part of the pipeline.'));
c.push(H3('4.6.1  Downloads that reported success and delivered nothing'));
c.push(P('Three of the source archives downloaded as zero-byte files while the download code reported success. The cause was traced to the request headers: sending a browser-like user agent caused the server to return an empty response with a success status. Removing the custom headers resolved it. The fix, however, was not only to remove the headers but to verify every download against its expected size and confirm that each archive opens and reads, so that a silent failure of any kind is caught at the point it happens rather than several stages later.'));
c.push(H3('4.6.2  A session of results lost to buffered writes'));
c.push(P('An early approach linked the working directories directly to cloud storage so that outputs would persist between Colab sessions. Writes to that storage are buffered, and a session was recycled before the buffer flushed, losing a full set of training outputs. The approach was replaced with an explicit save and restore mechanism that packs each directory into a single archive, copies it, and verifies it, which removes the dependency on buffering behaviour outside the program’s control.'));
c.push(H3('4.6.3  A test set containing none of one class'));
c.push(P('As described in Chapter 3, stratifying the split by class alone placed every montage tile group in the training set, leaving no healthy nail images in the test set. Healthy nail performance was therefore unmeasured, and no metric revealed this: overall accuracy, per-class recall and the confusion matrix all appeared normal, because the class was still represented by photographs of whole feet. The fix was to stratify by class and source together, and to add the verification described in Section 4.2.3 that refuses to complete a split in which any source is absent from any part.'));
c.push(H3('4.6.4  Better weights overwritten by worse ones'));
c.push(P('The checkpointing defect described in Section 4.3, in which the second training stage discarded better weights from the first. Reproduced deliberately with a synthetic case to confirm both the fault and the fix.'));
c.push(H3('4.6.5  An ablation that passed where it could not discriminate'));
c.push(P('The first version of the occlusion test masked the image border and observed that accuracy barely changed, which appeared to show that the border was unimportant. It showed nothing of the kind, because masking any small region costs little. Adding the equal-area interior control made the comparison interpretable. A second problem then emerged: on a test fixture where the border was the only informative region, both the border arm and the control arm fell to chance, and the difference between them was therefore meaningless while still appearing to be a pass. A fourth arm masking the centre was added to detect that situation explicitly.'));
c.push(H3('4.6.6  A report that contradicted itself'));
c.push(P('When the triage escalation rules were added, the screening report could produce an urgent banner above guidance reading that no concerning features had been detected. The cause was that the banner came from the escalated triage band while the guidance still came from the predicted class. For a healthy prediction escalated on ulcer probability, the two disagreed completely, and the reassuring sentence was the one a reader would act on. This was found by rendering the report and reading it rather than by inspecting the code, which had looked correct. The escalating condition now supplies the guidance, and the property was then checked exhaustively across 3,542 probability vectors to confirm that no urgent or prompt report can carry reassuring guidance.'));
c.push(P('Taken together these are the reason the results in Chapter 5 are accompanied by verification steps rather than presented on their own. A pipeline that fails loudly can be debugged. A pipeline that fails silently produces a dissertation full of numbers that are wrong in ways nobody can see.'));

// ---- 4.7 -----------------------------------------------------------------
c.push(H2('4.7  Summary'));
c.push(P('The system is built as a set of single-purpose scripts around one configuration module that validates itself, with a web prototype on top. The data pipeline verifies its own output before anything downstream trusts it. A single training script serves both architectures, so the controlled comparison is enforced by the structure of the code rather than by discipline. Evaluation is deliberately split so that the statistical analysis and the referral policy can be re-examined from saved predictions without a GPU.'));
c.push(P('Six defects found during development are described above, all of which produced wrong results silently. Each was resolved by adding a check rather than only a correction, and those checks now run as part of the pipeline.'));
c.push(P('Chapter 5 reports what the completed system produced.'));

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
  fs.writeFileSync(require('path').join(__dirname, 'Chapter4_Implementation_DRAFT.docx'), b);
  console.log('written', b.length, 'bytes');
});
