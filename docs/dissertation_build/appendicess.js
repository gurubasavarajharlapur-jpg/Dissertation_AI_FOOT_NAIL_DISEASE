const { P, H2, cap, T, BREAK } = require('./lib');
const c = [];

c.push(BREAK());
c.push(H2('Appendix A  —  Validation sweep of the escalation threshold'));
c.push(P('The full grid behind the threshold decision in Section 5.7, measured on the 1,261 validation images, of which 200 are foot ulcers with 11 misclassified and 685 are cases requiring no urgent care. The test set was not used for this and could not be, having already been read.'));
c.push(cap('A.1', 'Validation sweep of the urgent escalation threshold.'));
//@TABLEDATA 5.14
c.push(P('The cost column never approaches the 2 percent budget defined for the sweep at any value on the grid, which is why the selection rule reduced to maximising benefit. The adopted value of 0.10 is the least aggressive threshold reaching the best attainable benefit.'));

c.push(BREAK());
c.push(H2('Appendix B  —  External validation, every photograph'));
c.push(P('The ten photographs described in Section 5.8, scored with MobileNetV2 after temperature scaling. The final column applies the abstention threshold of 0.787 derived on validation. Subject 2’s two photographs were uploaded with their background names transposed; the table below is corrected.'));
c.push(cap('B.1', 'Every photograph in the external validation set.'));
//@TABLEDATA 5.16
c.push(P('The full calibrated probability vector for each image is written alongside these columns by the prototype’s batch tab, which is what allows the paired analysis in Section 5.8.3 to be recomputed without re-running inference.'));

c.push(BREAK());
c.push(H2('Appendix C  —  Per-class recall from the border alone'));
c.push(P('The fourth arm of the occlusion ablation in Section 5.6.3, which masks the interior and leaves only a 20-pixel frame. It measures how much class information each model can recover from the region a lesion does not occupy.'));
c.push(cap('C.1', 'Per-class recall with the interior masked and only a 20-pixel border visible.'));
//@TABLEDATA 5.13
c.push(P('Neither model recovers anything on the nail class. MobileNetV2 collapses into predicting healthy for almost everything, recovering 62 of 692 non-healthy images for a margin of five points over the majority-class floor of 0.4455; ResNet50 recovers 193 for a margin of fifteen points. Degrading into the majority class rather than into a confident diagnosis is the behaviour wanted from a screening tool.'));

c.push(BREAK());
c.push(H2('Appendix D  —  Source code'));
c.push(P('The complete implementation, including the configuration module listing every hyper-parameter, the analysis scripts, the prototype interface and the scripts that generated the figures in this dissertation, is in the project repository. The repository also contains the findings document from which every figure quoted in Chapter 5 is taken.'));

c.push(BREAK());
c.push(H2('Appendix E  —  Problems encountered during implementation'));
c.push(P('Each of these produced a wrong result that looked correct rather than an error message, which is why the fix in every case is a verification step rather than a code correction. They are recorded because a study reporting only its successes gives a reader no way to judge how carefully it was checked.'));
c.push(cap('E.1', 'Problems encountered during implementation, their cause, and the verification added in response.'));
c.push(T([2500, 2900, 2800], [
  ['Symptom', 'Cause', 'Fix, and the check added'],
  ['Three dataset archives downloaded as 0 bytes, reporting success', 'A JSON Accept header, then a browser User-Agent, on the download request; the default client returned the correct 1.8 GB payload', 'Send no custom headers; verify byte size and archive readability before treating a fetch as complete'],
  ['Archives reported missing from a drive that contained them', 'The Colab drive mount is case-sensitive and the export ships Normal.zip while the code looked for normal.zip', 'Resolve targets against the lower-cased names actually present'],
  ['An entire session of results lost', 'Drive buffers writes, and the runtime was recycled before they flushed', 'Write single archives and verify them, rather than syncing directories'],
  ['Test set contained no healthy nail images at all, unnoticed', 'Splitting stratified on class alone sent all twenty montage-tile groups into training', 'Stratify on class and source; preprocessing now fails if any source is absent from any split'],
  ['Best fine-tuning weights overwritten by worse ones', 'Stage 2 checkpointing started from no baseline, so its first epoch counted as best even when stage 1 had done better', 'Pass the stage 1 best validation loss in as the starting baseline'],
  ['Border ablation appeared to prove robustness where it could not discriminate', 'Masking anything removes information, so a small drop means nothing alone; both arms hit chance on a border-only fixture', 'Score an equal-area interior control and a centre-masked arm alongside, and detect the floor case explicitly'],
  ['A screening report headed URGENT whose advice read that no concerning features were found', 'Escalation raised the band but the guidance text still came from the top predicted class', 'The escalating condition owns the advice; verified by enumerating every distinct outcome'],
], { }));
c.push(P('Two deserve a further note. The border ablation appeared to prove robustness on a fixture where it could not discriminate at all, which is why the equal-area interior control and the explicit floor check were added. And the self-contradicting report was found by rendering the report rather than by reading the code: the classifier and the triage rules were each behaving correctly, and only the combination was wrong; after the fix, all 3,542 probability vectors producing a distinct outcome were enumerated and checked, with no violation found.'));

module.exports = c;
