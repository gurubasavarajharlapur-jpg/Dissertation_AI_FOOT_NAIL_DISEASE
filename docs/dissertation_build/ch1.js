const { P, TODO, CH, H2, H3, T, cap } = require('./lib');
const ch1 = [];
ch1.push(...CH(1, 'Introduction'));

ch1.push(H2('1.1  Background'));
ch1.push(P('Foot and nail conditions are among the most commonly overlooked health problems in rural communities. Agricultural workers in particular spend long hours in conditions that place their feet at risk: moisture, soil, minor injuries and limited opportunity for hygiene. Many of the resulting problems begin mildly, as a fungal nail infection, a small wound or a patch of broken skin, and many would resolve with early attention.'));
ch1.push(P('They frequently do not receive it. Access to healthcare in rural areas is limited, specialist consultation is rarer still, and awareness of which early symptoms matter is low. A condition that would have been straightforward to treat in its first weeks is often first presented when it has become serious. Foot ulceration is the clearest example of this pattern and the most consequential: it precedes the large majority of non-traumatic lower limb amputations, and its outcomes depend heavily on how early it is identified.'));
ch1.push(P('At the same time, mobile phones with usable cameras are widespread in precisely the communities where clinical services are scarce. This creates an opportunity that did not exist a decade ago. If a photograph taken on an ordinary phone could indicate that a foot needs professional attention, the barrier to seeking that attention would fall substantially, not because the photograph provides a diagnosis but because it provides a reason to act.'));
ch1.push(P('Advances in deep learning make that technically plausible. Convolutional neural networks have been applied successfully to a range of image-based medical tasks, and transfer learning allows such models to be trained from modest datasets by adapting networks already trained on general photographic data. Chapter 2 reviews this literature in detail.'));

ch1.push(H2('1.2  Problem statement'));
ch1.push(P('The gap this project addresses is not whether deep learning can classify foot and nail conditions, which the existing literature already demonstrates. It is whether it can do so under the constraints that a rural screening application actually imposes.'));
ch1.push(P('Those constraints are specific. The model must run on a phone rather than a server, which favours small architectures. Its output must be trustworthy enough to act on, which means a confidence value that means what it says. It must behave safely when it is unsure, because a confidently wrong reassurance to someone with an ulcer is worse than no tool at all. And it must have been shown to classify the condition rather than some incidental property of the dataset it was trained on, because a model that has learned to recognise which clinic took the photograph will fail the moment it meets a photograph from anywhere else.'));
ch1.push(P('Chapter 2 finds that the published work in this area addresses these constraints only partially. Comparative evaluation between architectures is often inconsistent, deployment suitability is rarely measured alongside accuracy, interpretability is frequently absent, and external validation on independently captured data is uncommon. This project is organised around those gaps.'));

ch1.push(H2('1.3  Aim and objectives'));
ch1.push(P('The aim of this project is to develop and evaluate a transfer learning based image classification system capable of supporting the early detection of common foot and nail conditions in rural healthcare settings.'));
ch1.push(P('Five objectives support that aim:'));
ch1.push(cap('1.1', 'Research objectives and where each is addressed.'));
ch1.push(T([5000, 3200], [
  ['Objective', 'Addressed in'],
  ['1. To identify and prepare publicly available image datasets representing common foot and nail conditions.', 'Chapters 3 and 5'],
  ['2. To preprocess and organise image data into predefined classification categories.', 'Chapters 3 and 4'],
  ['3. To develop an image classification model using transfer learning techniques.', 'Chapters 3, 4 and 5'],
  ['4. To evaluate the performance of the proposed model using standard classification metrics.', 'Chapter 5'],
  ['5. To develop a proof-of-concept prototype demonstrating the practical application of the proposed solution.', 'Chapter 4'],
]));
ch1.push(P('The system classifies images into four categories: Healthy Foot or Nail, Nail Fungal Infection, Foot Wound or Injury, and Foot Ulcer. These four were fixed at the proposal stage and are retained throughout.'));

ch1.push(H2('1.4  Scope'));
ch1.push(P('The work is a proof of concept. It uses publicly available research datasets. The only images collected for this project are ten photographs of five consenting adults with healthy feet, taken on a mobile phone for the external validation reported in Section 5.9. The system is positioned throughout as an assistive screening aid intended to encourage earlier consultation, and explicitly not as a diagnostic device or a substitute for professional assessment.'));
ch1.push(P('Two boundaries should be stated at the outset. First, although the work is motivated by rural healthcare, no data from a rural or field setting was used: all three source datasets are clinical collections. The claim this dissertation supports is that the approach is feasible on clinical photographs and efficient enough for phone deployment, not that it has been validated in the setting it is designed for. Second, the four categories do not cover the range of conditions that present in practice, and the system has no way to indicate that an image falls outside them. That limitation is measured rather than merely stated: Section 5.10 reports what the system does with 578 photographs of a real nail condition it has no category for.'));

ch1.push(H2('1.5  Contribution'));
ch1.push(P('The contribution of this work is methodological as much as technical. Four elements go beyond building a classifier and reporting its accuracy.'));
ch1.push(P('The two architectures are compared as a controlled experiment, sharing data, splits, preprocessing, augmentation and training procedure, with the difference between them assessed using a paired statistical test rather than by placing two accuracy figures side by side.'));
ch1.push(P('Three independent tests examine whether the models classify the condition or the dataset, including a controlled occlusion ablation in which the informative comparison is against an equal-area control rather than against the unmodified image.'));
ch1.push(P('Confidence values are calibrated and a referral threshold is derived, so that the system can decline to answer, and a triage policy converts a classification into a recommended action that escalates rather than reassures when a serious condition is plausible.'));
ch1.push(P('Efficiency is measured alongside accuracy on the device class the application targets, so that the trade-off between the two is quantified rather than assumed.'));
ch1.push(P('And the system is tested outside the data it was built on, in both of the ways that matters: on photographs of the four conditions taken on a phone rather than in a clinic, and on a real condition that falls outside its four classes. Both results qualify the accuracy figures materially, and reporting them is itself part of the contribution, because Chapter 2 finds that the published work in this area largely does not.'));

ch1.push(H2('1.6  Structure of the dissertation'));
ch1.push(P('Chapter 2 reviews the literature on artificial intelligence for medical image classification, the clinical background to foot and nail conditions, and the specific research gaps this project addresses.'));
ch1.push(P('Chapter 3 sets out the methodology: the datasets, how they were prepared and split, the preprocessing and augmentation applied, the two architectures and the transfer learning procedure, the evaluation metrics, the design of the tests for dataset shortcuts, the calibration and referral policy, and the statistical methods used.'));
ch1.push(P('Chapter 4 describes how the system was implemented, including the structure of the software, the prototype interface, and the problems encountered during development together with the verification steps introduced in response.'));
ch1.push(P('Chapter 5 reports the results: training, test set performance, the statistical comparison between architectures, computational cost, calibration, the three tests for dataset shortcuts, the behaviour of the referral policy, the external validation on photographs taken outside the source datasets, and what the system does with a condition outside its four classes.'));
ch1.push(P('Chapter 6 interprets those results against the research objectives and the reviewed literature, makes the case for the recommended architecture, and states the limitations that qualify the findings.'));
ch1.push(P('Chapter 7 concludes and sets out what further work would be needed to move from a proof of concept toward something usable in the setting that motivated it.'));
module.exports = ch1;
