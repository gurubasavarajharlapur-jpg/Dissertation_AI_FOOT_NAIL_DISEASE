const { P, CH, H2, T, cap } = require('./lib');
const c = [];
c.push(...CH(1, 'Introduction'));

c.push(H2('1.1  Background'));
c.push(P('Foot and nail conditions are among the most commonly overlooked health problems in rural communities. Agricultural workers in particular spend long hours in conditions that place their feet at risk: moisture, soil, minor injuries and limited opportunity for hygiene. Many of the resulting problems begin mildly and would resolve with early attention.'));
c.push(P('They frequently do not receive it. Access to healthcare in rural areas is limited, specialist consultation rarer still, and awareness of which early symptoms matter is low, so a condition that would have been straightforward to treat in its first weeks is often first presented once it has become serious. Foot ulceration is the clearest example and the most consequential: it precedes the large majority of non-traumatic lower limb amputations, and its outcomes depend heavily on how early it is identified (Armstrong, Boulton and Bus, 2017). At the same time phones with usable cameras are widespread in precisely the communities where clinical services are scarce, and advances in transfer learning make a photograph-based screening aid technically plausible.'));

c.push(H2('1.2  Problem statement'));
c.push(P('The gap this project addresses is not whether deep learning can classify foot and nail conditions, which the existing literature demonstrates. It is whether it can do so under the constraints a rural screening tool imposes: the model must run on a phone rather than a server, its confidence must mean what it says because the output has to be acted on, it must behave safely when unsure because a confidently wrong reassurance to somebody with an ulcer is worse than no tool at all, and it must be shown to classify the condition rather than an incidental property of the dataset it was trained on.'));
c.push(P('Chapter 2 finds that published work addresses these constraints only partially: comparative evaluation between architectures is inconsistent, deployment suitability is rarely measured alongside accuracy, interpretability is frequently absent, and external validation on independently captured data is uncommon. This project is organised around those four gaps.'));

c.push(H2('1.3  Aim and objectives'));
c.push(P('The aim is to develop and evaluate a transfer learning based image classification system capable of supporting the early detection of common foot and nail conditions in rural healthcare settings. Five objectives support it.'));
//@TABLE 1.1
c.push(P('The system classifies images into four categories: Healthy Foot or Nail, Nail Fungal Infection, Foot Wound or Injury, and Foot Ulcer. These were fixed at the proposal stage and are retained throughout.'));

c.push(H2('1.4  Scope'));
c.push(P('The work is a proof of concept built on public research datasets. The only images collected for it are ten photographs of five consenting adults with healthy feet, taken on mobile phones for the external validation in Section 5.8. The system is positioned throughout as an assistive screening aid rather than a diagnostic device. Two boundaries should be stated at the outset. No data from a rural or field setting was used, so the claim supported is feasibility on clinical photographs and efficiency sufficient for phone deployment rather than validation in the setting the system is designed for. And the four categories do not cover the range of conditions that present in practice; Section 5.9 measures what the system does with a real condition outside them.'));

c.push(H2('1.5  Contribution'));
c.push(P('The contribution is methodological as much as technical. The two architectures are compared as a controlled experiment and the difference between them assessed by a paired statistical test rather than by placing two accuracy figures side by side. Three independent tests examine whether the models classify the condition or the dataset, including an occlusion ablation whose informative comparison is against an equal-area control. Confidence is calibrated and a referral threshold derived, and a triage policy converts a classification into a recommended action that escalates rather than reassures when a serious condition is plausible. Efficiency is measured alongside accuracy on the device class the application targets. And the system is tested outside the data it was built on in both of the ways that matter: on photographs taken on a phone, and on a real condition outside its four classes.'));

c.push(H2('1.6  Structure of the dissertation'));
c.push(P('Chapter 2 reviews the literature and identifies the research gaps. Chapter 3 sets out the methodology: datasets, preparation and splitting, preprocessing and augmentation, the two architectures and the transfer learning procedure, evaluation metrics, the tests for dataset shortcuts, calibration, the referral policy, the statistical methods, and two tests that go beyond the source datasets. Chapter 4 describes the implementation and the problems encountered. Chapter 5 reports the results, Chapter 6 interprets them and states the limitations, and Chapter 7 concludes.'));

module.exports = c;
