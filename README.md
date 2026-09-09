# Symmetric Siamese Networks for Longitudinal Chest Radiograph Change Detection: A Leakage-Controlled Study on MIMIC-CXR
Code for a study of per-finding change detection from longitudinal chest radiograph pairs. A Siamese DenseNet-121 takes a patient's earliest and latest frontal radiographs and predicts, for each of 14 CheXpert findings, a four-class transition — absent→absent, onset (0→1), resolved (1→0), or persistent (1→1) — and is compared against a matched single-image baseline under patient-grouped cross-validation and three leakage-controlled encoder-pretraining regimes.

This repository contains the analysis code only. It does not contain any images, labels, or patient data. The data must be obtained separately from PhysioNet.

## Data
This project uses MIMIC-CXR-JPG v2.1.0, distributed by PhysioNet to credentialed users under the PhysioNet Credentialed Health Data Use Agreement. Access requires registration as a credentialed user, completion of the required human-subjects research training, and acceptance of the data use agreement.

No credentialed data is redistributed here.

## Repository Contents
| File | Role |
|------|------|
| preprocess.py | Manifests + 320px image cache; writes raw label per finding |
| build_pairs.py | Longitudinal pairs, four-class transitions, shared CV fold map |
| train_encoder.py | Single-image DenseNet-121 encoder (disjoint / fold-nested) |
| train_change_model.py | Siamese and single-image change models; saves per-fold predictions |
| analyze_results.py | Per-finding metrics, cross-seed summary, Siamese-vs-baseline |
| analyze_stats_bootstrap.py | Patient-level bootstrap CIs, AUPRC, Holm/BH correction |
| analyze_dedicated_tasks.py | The direct test of the value of the prior image |
| analyze_projection_control.py | AP/PA projection-change control |
| analyze_time_stratified.py | Stratification by inter-study interval |
| analyze_radiologist_agreement.py | Agreement with the single-radiologist reference |
| analyze_unmentioned_sensitivity.py | Unmentioned-vs-negative label sensitivity |
| make_figure_auroc_gain.py | Onset-vs-resolution AUROC-gain figure |
| list_transition_examples.py | List example pairs per transition state |
| explain_change.py | Paired and change-pathway Grad-CAM |
| train_encoder_policy_sweep.py | Soft-target sensitivity sweep (optional) |

## Citation
If you use this code, please cite the accompanying paper:

- Işık, Ş.; Eren, H.A. Symmetric Siamese Networks for Longitudinal Chest Radiograph Change Detection: A Leakage-Controlled Study on MIMIC-CXR. Tomography 2026, 12, 129. https://doi.org/10.3390/tomography12090129

and the MIMIC-CXR / MIMIC-CXR-JPG data sources:

- Johnson, A.E.W.; Pollard, T.J.; Berkowitz, S.J.; Greenbaum, N.R.; Lungren, M.P.; Deng, C.; Mark, R.G.; Horng, S. MIMIC-CXR, a De-Identified Publicly Available Database of Chest Radiographs with Free-Text Reports. Sci. Data 2019, 6, 317. https://doi.org/10.1038/s41597-019-0322-0.
- Johnson, A.; Lungren, M.; Peng, Y.; Lu, Z.; Mark, R.; Berkowitz, S.; Horng, S. MIMIC-CXR-JPG, Chest Radiographs with Structured Labels, Version 2.1.0; PhysioNet: Cambridge, MA, USA, 2024. https://doi.org/10.13026/jsn5-t979.
- Pollard, T.; Moody, B.E.; Lehman, L.; Gow, B.; Fernandes, C.; Xie, C.; Johnson, A.; Mark, R.G.; Heldt, T. PhysioNet as a Global Platform for Biomedical Research. Nat. Health 2026, 1, 792–795. https://doi.org/10.1038/s44360-026-00096-z.
