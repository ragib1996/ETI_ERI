# Beyond Accuracy: Explainable Trustworthy Index and Explainable Robustness Index for Evaluating CNN

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical research evaluation framework introducing two complementary model-level metrics that integrate predictive performance with visual explanation quality for Convolutional Neural Networks (CNNs):
- **ETI (Explainable Trustworthy Index)**: Combines classification accuracy with the **Localization Mask Overlap Score (LMOS)** to evaluate whether predictions are grounded on the target object rather than background or irrelevant regions under clean, unperturbed conditions.
- **ERI (Explainable Robustness Index)**: Quantifies how well a model preserves both classification accuracy and object-focused explanation alignment under input perturbations relative to its clean-image baseline.

This repository provides **107 end-to-end reproducible Jupyter Notebooks** benchmarking **8 CNN architectures** across **4 real-world leaf and flower image datasets** subjected to **9 physical and digital image perturbations**.

---

## Authors & Affiliations

- **Ragib Mehedi**<sup>1,*</sup> &nbsp;[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--1555--7338-green.svg)](https://orcid.org/0009-0003-1555-7338) &nbsp;(`ragib6627@gmail.com`)  
  *Conceptualization, Methodology, Software, Investigation, Formal analysis, Writing – Original Draft, Writing – Review & Editing*
- **Azmain Inquaid Haque**<sup>1</sup> &nbsp;[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--4032--2622-green.svg)](https://orcid.org/0009-0009-4032-2622) &nbsp;(`azmaininquaidhaque@gmail.com`)  
  *Investigation, Data Curation, Software, Validation*
- **Md. Nazmul Abdal**<sup>2</sup> &nbsp;[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--5233--0920-green.svg)](https://orcid.org/0009-0001-5233-0920) &nbsp;(`nazmul.abdal@ulab.edu.bd`)  
  *Data Curation*
- **Abdullah Al Noman**<sup>1,*</sup> &nbsp;[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7537--1402-green.svg)](https://orcid.org/0000-0001-7537-1402) &nbsp;(`noman@cse.ku.ac.bd`)  
  *Supervision, Writing – Review & Editing*
- **Md. Zahidul Islam**<sup>1</sup> &nbsp;[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--9209--9524-green.svg)](https://orcid.org/0000-0002-9209-9524) &nbsp;(`zahid@cse.ku.ac.bd`)  
  *Supervision, Writing – Review & Editing*

<sup>1</sup> **Computer Science and Engineering Discipline, Khulna University**, Khulna 9208, Bangladesh  
<sup>2</sup> **Department of Computer Science and Engineering, University of Liberal Arts Bangladesh**, Dhaka 1207, Bangladesh  
<sup>*</sup> *Corresponding authors*

---

## Table of Contents

- [Abstract & Research Highlights](#abstract--research-highlights)
- [Motivation & Problem Definition](#motivation--problem-definition)
- [System Overview](#system-overview)
- [Methodology & Mathematical Formulations](#methodology--mathematical-formulations)
  - [1. Localization Mask Overlap Score (LMOS)](#1-localization-mask-overlap-score-lmos)
  - [2. Justification & Selection of Threshold $\tau$](#2-justification--selection-of-threshold-tau)
  - [3. Explainable Trustworthy Index (ETI)](#3-explainable-trustworthy-index-eti)
  - [4. Explainable Robustness Index (ERI)](#4-explainable-robustness-index-eri)
- [Taxonomy of Evaluated Perturbations](#taxonomy-of-evaluated-perturbations)
- [Datasets & Benchmark Architectures](#datasets--benchmark-architectures)
  - [Datasets & Annotations](#datasets--annotations)
  - [Evaluated CNN Architectures](#evaluated-cnn-architectures)
  - [Training & Optimization Configurations](#training--optimization-configurations)
- [Empirical Benchmark Results](#empirical-benchmark-results)
  - [Dataset 1: EPLI (Egyptian Plant Leaves Identification)](#dataset-1-epli-egyptian-plant-leaves-identification)
  - [Dataset 2: BDML (Bangladeshi Medicinal Leaves)](#dataset-2-bdml-bangladeshi-medicinal-leaves)
  - [Dataset 3: FLRS (TensorFlow Flowers)](#dataset-3-flrs-tensorflow-flowers)
  - [Dataset 4: FR (Flowers Recognition)](#dataset-4-fr-flowers-recognition)
  - [Comprehensive ETI Evaluation & Ablation Study](#comprehensive-eti-evaluation--ablation-study)
  - [Comprehensive ERI Evaluation & Ablation Study](#comprehensive-eri-evaluation--ablation-study)
  - [Perturbation-Specific Robustness Analysis ($ERI^p$)](#perturbation-specific-robustness-analysis-erip)
  - [Accuracy Stability ($S_{\text{acc}}$) & LMOS Stability ($S_{\text{LMOS}}$) Breakdown](#accuracy-stability-s_textacc--lmos-stability-s_textlmos-breakdown)
  - [Key Findings & Cross-Dataset Discussion](#key-findings--cross-dataset-discussion)
- [Repository Organization](#repository-organization)
- [Installation & Environment Setup](#installation--environment-setup)
- [Workflow & Reproduction Guide](#workflow--reproduction-guide)
- [Data and Code Availability](#data-and-code-availability)
- [Citation](#citation)
- [License & Acknowledgments](#license--acknowledgments)

---

## Abstract & Research Highlights

Convolutional neural networks (CNNs) can achieve high classification accuracy while grounding their predictions on background or irrelevant image regions. Furthermore, external input perturbations can alter both what the models predict and what evidence they consider for making the predictions. Accuracy-based evaluation captures neither problem. We propose two complementary model-level metrics that integrate predictive performance with explanation quality:
- The **Explainable Trustworthy Index (ETI)** combines classification accuracy with the **Localization Mask Overlap Score (LMOS)**, the proportion of salient Grad-CAM activation that falls within the ground-truth object mask.
- The **Explainable Robustness Index (ERI)** measures how well a model preserves both accuracy and LMOS under input perturbations, relative to its behavior on clean images.

### Key Highlights
- **Novel Trustworthiness and Robustness Metrics**: Proposed ETI and ERI to assess CNN trustworthiness and robustness beyond standalone predictive accuracy.
- **Localization Mask Overlap Score (LMOS)**: Introduced a Grad-CAM-based score linking decision-relevant model focus to ground-truth object regions with optimal binarization threshold $\tau = 0.70$.
- **Comprehensive Benchmark**: Evaluated 8 CNN architectures on 4 real-world leaf and flower image datasets under 9 common physical and sensor perturbations.
- **High Accuracy Does Not Guarantee Reliability**: On EPLI, Custom CNN attains **99.45% accuracy**, yet only **19%** of its salient activation falls on the leaf, yielding the lowest ETI (**0.5922**). Across all datasets, DenseNet121 achieves the highest accuracy but the highest ETI on none.
- **Xception Consistently Leads in Robustness**: Xception obtains the highest ERI on all four datasets (**0.9346** on EPLI, **0.8982** on BDML, **0.9308** on FLRS, and **0.9689** on FR).
- **Divergence of Predictive and Explanation Stability**: Under contrast perturbation on EPLI, Custom CNN largely preserves its accuracy while its object focus shifts substantially, whereas ResNet50 exhibits the reverse pattern.

---

## Motivation & Problem Definition

Evaluating CNN models requires asking not only whether predictions are correct, but whether they are grounded on the target object rather than spurious background cues.

```
       Accuracy-based Evaluation                   ETI & ERI Evaluation Framework
+------------------------------------+          +------------------------------------+
|  Prediction: 99.45% Accuracy       |          |  Prediction: 99.45% Accuracy       |
|  Status: Appears "Accurate"        |   VS     |  Object Focus (LMOS): Only 19.00%  |
|  Blind Spot: Relying on background |          |  ETI: 0.5922 (Penalized)           |
|  Shortcut Learning / Spurious Cues |          |  ERI: 0.7944 (Unstable under noise)|
+------------------------------------+          +------------------------------------+
```

### Two Fundamental Questions:
1. **ETI Question**: *"Is the model accurate and appropriately focused on the target object under normal conditions?"*
2. **ERI Question**: *"Do these properties (predictive performance and object-focused explanations) remain stable when the input is perturbed?"*

### Formal Mathematical Problem Setting
Let $\mathcal{D} = \{(x_i, y_i, M_i)\}_{i=1}^N$ be a dataset containing $N$ test samples, where $x_i \in \mathbb{R}^{n \times m}$ is the $i$-th image, $y_i$ is its ground-truth class label, and $M_i \in \{0, 1\}^{n \times m}$ is the corresponding ground-truth binary mask of the target object.
- Classification accuracy alone, $acc \in [0, 1]$, does not indicate whether decision-relevant activations are spatially aligned with $M_i$.
- Let $H_i$ denote the resized Grad-CAM heatmap for image $x_i$ and prediction $\hat{y}_i$.
- $\text{LMOS}_i \in [0, 1]$ measures the ratio between decision-relevant regions of $H_i$ and ground-truth mask $M_i$.
- **Problem 1 (Trustworthiness)**: Formulate $T = \Psi(acc, \overline{\text{LMOS}})$, solved by **ETI**.
- **Problem 2 (Robustness under Perturbations $\mathcal{P}$)**: Formulate $R = \Phi(S_{\text{acc}}, S_{\text{LMOS}})$, solved by **ERI**.

---

## System Overview

```
                                    +-----------------------+
                                    |     Image Datasets    |
                                    | (EPLI, BDML, FLRS, FR)|
                                    +-----------+-----------+
                                                |
                         +----------------------+----------------------+
                         |                                             |
                         v                                             v
              +---------------------+                       +---------------------+
              |   Training & Val    |                       |     Test Images     |
              |     (80% / 10%)     |                       |        (10%)        |
              +----------+----------+                       +----------+----------+
                         |                                             |
                         v                                     +-------+-------+
              +---------------------+                          |               |
              | 8 CNN Architectures |                          v               v
              |  Model Training     |                   [Clean Images]  [Perturbed Images]
              +----------+----------+                          |          (9 Modalities)
                         |                                     |               |
                         +------------------+------------------+               |
                                            |                                  |
                                            v                                  v
                                  +-------------------+              +-------------------+
                                  | Model Predictions |              | Model Predictions |
                                  |    & Grad-CAM     |              |    & Grad-CAM     |
                                  +---------+---------+              +---------+---------+
                                            |                                  |
                                            v                                  v
                                  +-------------------+              +-------------------+
                                  |    Binary Mask    |              |    Binary Mask    |
                                  |    (tau = 0.70)   |              |    (tau = 0.70)   |
                                  +---------+---------+              +---------+---------+
                                            |                                  |
                                            v                                  v
                                  +-------------------+              +-------------------+
                                  |   Calculate LMOS  |              | Calculate LMOS_p  |
                                  | (Target Overlap)  |              | & Accuracy_p      |
                                  +---------+---------+              +---------+---------+
                                            |                                  |
                                            v                                  v
                                  +===================+              +===================+
                                  |        ETI        |              |        ERI        |
                                  |  Trustworthiness  |              | Robustness Index  |
                                  |  (acc + LMOS) / 2 |              |  (S_acc + S_L) / 2|
                                  +===================+              +===================+
```

The pipeline operates across four unified stages:
1. **Dataset Preparation & Mask Annotation**: Test sets are manually annotated with polygon boundaries converted into binary segmentation masks isolating target leaves and flowers from backgrounds.
2. **Model Training**: 8 CNN architectures are trained under unified configurations.
3. **Explanation-Based Assessment (LMOS & ETI)**: Grad-CAM generates heatmaps resized to model input resolution, binarized at $\tau = 0.70$, and compared with ground-truth masks to yield LMOS and ETI.
4. **Perturbation Stress-Testing & ERI**: Test images are transformed across 9 perturbation modalities to calculate Accuracy Stability ($S_{\text{acc}}$), LMOS Stability ($S_{\text{LMOS}}$), and ERI.

---

## Methodology & Mathematical Formulations

### 1. Localization Mask Overlap Score (LMOS)

Grad-CAM generates class-discriminative heatmaps from the final convolutional layer of a network, resized to the input image dimensions $h \times w$ ($299 \times 299$ for Xception, $224 \times 224$ for other architectures). Let $H \in \mathbb{R}^{h \times w}$ denote the resized heatmap.

A binary attention map $\hat{H}(i, j)$ is obtained by thresholding:
$$\hat{H}(i, j) = \mathds{1}\bigl(H(i, j) > \tau\bigr)$$
where $\mathds{1}(\cdot)$ is the indicator function and $\tau = 0.70$.

**LMOS** is defined as the ratio of activated decision-relevant pixels inside the ground-truth target object mask $M \in \{0, 1\}^{h \times w}$ to the total activated decision-relevant pixels:
$$\mathrm{LMOS} = \frac{\sum_{i=1}^h \sum_{j=1}^w \hat{H}(i, j)\, M(i, j)}{\sum_{i=1}^h \sum_{j=1}^w \hat{H}(i, j)}$$

### 2. Justification & Selection of Threshold $\tau$

In Grad-CAM visualizations, red and yellow regions denote highest activation levels and dominant influence on predictions. The parameter $\tau \in \{0.5, 0.6, 0.7, 0.8\}$ was systematically tested:
- **$\tau = 0.5$**: Retains broad portions of the heatmap, capturing green peripheral regions and admitting irrelevant background cues.
- **$\tau = 0.6$**: Suppresses most green activations, but still permits peripheral, weakly activated regions alongside core features.
- **$\tau = 0.8$**: Overly stringent; preserves predominantly red areas while discarding valid, decision-relevant yellow activations.
- **$\tau = 0.7$ (Selected)**: Achieves optimal balance by isolating core red and yellow activations representing primary decision-relevant evidence while filtering background noise. Applied consistently across all architectures, datasets, and perturbations.

### 3. Explainable Trustworthy Index (ETI)

ETI integrates global predictive classification accuracy ($acc$) with the mean localized explanation quality ($\overline{\mathrm{LMOS}}$) across test samples:
$$\mathrm{ETI} = \frac{acc + \overline{\mathrm{LMOS}}}{2}$$

Both components are normalized to $[0, 1]$ and assigned equal weighting ($0.5 / 0.5$) to evaluate predictive accuracy and explanation fidelity symmetrically.

**ETI Ablation Variants**:
$$\mathrm{ETI}_{\text{w/o LMOS}} = \frac{acc}{2}, \quad \mathrm{ETI}_{\text{w/o acc}} = \frac{\overline{\mathrm{LMOS}}}{2}$$

### 4. Explainable Robustness Index (ERI)

ERI evaluates whether a model's predictive accuracy and visual explanation focus remain consistent under input perturbations relative to clean images.

#### A. Accuracy Stability ($S_{\text{acc}}$)
For perturbation $p \in \mathcal{P}$:
$$S_{\mathrm{acc}}^p = 1 - \frac{|acc_c - acc_p|}{acc_c + acc_p + \epsilon}$$
where $acc_c$ is clean accuracy, $acc_p$ is perturbed accuracy, and $\epsilon = 10^{-7}$ prevents division by zero. The overall accuracy stability across all evaluated perturbations $|\mathcal{P}| = 9$ is:
$$S_{\mathrm{acc}} = \frac{1}{|\mathcal{P}|} \sum_{p=1}^{|\mathcal{P}|} S_{\mathrm{acc}}^p$$

#### B. LMOS Stability ($S_{\text{LMOS}}$)
For test sample $i$ under perturbation $p$:
$$S_{\mathrm{LMOS}}^{p, i} = 1 - \frac{|\mathrm{LMOS}_c^i - \mathrm{LMOS}_p^i|}{\mathrm{LMOS}_c^i + \mathrm{LMOS}_p^i + \epsilon}$$
Averaged across all $N$ test samples:
$$S_{\mathrm{LMOS}}^p = \frac{1}{N} \sum_{i=1}^N S_{\mathrm{LMOS}}^{p, i}$$
The overall LMOS stability across all evaluated perturbations is:
$$S_{\mathrm{LMOS}} = \frac{1}{|\mathcal{P}|} \sum_{p=1}^{|\mathcal{P}|} S_{\mathrm{LMOS}}^p$$

#### C. ERI per Perturbation ($\mathrm{ERI}^p$) & Overall ERI
$$\mathrm{ERI}^p = \frac{S_{\mathrm{acc}}^p + S_{\mathrm{LMOS}}^p}{2}$$
$$\mathrm{ERI} = \frac{S_{\mathrm{acc}} + S_{\mathrm{LMOS}}}{2}$$

**ERI Ablation Variants**:
$$\mathrm{ERI}_{\text{w/o LMOS}} = \frac{S_{\mathrm{acc}}}{2}, \quad \mathrm{ERI}_{\text{w/o acc}} = \frac{S_{\mathrm{LMOS}}}{2}$$

---

## Taxonomy of Evaluated Perturbations

All 9 perturbations simulate realistic degradations encountered during image capture, transmission, and real-world deployment. Fixed parameters ensure cross-model reproducibility:

| Perturbation Modality | Parameters & Formulation | Physical & Environmental Simulation |
| :--- | :--- | :--- |
| **Gaussian Blur** | $5 \times 5$ kernel | Defocus blur, lens imperfections, optical distortion |
| **Gaussian Noise** | Additive Gaussian, $\mu = 0$, $\sigma = 0.05$ | Sensor thermal noise, low-light ISO noise |
| **Brightness Adjustment**| Constant $+0.2$ added to normalized pixels, clipped $[0, 1]$ | Overexposure, intense sunlight, direct illumination |
| **Hue Shift** | HSV hue channel shifted by $+15$ units | Illumination shifts, twilight/dawn, camera color cast |
| **Contrast Adjustment** | $I' = (I - \mu_I) \times 0.5 + \mu_I$, clipped $[0, 1]$ | Fog, atmospheric haze, dynamic range compression |
| **Gamma Correction** | $I' = I^{2.0}$ ($\gamma = 2.0$) | Non-linear sensor response, shadowed underexposure |
| **Motion Blur** | $7 \times 7$ horizontal averaging kernel | Handheld jitter, camera motion, wind-induced motion |
| **JPEG Compression** | Quality factor = 30 | Lossy digital storage, bandwidth-constrained transfer |
| **Random Occlusion** | Centered/random patch ($20\%$ height $\times$ $20\%$ width) set to 0 | Physical obstructions, overlapping leaves, dirt/debris |

---

## Datasets & Benchmark Architectures

### Datasets & Annotations

Experiments evaluate 4 real-world image datasets encompassing botanical leaf and floral domains:

| Dataset | Domain | Samples | Resolution | Classes / Target Species | Splitting Protocol |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **EPLI** (EgyPLI) | Leaf Identification | 3,588 | $256 \times 256$ | 8 species (apple, berry, fig, guava, orange, plum, persimmon, tomato) | 80% Train, 10% Val, 10% Test |
| **BDML** (BDMediLeaves)| Medicinal Leaves | 2,048 | $3024 \times 4032$ | 10 indigenous medicinal plant species (iPhone 13, natural light) | Pre-split (Train / Val / Test) |
| **FLRS** (TF Flowers) | Flower Classification | 3,670 | Variable | 5 species (daisy, dandelion, rose, sunflower, tulip) | 80% Train, 10% Val, 10% Test |
| **FR** (Flowers Recognition)| Flower Recognition | 4,317 | $320 \times 240$ | 5 species (daisy, dandelion, rose, sunflower, tulip) | 80% Train, 10% Val, 10% Test |

- **Ground-Truth Masks**: All test samples were manually annotated with polygonal boundaries saved as JSON files and converted into binary masks representing strictly the target leaf or flower, excluding backgrounds.

### Evaluated CNN Architectures
1. **MobileNetV2**: Inverted residual blocks with linear bottlenecks and depthwise separable convolutions.
2. **Xception**: Extreme Inception with depthwise separable convolutions explicitly separating spatial and channel-wise learning.
3. **DenseNet121**: Dense connectivity where each layer receives feature maps from all preceding layers within a dense block.
4. **InceptionResNetV2**: Multi-branch Inception modules coupled with residual shortcut connections.
5. **VGG16**: Deep uniform architecture using stacked $3 \times 3$ convolutional filters and pooling layers.
6. **InceptionV3**: Factorized convolutions, auxiliary classifiers, and efficient Inception grid reductions.
7. **ResNet50**: Residual learning through shortcut connections bypassing intermediate convolutional layers.
8. **Custom CNN**: Multi-layer convolutional architecture trained from scratch following the original EPLI study.

### Training & Optimization Configurations
- **Environment**: TensorFlow/Keras on Kaggle with 2 $\times$ NVIDIA Tesla T4 GPUs.
- **Weights & Backbones**: Custom CNN trained from scratch; 7 architectures initialized with ImageNet-pretrained weights with backbone layers frozen.
- **Classification Head**: Global Average Pooling (GAP) $\to$ Dropout (rate 0.3) $\to$ Dense (128 ReLU units) $\to$ Softmax.
- **Optimizer & Loss**: Adam optimizer ($lr = 5 \times 10^{-5}$), Categorical Cross-Entropy loss.

---

## Empirical Benchmark Results

### Dataset 1: EPLI (Egyptian Plant Leaves Identification)

| Rank (ETI) | Model | Clean Accuracy | $\overline{\text{LMOS}}$ | **ETI** | $S_{\text{acc}}$ | $S_{\text{LMOS}}$ | **ERI** | Rank (ERI) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **MobileNetV2** | 0.9945 | **0.7765** | **0.8855** | 0.9704 | 0.8404 | 0.9054 | 2 |
| 2 | **Xception** | 0.9945 | 0.7638 | 0.8791 | **0.9887** | **0.8805** | **0.9346** | 1 |
| 3 | **DenseNet121** | **1.0000** | 0.6964 | 0.8482 | 0.9849 | 0.8251 | 0.9050 | 3 |
| 4 | **InceptionResNetV2** | 0.9889 | 0.5529 | 0.7709 | 0.9766 | 0.7463 | 0.8614 | 4 |
| 5 | **VGG16** | 0.9834 | 0.5268 | 0.7551 | 0.9469 | 0.7033 | 0.8251 | 7 |
| 6 | **InceptionV3** | 0.9917 | 0.3617 | 0.6767 | 0.9703 | 0.6961 | 0.8332 | 5 |
| 7 | **ResNet50** | 0.6122 | 0.7256 | 0.6689 | 0.8263 | 0.8281 | 0.8272 | 6 |
| 8 | **Custom CNN** | 0.9945 | 0.1900 | 0.5922 | 0.9302 | 0.6586 | 0.7944 | 8 |

---

### Dataset 2: BDML (Bangladeshi Medicinal Leaves)

| Rank (ETI) | Model | Clean Accuracy | $\overline{\text{LMOS}}$ | **ETI** | $S_{\text{acc}}$ | $S_{\text{LMOS}}$ | **ERI** | Rank (ERI) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **MobileNetV2** | 0.9703 | 0.7611 | **0.8657** | 0.9523 | 0.7840 | 0.8682 | 3 |
| 2 | **Xception** | 0.9505 | 0.7800 | 0.8653 | 0.9471 | 0.8493 | **0.8982** | 1 |
| 3 | **DenseNet121** | **0.9802** | 0.7118 | 0.8460 | **0.9665** | 0.8130 | 0.8897 | 2 |
| 4 | **VGG16** | 0.8960 | 0.5873 | 0.7416 | 0.9526 | 0.6988 | 0.8257 | 5 |
| 5 | **InceptionResNetV2** | 0.9059 | 0.5184 | 0.7122 | 0.9552 | 0.6969 | 0.8260 | 4 |
| 6 | **InceptionV3** | 0.9307 | 0.3344 | 0.6325 | 0.9544 | 0.6253 | 0.7898 | 7 |
| 7 | **ResNet50** | 0.4604 | **0.7817** | 0.6211 | 0.7553 | **0.8565** | 0.8059 | 6 |
| 8 | **Custom CNN** | 0.7376 | 0.3310 | 0.5343 | 0.7864 | 0.5556 | 0.6710 | 8 |

---

### Dataset 3: FLRS (TensorFlow Flowers)

| Rank (ETI) | Model | Clean Accuracy | $\overline{\text{LMOS}}$ | **ETI** | $S_{\text{acc}}$ | $S_{\text{LMOS}}$ | **ERI** | Rank (ERI) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Xception** | 0.8828 | **0.6869** | **0.7849** | **0.9876** | **0.8741** | **0.9308** | 1 |
| 2 | **DenseNet121** | **0.9046** | 0.6188 | 0.7617 | 0.9707 | 0.8286 | 0.8997 | 2 |
| 3 | **MobileNetV2** | 0.8665 | 0.5958 | 0.7311 | 0.9741 | 0.7942 | 0.8842 | 5 |
| 4 | **VGG16** | 0.8147 | 0.6311 | 0.7229 | 0.9723 | 0.8214 | 0.8968 | 3 |
| 5 | **InceptionResNetV2** | 0.8501 | 0.5581 | 0.7041 | 0.9783 | 0.8018 | 0.8900 | 4 |
| 6 | **InceptionV3** | 0.8229 | 0.3749 | 0.5989 | 0.9860 | 0.7326 | 0.8593 | 7 |
| 7 | **ResNet50** | 0.5123 | 0.5623 | 0.5373 | 0.9064 | 0.8347 | 0.8706 | 6 |
| 8 | **Custom CNN** | 0.7003 | 0.2891 | 0.4947 | 0.9528 | 0.6763 | 0.8146 | 8 |

---

### Dataset 4: FR (Flowers Recognition)

| Rank (ETI) | Model | Clean Accuracy | $\overline{\text{LMOS}}$ | **ETI** | $S_{\text{acc}}$ | $S_{\text{LMOS}}$ | **ERI** | Rank (ERI) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Xception** | 0.9032 | **0.6029** | **0.7531** | **0.9901** | **0.9478** | **0.9689** | 1 |
| 2 | **DenseNet121** | **0.9124** | 0.5382 | 0.7253 | 0.9808 | 0.8978 | 0.9393 | 2 |
| 3 | **VGG16** | 0.8387 | 0.5786 | 0.7087 | 0.9644 | 0.8880 | 0.9262 | 4 |
| 4 | **MobileNetV2** | 0.8986 | 0.5091 | 0.7038 | 0.9758 | 0.8541 | 0.9149 | 5 |
| 5 | **InceptionResNetV2** | 0.8894 | 0.4705 | 0.6800 | 0.9811 | 0.8736 | 0.9273 | 3 |
| 6 | **InceptionV3** | 0.8825 | 0.3063 | 0.5944 | 0.9795 | 0.7858 | 0.8827 | 7 |
| 7 | **ResNet50** | 0.4977 | 0.4769 | 0.4873 | 0.8944 | 0.8960 | 0.8952 | 6 |
| 8 | **Custom CNN** | 0.6820 | 0.2714 | 0.4767 | 0.9563 | 0.7798 | 0.8680 | 8 |

---

### Comprehensive ETI Evaluation & Ablation Study

Comparison of classification accuracy, mean LMOS, ETI, and ablation variants across all benchmark datasets (Table 1 of the paper):

| Dataset | Rank | Model | Acc | $\overline{\text{LMOS}}$ | **ETI** | $\text{ETI}_{\text{w/o Acc}}$ | $\text{ETI}_{\text{w/o LMOS}}$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **EPLI** | 1 | MobileNetV2 | 0.9945 | 0.7765 | **0.8855** | 0.3882 | 0.4972 |
| | 2 | Xception | 0.9945 | 0.7638 | 0.8791 | 0.3819 | 0.4972 |
| | 3 | DenseNet121 | 1.0000 | 0.6964 | 0.8482 | 0.3482 | 0.5000 |
| | 4 | InceptionResNetV2 | 0.9889 | 0.5529 | 0.7709 | 0.2764 | 0.4945 |
| | 5 | VGG16 | 0.9834 | 0.5268 | 0.7551 | 0.2634 | 0.4917 |
| | 6 | InceptionV3 | 0.9917 | 0.3617 | 0.6767 | 0.1809 | 0.4958 |
| | 7 | ResNet50 | 0.6122 | 0.7256 | 0.6689 | 0.3628 | 0.3061 |
| | 8 | Custom CNN | 0.9945 | 0.1900 | 0.5922 | 0.0950 | 0.4972 |
| **BDML** | 1 | MobileNetV2 | 0.9703 | 0.7611 | **0.8657** | 0.3805 | 0.4851 |
| | 2 | Xception | 0.9505 | 0.7800 | 0.8653 | 0.3900 | 0.4752 |
| | 3 | DenseNet121 | 0.9802 | 0.7118 | 0.8460 | 0.3559 | 0.4901 |
| | 4 | VGG16 | 0.8960 | 0.5873 | 0.7416 | 0.2936 | 0.4480 |
| | 5 | InceptionResNetV2 | 0.9059 | 0.5184 | 0.7122 | 0.2592 | 0.4530 |
| | 6 | InceptionV3 | 0.9307 | 0.3344 | 0.6325 | 0.1672 | 0.4653 |
| | 7 | ResNet50 | 0.4604 | 0.7817 | 0.6211 | 0.3909 | 0.2302 |
| | 8 | Custom CNN | 0.7376 | 0.3310 | 0.5343 | 0.1655 | 0.3688 |
| **FLRS** | 1 | Xception | 0.8828 | 0.6869 | **0.7849** | 0.3435 | 0.4414 |
| | 2 | DenseNet121 | 0.9046 | 0.6188 | 0.7617 | 0.3094 | 0.4523 |
| | 3 | MobileNetV2 | 0.8665 | 0.5958 | 0.7311 | 0.2979 | 0.4332 |
| | 4 | VGG16 | 0.8147 | 0.6311 | 0.7229 | 0.3155 | 0.4074 |
| | 5 | InceptionResNetV2 | 0.8501 | 0.5581 | 0.7041 | 0.2791 | 0.4251 |
| | 6 | InceptionV3 | 0.8229 | 0.3749 | 0.5989 | 0.1875 | 0.4114 |
| | 7 | ResNet50 | 0.5123 | 0.5623 | 0.5373 | 0.2812 | 0.2561 |
| | 8 | Custom CNN | 0.7003 | 0.2891 | 0.4947 | 0.1446 | 0.3501 |
| **FR** | 1 | Xception | 0.9032 | 0.6029 | **0.7531** | 0.3015 | 0.4516 |
| | 2 | DenseNet121 | 0.9124 | 0.5382 | 0.7253 | 0.2691 | 0.4562 |
| | 3 | VGG16 | 0.8387 | 0.5786 | 0.7087 | 0.2893 | 0.4194 |
| | 4 | MobileNetV2 | 0.8986 | 0.5091 | 0.7038 | 0.2545 | 0.4493 |
| | 5 | InceptionResNetV2 | 0.8894 | 0.4705 | 0.6800 | 0.2353 | 0.4447 |
| | 6 | InceptionV3 | 0.8825 | 0.3063 | 0.5944 | 0.1531 | 0.4412 |
| | 7 | ResNet50 | 0.4977 | 0.4769 | 0.4873 | 0.2384 | 0.2488 |
| | 8 | Custom CNN | 0.6820 | 0.2714 | 0.4767 | 0.1357 | 0.3410 |

---

### Comprehensive ERI Evaluation & Ablation Study

ERI ranking and ablation breakdown across all benchmark datasets (Table 5 of the paper):

| Dataset | Rank | Model | $S_{\text{acc}}$ | $S_{\text{LMOS}}$ | **ERI** | $\text{ERI}_{\text{w/o acc}}$ | $\text{ERI}_{\text{w/o LMOS}}$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **EPLI** | 1 | Xception | 0.9887 | 0.8805 | **0.9346** | 0.4403 | 0.4944 |
| | 2 | MobileNetV2 | 0.9704 | 0.8404 | 0.9054 | 0.4202 | 0.4852 |
| | 3 | DenseNet121 | 0.9849 | 0.8251 | 0.9050 | 0.4125 | 0.4924 |
| | 4 | InceptionResNetV2 | 0.9766 | 0.7463 | 0.8614 | 0.3731 | 0.4883 |
| | 5 | InceptionV3 | 0.9703 | 0.6961 | 0.8332 | 0.3481 | 0.4852 |
| | 6 | ResNet50 | 0.8263 | 0.8281 | 0.8272 | 0.4140 | 0.4132 |
| | 7 | VGG16 | 0.9469 | 0.7033 | 0.8251 | 0.3517 | 0.4735 |
| | 8 | Custom CNN | 0.9302 | 0.6586 | 0.7944 | 0.3293 | 0.4651 |
| **BDML** | 1 | Xception | 0.9471 | 0.8493 | **0.8982** | 0.4246 | 0.4736 |
| | 2 | DenseNet121 | 0.9665 | 0.8130 | 0.8897 | 0.4065 | 0.4832 |
| | 3 | MobileNetV2 | 0.9523 | 0.7840 | 0.8682 | 0.3920 | 0.4762 |
| | 4 | InceptionResNetV2 | 0.9552 | 0.6969 | 0.8260 | 0.3485 | 0.4776 |
| | 5 | VGG16 | 0.9526 | 0.6988 | 0.8257 | 0.3494 | 0.4763 |
| | 6 | ResNet50 | 0.7553 | 0.8565 | 0.8059 | 0.4283 | 0.3777 |
| | 7 | InceptionV3 | 0.9544 | 0.6253 | 0.7898 | 0.3127 | 0.4772 |
| | 8 | Custom CNN | 0.7864 | 0.5556 | 0.6710 | 0.2778 | 0.3932 |
| **FLRS** | 1 | Xception | 0.9876 | 0.8741 | **0.9308** | 0.4371 | 0.4938 |
| | 2 | DenseNet121 | 0.9707 | 0.8286 | 0.8997 | 0.4143 | 0.4854 |
| | 3 | VGG16 | 0.9723 | 0.8214 | 0.8968 | 0.4107 | 0.4862 |
| | 4 | InceptionResNetV2 | 0.9783 | 0.8018 | 0.8900 | 0.4009 | 0.4891 |
| | 5 | MobileNetV2 | 0.9741 | 0.7942 | 0.8842 | 0.3971 | 0.4871 |
| | 6 | ResNet50 | 0.9064 | 0.8347 | 0.8706 | 0.4174 | 0.4532 |
| | 7 | InceptionV3 | 0.9860 | 0.7326 | 0.8593 | 0.3663 | 0.4930 |
| | 8 | Custom CNN | 0.9528 | 0.6763 | 0.8146 | 0.3382 | 0.4764 |
| **FR** | 1 | Xception | 0.9901 | 0.9478 | **0.9689** | 0.4739 | 0.4950 |
| | 2 | DenseNet121 | 0.9808 | 0.8978 | 0.9393 | 0.4489 | 0.4904 |
| | 3 | InceptionResNetV2 | 0.9811 | 0.8736 | 0.9273 | 0.4368 | 0.4905 |
| | 4 | VGG16 | 0.9644 | 0.8880 | 0.9262 | 0.4440 | 0.4822 |
| | 5 | MobileNetV2 | 0.9758 | 0.8541 | 0.9149 | 0.4270 | 0.4879 |
| | 6 | ResNet50 | 0.8944 | 0.8960 | 0.8952 | 0.4480 | 0.4472 |
| | 7 | InceptionV3 | 0.9795 | 0.7858 | 0.8827 | 0.3929 | 0.4898 |
| | 8 | Custom CNN | 0.9563 | 0.7798 | 0.8680 | 0.3899 | 0.4781 |

---

### Perturbation-Specific Robustness Analysis ($\mathrm{ERI}^p$)

$\mathrm{ERI}^p$ per perturbation modality across all datasets (Table 4 of the paper):

| Dataset | Model | Noise | Blur | Brightness | Hue | Contrast | Gamma | Motion Blur | Compression | Occlusion | Overall ERI |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EPLI** | Custom CNN | 0.8663 | 0.8690 | 0.7182 | 0.7206 | 0.6520 | 0.6543 | 0.8579 | 0.9528 | 0.8589 | 0.7944 |
| | DenseNet121 | 0.9133 | 0.8998 | 0.9676 | 0.9355 | 0.9058 | 0.9157 | 0.7855 | 0.9017 | 0.9201 | 0.9050 |
| | MobileNetV2 | 0.8770 | 0.8867 | 0.9651 | 0.9413 | 0.9142 | 0.9375 | 0.8014 | 0.9155 | 0.9102 | 0.9054 |
| | InceptionResNetV2 | 0.8286 | 0.8403 | 0.9316 | 0.9271 | 0.8905 | 0.8804 | 0.7559 | 0.8660 | 0.8323 | 0.8614 |
| | InceptionV3 | 0.8072 | 0.8077 | 0.9192 | 0.9078 | 0.8661 | 0.8682 | 0.7389 | 0.8025 | 0.7813 | 0.8332 |
| | ResNet50 | 0.9117 | 0.7285 | 0.9644 | 0.9466 | 0.6215 | 0.8494 | 0.6491 | 0.9089 | 0.8650 | 0.8272 |
| | VGG16 | 0.8702 | 0.8355 | 0.9385 | 0.8824 | 0.6871 | 0.8638 | 0.6645 | 0.8818 | 0.8023 | 0.8251 |
| | Xception | 0.9170 | 0.9256 | 0.9734 | 0.9745 | 0.9547 | 0.9380 | 0.8581 | 0.9335 | 0.9368 | **0.9346** |
| **BDML** | Custom CNN | 0.8108 | 0.6328 | 0.6256 | 0.6783 | 0.4748 | 0.6512 | 0.5541 | 0.8556 | 0.7563 | 0.6710 |
| | DenseNet121 | 0.9150 | 0.8281 | 0.9356 | 0.9353 | 0.9051 | 0.9236 | 0.7644 | 0.8967 | 0.9040 | 0.8897 |
| | MobileNetV2 | 0.8918 | 0.8074 | 0.9399 | 0.9312 | 0.8919 | 0.9090 | 0.7109 | 0.8672 | 0.8643 | 0.8682 |
| | InceptionResNetV2 | 0.8084 | 0.7689 | 0.9033 | 0.8772 | 0.8773 | 0.8811 | 0.7329 | 0.7790 | 0.8063 | 0.8260 |
| | InceptionV3 | 0.8054 | 0.7566 | 0.8546 | 0.8606 | 0.8385 | 0.8124 | 0.7072 | 0.7387 | 0.7349 | 0.7898 |
| | ResNet50 | 0.9425 | 0.6077 | 0.8769 | 0.9391 | 0.5982 | 0.8674 | 0.6097 | 0.9033 | 0.9088 | 0.8059 |
| | VGG16 | 0.9189 | 0.8181 | 0.8956 | 0.8677 | 0.7599 | 0.8384 | 0.6251 | 0.8697 | 0.8382 | 0.8257 |
| | Xception | 0.9227 | 0.8133 | 0.9540 | 0.9513 | 0.9483 | 0.9334 | 0.8069 | 0.8575 | 0.8965 | **0.8982** |
| **FLRS** | Custom CNN | 0.8844 | 0.8584 | 0.7938 | 0.7374 | 0.6951 | 0.7805 | 0.8337 | 0.9089 | 0.8387 | 0.8146 |
| | DenseNet121 | 0.9212 | 0.8638 | 0.9415 | 0.8871 | 0.9131 | 0.9031 | 0.8541 | 0.9096 | 0.9035 | 0.8997 |
| | MobileNetV2 | 0.8813 | 0.8583 | 0.9374 | 0.9021 | 0.8738 | 0.9010 | 0.8447 | 0.8626 | 0.8963 | 0.8842 |
| | InceptionResNetV2 | 0.8919 | 0.8627 | 0.9224 | 0.9217 | 0.9109 | 0.9116 | 0.8582 | 0.8824 | 0.8485 | 0.8900 |
| | InceptionV3 | 0.8662 | 0.8275 | 0.9085 | 0.8962 | 0.8897 | 0.8661 | 0.7860 | 0.8338 | 0.8595 | 0.8593 |
| | ResNet50 | 0.9544 | 0.8008 | 0.9253 | 0.9392 | 0.6976 | 0.9095 | 0.7670 | 0.9117 | 0.9300 | 0.8706 |
| | VGG16 | 0.9370 | 0.8930 | 0.9319 | 0.8907 | 0.8308 | 0.9092 | 0.8332 | 0.9359 | 0.9099 | 0.8968 |
| | Xception | 0.9279 | 0.9203 | 0.9557 | 0.9550 | 0.9457 | 0.9427 | 0.8965 | 0.9144 | 0.9191 | **0.9308** |
| **FR** | Custom CNN | 0.9318 | 0.9084 | 0.8468 | 0.8045 | 0.7774 | 0.8469 | 0.9037 | 0.9409 | 0.8519 | 0.8680 |
| | DenseNet121 | 0.9549 | 0.9295 | 0.9682 | 0.9344 | 0.9459 | 0.9500 | 0.8925 | 0.9413 | 0.9369 | 0.9393 |
| | MobileNetV2 | 0.9085 | 0.8765 | 0.9492 | 0.9350 | 0.9240 | 0.9254 | 0.8874 | 0.9056 | 0.9228 | 0.9149 |
| | InceptionResNetV2 | 0.9275 | 0.9114 | 0.9609 | 0.9482 | 0.9470 | 0.9407 | 0.9007 | 0.9279 | 0.8817 | 0.9273 |
| | InceptionV3 | 0.8840 | 0.8613 | 0.9200 | 0.9188 | 0.9029 | 0.8940 | 0.8290 | 0.8749 | 0.8592 | 0.8827 |
| | ResNet50 | 0.9502 | 0.8544 | 0.9470 | 0.9508 | 0.7512 | 0.9381 | 0.8025 | 0.9159 | 0.9468 | 0.8952 |
| | VGG16 | 0.9626 | 0.9176 | 0.9594 | 0.9271 | 0.8702 | 0.9416 | 0.8650 | 0.9583 | 0.9339 | 0.9262 |
| | Xception | 0.9667 | 0.9715 | 0.9789 | 0.9821 | 0.9791 | 0.9775 | 0.9453 | 0.9585 | 0.9608 | **0.9689** |

---

### Accuracy Stability ($S_{\text{acc}}$) & LMOS Stability ($S_{\text{LMOS}}$) Breakdown

#### Accuracy Stability ($S_{\text{acc}}$) Across Perturbations (Table 2 of the paper)
- **EPLI**: Xception leads ($0.9887$), followed closely by DenseNet121 ($0.9849$) and InceptionResNetV2 ($0.9766$). ResNet50 exhibits the lowest stability ($0.8263$), dropping sharply under contrast ($0.4599$) and motion blur ($0.5316$).
- **BDML**: DenseNet121 leads ($0.9665$), followed by InceptionResNetV2 ($0.9552$), InceptionV3 ($0.9544$), and VGG16 ($0.9526$). ResNet50 ($0.7553$) and Custom CNN ($0.7864$) suffer marked degradations under blur ($0.4103$ / $0.7265$) and contrast ($0.3966$ / $0.5174$).
- **FLRS**: InceptionV3 achieves $0.9860$, Xception reaches $0.9876$, and InceptionResNetV2 obtains $0.9783$.
- **FR**: Xception achieves $0.9901$, DenseNet121 achieves $0.9808$, and InceptionResNetV2 achieves $0.9811$.

#### LMOS Stability ($S_{\text{LMOS}}$) Across Perturbations (Table 3 of the paper)
- **EPLI**: Xception leads ($0.8805$), MobileNetV2 obtains $0.8404$, DenseNet121 attains $0.8251$, and ResNet50 obtains $0.8281$. Custom CNN scores lowest ($0.6586$, falling to $0.4545$ under contrast).
- **BDML**: ResNet50 achieves highest LMOS stability ($0.8565$), closely followed by Xception ($0.8493$) and DenseNet121 ($0.8130$). InceptionV3 ($0.6253$) and Custom CNN ($0.5556$) show severe explanation drift.
- **FLRS**: Xception leads ($0.8741$), followed by ResNet50 ($0.8347$) and DenseNet121 ($0.8286$). InceptionV3 achieves $0.7326$, and Custom CNN drops to $0.6763$.
- **FR**: Xception leads with an outstanding $0.9478$, followed by DenseNet121 ($0.8978$), ResNet50 ($0.8960$), and VGG16 ($0.8880$).

---

### Key Findings & Cross-Dataset Discussion

1. **High Accuracy Does Not Guarantee Reliability (ETI)**:
   - On the EPLI dataset, Custom CNN achieves **99.45% accuracy**, yet its mean LMOS is only **0.1900** (81% of its salient activation falls on the background), giving it the lowest ETI (**0.5922**). In contrast, MobileNetV2 with identical accuracy (99.45%) directs 77.65% of its focus onto the leaf ($\text{LMOS} = 0.7765$, $\text{ETI} = 0.8855$).
   - Across all four datasets, **DenseNet121 achieves the highest classification accuracy** (1.0000 on EPLI, 0.9802 on BDML, 0.9046 on FLRS, 0.9124 on FR), but **achieves the highest ETI on none**.
   - MobileNetV2 obtains the highest ETI on both leaf datasets (EPLI: 0.8855, BDML: 0.8657), while Xception obtains the highest ETI on both flower datasets (FLRS: 0.7849, FR: 0.7531).
2. **Xception Demonstrates Dominant Explainable Robustness (ERI)**:
   - Across all four benchmark datasets, **Xception achieves the highest overall ERI on every single dataset**: **0.9346** on EPLI, **0.8982** on BDML, **0.9308** on FLRS, and **0.9689** on FR. Depthwise separable convolutions provide superior coupled stability across both classification predictions and spatial activation heatmaps.
   - DenseNet121 consistently secures second or third place in ERI across all datasets.
   - Custom CNN consistently yields the lowest ERI on all four datasets (0.7944 on EPLI, 0.6710 on BDML, 0.8146 on FLRS, 0.8680 on FR).
3. **Predictive and Explanation Stability Diverge Under Perturbation**:
   - A model can preserve its predictions while substantially shifting its visual explanation, or vice versa.
   - On EPLI under **contrast perturbation**, Custom CNN maintains high accuracy stability ($S_{\text{acc}} = 0.8494$) but suffers severe explanation collapse ($S_{\text{LMOS}} = 0.4545$). Conversely, ResNet50 exhibits low accuracy stability ($S_{\text{acc}} = 0.4599$) while maintaining high explanation stability ($S_{\text{LMOS}} = 0.7830$).
   - On FLRS, InceptionV3 maintains very high accuracy stability ($S_{\text{acc}} = 0.9860$), but its LMOS stability drops to $0.7326$. ERI unifies both dimensions into a joint metric.
4. **Domain Variations (Leaves vs. Flowers)**:
   - Leaf datasets (EPLI, BDML) produce higher ETI scores than flower datasets (FLRS, FR), primarily due to higher LMOS on well-defined leaf margins compared to complex floral petals.
   - However, modest clean ETI on flower datasets does not imply low robustness: Xception attains its highest ERI on FR (**0.9689**) and FLRS (**0.9308**), confirming that clean-image trustworthiness (ETI) and perturbation robustness (ERI) provide complementary, non-interchangeable evaluation criteria.

---

## Repository Organization

```
ETI_ERI/
├── README.md                                    # Master documentation and empirical benchmark report
└── ETI_ERI/                                     # Core experimental suite (107 Jupyter Notebooks)
    │
    ├── ETI_ERI_EgyPLI/                          # EPLI Benchmark Suite (27 notebooks)
    │   ├── egypli-data-split-into-fold.ipynb    # Stratified Train/Val/Test partitioning
    │   ├── Model_training_validation_testing/   # 8 model training & validation notebooks
    │   ├── Trustworthy/                         # 8 LMOS/ETI evaluation notebooks + cross-model comparison
    │   └── Robustness/                          # 8 perturbation stress notebooks + ERI comparison
    │
    ├── ETI_ERI_BDMediLeaves/                    # BDML Benchmark Suite (26 notebooks)
    │   ├── Model_training_validation_testing/   # 8 model training & validation notebooks
    │   ├── Trustworthy/                         # 8 LMOS/ETI evaluation notebooks + cross-model comparison
    │   └── Robustness/                          # 8 perturbation stress notebooks + ERI comparison
    │
    ├── flower_dataset_tensorflow/               # FLRS Benchmark Suite (27 notebooks)
    │   ├── tf-flowers-split.ipynb               # Train/Val/Test partitioning
    │   ├── training_validation_testing/         # 8 model training & validation notebooks
    │   ├── Trustworthiness/                     # 8 LMOS/ETI evaluation notebooks + cross-model comparison
    │   └── Robustness/                          # 8 perturbation stress notebooks + ERI comparison
    │
    └── Alxmamaev_flower_dataset/                # FR Benchmark Suite (27 notebooks)
        ├── alxmamaev-flower-dataset-split.ipynb # Train/Val/Test partitioning
        ├── training_validation_testing/         # 8 model training & validation notebooks
        ├── Trustworthiness/                     # 8 LMOS/ETI evaluation notebooks + cross-model comparison
        └── Robustness/                          # 8 perturbation stress notebooks + ERI comparison
```

---

## Installation & Environment Setup

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support (tested on 2 $\times$ NVIDIA Tesla T4 GPUs on Kaggle)

### Dependencies
Install the required packages:
```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

---

## Workflow & Reproduction Guide

### Phase 1: Data Acquisition & Preprocessing
1. **Acquire Benchmarks**: Download datasets and polygonal JSON annotations from the associated resources:
   - **EPLI**: [Splits](https://www.kaggle.com/datasets/ragibmehedi/egypli-split) | [JSON Polygon Masks](https://www.kaggle.com/datasets/ragibmehedi/egypli-annotated-json-files) | [Checkpoints](https://www.kaggle.com/datasets/ragibmehedi/egypli-keras-files)
   - **BDML**: [BDMediLeaves Benchmark](https://www.kaggle.com/datasets/ragibmehedi/bdmedidataset)
   - **FLRS**: [TF Flowers Benchmark](https://www.kaggle.com/datasets/ragibmehedi/flower-dataset-tensorflow)
   - **FR**: [Alxmamaev Flowers Benchmark](https://www.kaggle.com/datasets/ragibmehedi/alxmamaev-flower-dataset)
2. **Execute Partitioning**:
   - `ETI_ERI/ETI_ERI_EgyPLI/egypli-data-split-into-fold.ipynb`
   - `ETI_ERI/flower_dataset_tensorflow/tf-flowers-split.ipynb`
   - `ETI_ERI/Alxmamaev_flower_dataset/alxmamaev-flower-dataset-split.ipynb`

### Phase 2: Model Training & Validation
Navigate to the corresponding `Model_training_validation_testing/` (or `training_validation_testing/`) directory for each dataset and execute the notebook for your target architecture (e.g., `inceptionresnetv2-egypli.ipynb` or `xception-egypli.ipynb`). This trains the model with Adam ($lr = 5 \times 10^{-5}$) and saves the `.keras` model checkpoints.

### Phase 3: Trustworthiness (ETI) Evaluation
Navigate to `Trustworthy/` (or `Trustworthiness/`) for the target dataset:
1. Load test images and corresponding polygon JSON files.
2. Generate Grad-CAM heatmaps from the final convolutional layer (resized to $299 \times 299$ for Xception, $224 \times 224$ for other models).
3. Binarize attention maps at threshold $\tau = 0.70$.
4. Compute per-sample LMOS and aggregate ETI via $\mathrm{ETI} = (acc + \overline{\mathrm{LMOS}}) / 2$.
5. Run the model comparison notebook to generate cross-architecture ETI rankings and ablation metrics.

### Phase 4: Perturbation Stress-Testing & Robustness (ERI)
Navigate to `Robustness/` for the target dataset:
1. Apply the 9 fixed image perturbations (noise, blur, brightness, hue, contrast, gamma, motion blur, compression, occlusion).
2. Measure perturbed classification accuracy ($acc_p$) and perturbed explanation overlap ($\mathrm{LMOS}_p$).
3. Compute $S_{\mathrm{acc}}^p$, $S_{\mathrm{LMOS}}^p$, perturbation ERI ($\mathrm{ERI}^p$), and overall ERI.
4. Run the model comparison notebook to generate multi-architecture radar charts and robustness rankings.

---

## Data and Code Availability

Supplementary code, notebooks, dataset splits, LabelMe polygon annotation files (`.json`), and trained model weights (`.keras` files) associated with this research are accessible at:

**Official Repository**: [https://github.com/ragib1996/ETI_ERI](https://github.com/ragib1996/ETI_ERI)

---

## Citation

If you use the ETI & ERI evaluation framework, metrics, notebooks, or benchmark results in your research, please cite:

```bibtex
@article{mehedi2026beyond,
  title={Beyond Accuracy: Explainable Trustworthy Index and Explainable Robustness Index for Evaluating CNN},
  author={Mehedi, Ragib and Haque, Azmain Inquaid and Abdal, Md. Nazmul and Noman, Abdullah Al and Islam, Md. Zahidul},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/ragib1996/ETI_ERI}
}
```

---

## License & Acknowledgments

- **License**: This project is open-source and distributed under the [MIT License](LICENSE).
- **Acknowledgments**: The authors are grateful to the Computer Science and Engineering Discipline, Khulna University, for their continuous support and research facilities.
- **Disclosure**: No conflict of interest was reported by the authors. This research received no external funding.
