# ETI & ERI: Explainable Trustworthiness Index & Explainable Robustness Index for Deep Vision Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical research framework and comprehensive benchmark suite introducing two novel quantitative evaluation metrics for deep computer vision models:
- **ETI (Explainable Trustworthiness Index)**: Unifying classification performance with visual explanation faithfulness to detect and penalize "Clever Hans" behavior.
- **ERI (Explainable Robustness Index)**: Quantifying the coupled resilience of both classification predictions and explanation maps under real-world visual perturbations.

This repository provides **107 end-to-end reproducible Jupyter Notebooks** evaluating **8 deep CNN architectures** across **4 specialized botanical and floral image datasets** subjected to **9 physical and digital corruption modalities**.

---

## Table of Contents

- [Motivation](#motivation)
- [Methodology & Mathematical Formulations](#methodology--mathematical-formulations)
  - [1. Localization-Mask Overlap Score (LMOS)](#1-localization-mask-overlap-score-lmos)
  - [2. Explainable Trustworthiness Index (ETI)](#2-explainable-trustworthiness-index-eti)
  - [3. Explainable Robustness Index (ERI)](#3-explainable-robustness-index-eri)
- [Taxonomy of Evaluated Perturbations](#taxonomy-of-evaluated-perturbations)
- [Datasets & Benchmark Architectures](#datasets--benchmark-architectures)
  - [Datasets](#datasets)
  - [Evaluated Architectures](#evaluated-architectures)
- [Empirical Benchmark Results](#empirical-benchmark-results)
  - [1. EgyPLI Dataset](#1-egypli-dataset)
  - [2. BDMediLeaves Dataset](#2-bdmedileaves-dataset)
  - [3. Alxmamaev Flowers Dataset](#3-alxmamaev-flowers-dataset)
  - [4. TensorFlow Flowers Dataset](#4-tensorflow-flowers-dataset)
  - [Key Findings & Takeaways](#key-findings--takeaways)
- [Repository Organization](#repository-organization)
- [Installation & Environment Setup](#installation--environment-setup)
- [Workflow & Reproduction Guide](#workflow--reproduction-guide)
  - [Phase 1: Data Preprocessing & Splitting](#phase-1-data-preprocessing--splitting)
  - [Phase 2: Model Training & Evaluation](#phase-2-model-training--evaluation)
  - [Phase 3: Trustworthiness (ETI) Evaluation](#phase-3-trustworthiness-eti-evaluation)
  - [Phase 4: Perturbation Stress-Testing & Robustness (ERI)](#phase-4-perturbation-stress-testing--robustness-eri)
- [Citation](#citation)

---

## Motivation

Standard deep learning classification benchmarks routinely prioritize top-line metrics such as accuracy, precision, and F1-score. However, high accuracy often masks deep model vulnerabilities:
1. **The "Clever Hans" Effect**: A network may achieve 99%+ accuracy by latching onto spurious background correlations (e.g., soil textures, watermarks, illumination artifacts) rather than the actual object of interest (the leaf or flower petal).
2. **Brittle Visual Explanations**: A model may remain accurate on clean images, but under slight atmospheric or sensor perturbations, its internal representation and attention maps shift wildly, degrading reliability in mission-critical applications (such as agricultural disease diagnosis and medical botany).

**ETI** and **ERI** bridge the gap between traditional classification metrics and Explainable AI (XAI) by providing mathematically rigorous, bounded $[0, 1]$ indices that penalize models that either hallucinate explanations or lose explanatory fidelity under stress.

---

## Methodology & Mathematical Formulations

```
+-----------------------------------------------------------------------------------+
|                                 INPUT IMAGE                                       |
+-----------------------------------------+-----------------------------------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
            [ Clean Image ]                              [ Perturbed Image ]
                   |                                      (9 Perturbations)
                   v                                             |
       +-----------------------+                                 v
       |   Deep CNN Backbone   |                     +-----------------------+
       +-----------+-----------+                     |   Deep CNN Backbone   |
                   |                                 +-----------+-----------+
         +---------+---------+                                   |
         |                   |                         +---------+---------+
         v                   v                         v                   v
     [Pred: y_clean]    [Grad-CAM]                 [Pred: y_pert]     [Grad-CAM]
         |                   |                         |                   |
         v                   v                         v                   v
   (Accuracy_clean)       (LMOS)                   (Accuracy_pert)      (LMOS_pert)
         |                   |                         |                   |
         +---------+---------+                         +---------+---------+
                   |                                             |
                   v                                             v
         =====================                         =====================
           ETI Calculation                               ERI Calculation
          (Trustworthiness)                            (Coupled Stability)
         =====================                         =====================
```

### 1. Localization-Mask Overlap Score (LMOS)

Given a trained model, the Grad-CAM heatmap $\mathcal{H} \in \mathbb{R}^{H \times W}$ is generated for the target class with respect to the final convolutional feature maps:

$$\mathcal{H} = \text{ReLU}\left(\sum_{k} \alpha_k A^k\right), \quad \text{where} \quad \alpha_k = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial y^c}{\partial A_{i,j}^k}$$

The heatmap is normalized and binarized with threshold $\tau = 0.70$ to produce a binary attention mask $\mathcal{M}_{\text{pred}} \in \{0, 1\}^{H \times W}$. Comparing $\mathcal{M}_{\text{pred}}$ against the ground-truth segmentation mask $\mathcal{M}_{\text{gt}} \in \{0, 1\}^{H \times W}$, the **Localization-Mask Overlap Score (LMOS)** quantifies the proportion of model attention allocated within the true object boundaries:

$$\text{LMOS} = \begin{cases} \dfrac{\sum_{i,j} \left(\mathcal{M}_{\text{pred}}(i,j) \land \mathcal{M}_{\text{gt}}(i,j)\right)}{\sum_{i,j} \mathcal{M}_{\text{pred}}(i,j)}, & \text{if } \sum_{i,j} \mathcal{M}_{\text{pred}}(i,j) > 0 \\ 0.0, & \text{otherwise} \end{cases}$$

### 2. Explainable Trustworthiness Index (ETI)

The **Explainable Trustworthiness Index (ETI)** evaluates whether a model simultaneously achieves high classification accuracy while grounding its predictions on semantically valid object regions:

$$\text{ETI} = \frac{\text{Accuracy} + \text{LMOS}}{2}, \quad \text{where } \text{ETI} \in [0, 1]$$

To quantify the individual contributions of predictive accuracy versus explanation faithfulness, ablation components are defined as:
- **ETI without Accuracy**: $\text{ETI}_{\setminus \text{Acc}} = \frac{\text{LMOS}}{2}$
- **ETI without LMOS**: $\text{ETI}_{\setminus \text{LMOS}} = \frac{\text{Accuracy}}{2}$

### 3. Explainable Robustness Index (ERI)

When subjected to a visual perturbation $p \in \mathcal{P}$, model stability is evaluated across both prediction and explanation:

#### A. Accuracy Stability ($S_{\text{acc}}$)
$$S_{\text{acc}}(p) = 1 - \frac{|\text{Acc}_{\text{clean}} - \text{Acc}_{\text{perturbed}}(p)|}{\text{Acc}_{\text{clean}} + \text{Acc}_{\text{perturbed}}(p) + \epsilon}$$

#### B. Explanation / LMOS Stability ($S_{\text{LMOS}}$)
For each test sample $i \in \{1, \dots, N\}$, explanation stability measures the shift in LMOS between clean and perturbed conditions:
$$S_{\text{LMOS}}^{(i)}(p) = 1 - \frac{|\text{LMOS}_{\text{clean}}^{(i)} - \text{LMOS}_{\text{perturbed}}^{(i)}(p)|}{\text{LMOS}_{\text{clean}}^{(i)} + \text{LMOS}_{\text{perturbed}}^{(i)}(p) + \epsilon}$$
$$S_{\text{LMOS}}(p) = \frac{1}{N}\sum_{i=1}^{N} S_{\text{LMOS}}^{(i)}(p)$$

#### C. ERI per Perturbation and Overall ERI
$$\text{ERI}(p) = \frac{S_{\text{acc}}(p) + S_{\text{LMOS}}(p)}{2}$$
$$\text{ERI}_{\text{overall}} = \frac{\bar{S}_{\text{acc}} + \bar{S}_{\text{LMOS}}}{2}$$

---

## Taxonomy of Evaluated Perturbations

The framework subjects test sets to 9 realistic environmental and sensor corruptions:

| Perturbation Modality | Parameters | Physical Simulation |
| :--- | :--- | :--- |
| **Gaussian Blur** | `kernel_size = 5` | Lens defocus, atmospheric haze, depth-of-field blur |
| **Gaussian Noise** | `mean = 0`, `std = 0.05` | Sensor thermal noise, low-light ISO noise |
| **Brightness Adjustment** | `factor = +0.2` | Overexposure, direct sunlight, flash artifacts |
| **Hue Shift** | `shift = +15` (HSV) | Color cast, spectral distortion, twilight illumination |
| **Contrast Modification** | `factor = 0.5` | Fog, diffuse lighting, dynamic range compression |
| **Gamma Correction** | `gamma = 2.0` | Nonlinear sensor response, display gamma distortion |
| **Motion Blur** | `kernel_size = 7` (1D) | Camera motion, subject movement, handheld jitter |
| **JPEG Compression** | `quality = 30` | Lossy digital transmission, bandwidth compression |
| **Random Occlusion** | `patch_size = 20%` | Physical obstructions, overlapping leaves, debris |

---

## Datasets & Benchmark Architectures

### Datasets

1. **EgyPLI (Egyptian Plant Leaves Identification)**:
   - High-resolution leaf specimens across diverse botanical species.
   - Ground-truth segmentation masks delineating precise leaf contours.
2. **BDMediLeaves (Bangladeshi Medicinal Leaves Dataset)**:
   - Indigenous medicinal plant leaves categorized by therapeutic classes.
   - Exact polygon annotations generated via LabelMe.
3. **Alxmamaev Flowers Dataset**:
   - Comprehensive flower categorization benchmark (daisy, dandelion, rose, sunflower, tulip).
   - Polygon boundary annotations for floral foreground regions.
4. **TensorFlow Flowers Dataset (`tf_flowers`)**:
   - Standard 5-class botanical vision dataset paired with corresponding polygon segmentation ground truth.

### Evaluated Architectures

All experiments benchmark 8 convolutional backbones trained under identical optimization criteria:
- **Custom CNN** (baseline multi-layer convolutional architecture)
- **DenseNet121** (dense feature reuse)
- **InceptionV3** (factorized convolutions)
- **InceptionResNetV2** (residual inception hybrid)
- **MobileNetV2** (inverted residual blocks with linear bottlenecks)
- **ResNet50 / ResNet** (residual skip connections)
- **VGG16** (deep uniform $3\times3$ convolutions)
- **Xception** (depthwise separable convolutions)

---

## Empirical Benchmark Results

### 1. EgyPLI Dataset

| Model | Accuracy | LMOS | **ETI (Full)** | Acc Stability | LMOS Stability | **ERI (Full)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | 0.9945 | 0.1900 | 0.5922 | 0.9302 | 0.6586 | 0.7944 |
| **InceptionV3** | 0.9917 | 0.3617 | 0.6767 | 0.9849 | 0.8251 | 0.9050 |
| **InceptionResNetV2** | 0.9889 | 0.5529 | 0.7709 | 0.9766 | 0.7463 | 0.8614 |
| **DenseNet121** | **1.0000** | 0.6964 | 0.8482 | 0.9703 | 0.6961 | 0.8332 |
| **MobileNetV2** | 0.9945 | **0.7765** | **0.8855** | 0.9704 | 0.8404 | 0.9054 |
| **ResNet** | 0.6122 | 0.7256 | 0.6689 | 0.8263 | 0.8281 | 0.8272 |
| **VGG16** | 0.9834 | 0.5268 | 0.7551 | 0.9469 | 0.7033 | 0.8251 |
| **Xception** | 0.9945 | 0.7638 | 0.8791 | **0.9887** | **0.8805** | **0.9346** |

### 2. BDMediLeaves Dataset

| Model | Accuracy | LMOS | **ETI (Full)** | Acc Stability | LMOS Stability | **ERI (Full)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | 0.7376 | 0.3310 | 0.5343 | 0.7864 | 0.5556 | 0.6710 |
| **InceptionV3** | 0.9307 | 0.3344 | 0.6325 | **0.9665** | 0.8130 | 0.8898 |
| **InceptionResNetV2** | 0.9059 | 0.5184 | 0.7122 | 0.9552 | 0.6969 | 0.8260 |
| **DenseNet121** | **0.9802** | 0.7118 | 0.8460 | 0.9544 | 0.6253 | 0.7898 |
| **MobileNetV2** | 0.9703 | 0.7611 | **0.8657** | 0.9523 | 0.7840 | 0.8681 |
| **ResNet** | 0.4604 | **0.7817** | 0.6211 | 0.7553 | **0.8565** | 0.8059 |
| **VGG16** | 0.8960 | 0.5873 | 0.7416 | 0.9526 | 0.6988 | 0.8257 |
| **Xception** | 0.9505 | 0.7800 | 0.8653 | 0.9471 | 0.8493 | **0.8982** |

### 3. Alxmamaev Flowers Dataset

| Model | Accuracy | LMOS | **ETI (Full)** | Acc Stability | LMOS Stability | **ERI (Full)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | 0.6820 | 0.2714 | 0.4767 | 0.9563 | 0.7798 | 0.8680 |
| **InceptionV3** | 0.8825 | 0.3063 | 0.5944 | 0.9808 | 0.8978 | 0.9393 |
| **InceptionResNetV2** | 0.8894 | 0.4705 | 0.6800 | 0.9811 | 0.8736 | 0.9274 |
| **DenseNet121** | **0.9124** | 0.5382 | 0.7253 | 0.9795 | 0.7858 | 0.8827 |
| **MobileNetV2** | 0.8986 | 0.5091 | 0.7038 | 0.9758 | 0.8541 | 0.9149 |
| **ResNet** | 0.4977 | 0.4769 | 0.4873 | 0.8944 | 0.8960 | 0.8952 |
| **VGG16** | 0.8387 | 0.5786 | 0.7087 | 0.9644 | 0.8880 | 0.9262 |
| **Xception** | 0.9032 | **0.6029** | **0.7531** | **0.9901** | **0.9478** | **0.9689** |

### 4. TensorFlow Flowers Dataset

| Model | Accuracy | LMOS | **ETI (Full)** | Acc Stability | LMOS Stability | **ERI (Full)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom CNN** | 0.7003 | 0.2891 | 0.4947 | 0.9528 | 0.6763 | 0.8145 |
| **InceptionV3** | 0.8229 | 0.3749 | 0.5989 | 0.9707 | 0.8286 | 0.8997 |
| **InceptionResNetV2** | 0.8501 | 0.5581 | 0.7041 | 0.9783 | 0.8018 | 0.8901 |
| **DenseNet121** | **0.9046** | 0.6188 | 0.7617 | 0.9860 | 0.7326 | 0.8593 |
| **MobileNetV2** | 0.8665 | 0.5958 | 0.7311 | 0.9741 | 0.7942 | 0.8841 |
| **ResNet** | 0.5123 | 0.5623 | 0.5373 | 0.9064 | 0.8347 | 0.8705 |
| **VGG16** | 0.8147 | 0.6311 | 0.7229 | 0.9723 | 0.8214 | 0.8969 |
| **Xception** | 0.8828 | **0.6869** | **0.7849** | **0.9876** | **0.8741** | **0.9308** |

### Key Findings & Takeaways

1. **High Accuracy Does Not Equal High Trustworthiness**:
   - On the EgyPLI benchmark, Custom CNN achieves **99.45% accuracy**, yet its LMOS is merely **0.1900**, resulting in an ETI of only **0.5922**. It largely relies on peripheral background features rather than genuine botanical characteristics.
2. **Depthwise Separability Maximizes Robustness**:
   - Across all four benchmark datasets, **Xception** consistently secures the highest or second-highest ERI (reaching **0.9689** on Alxmamaev Flowers and **0.9346** on EgyPLI). Depthwise separable convolutions provide superior resilience against frequency-domain and spatial corruptions.
3. **Dense Connectivity Enhances Explanation Localization**:
   - **DenseNet121** and **MobileNetV2** exhibit the highest LMOS alignment, proving that multi-scale feature propagation and residual bottleneck structures preserve fine-grained localized spatial cues better than shallow or unregularized backbones.

---

## Repository Organization

```
ETI_ERI/
├── README.md                                    # Comprehensive documentation and benchmark report
└── ETI_ERI/                                     # Core experimental suite (107 notebooks)
    │
    ├── ETI_ERI_EgyPLI/                          # Egyptian Plant Leaves Benchmark (27 notebooks)
    │   ├── egypli-data-split-into-fold.ipynb    # Stratified K-fold dataset split
    │   ├── Model_training_validation_testing/   # 8 model training & evaluation notebooks
    │   ├── Trustworthy/                         # 8 LMOS/ETI component notebooks + model comparison
    │   └── Robustness/                          # 8 perturbation stress notebooks + ERI comparison
    │
    ├── ETI_ERI_BDMediLeaves/                    # BD Medicinal Leaves Benchmark (26 notebooks)
    │   ├── Model_training_validation_testing/   # 8 model training & evaluation notebooks
    │   ├── Trustworthy/                         # 8 LMOS/ETI component notebooks + model comparison
    │   └── Robustness/                          # 8 perturbation stress notebooks + ERI comparison
    │
    ├── Alxmamaev_flower_dataset/                # Alxmamaev Flowers Benchmark (27 notebooks)
    │   ├── alxmamaev-flower-dataset-split.ipynb # Train/Val/Test partitioning
    │   ├── training_validation_testing/         # 8 model training & evaluation notebooks
    │   ├── Trustworthiness/                     # 8 LMOS/ETI component notebooks + model comparison
    │   └── Robustness/                          # 8 perturbation stress notebooks + ERI comparison
    │
    └── flower_dataset_tensorflow/               # TF Flowers Benchmark (27 notebooks)
        ├── tf-flowers-split.ipynb               # Train/Val/Test partitioning
        ├── training_validation_testing/         # 8 model training & evaluation notebooks
        ├── Trustworthiness/                     # 8 LMOS/ETI component notebooks + model comparison
        └── Robustness/                          # 8 perturbation stress notebooks + ERI comparison
```

---

## Installation & Environment Setup

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended for accelerated model inference and Grad-CAM generation)

### Dependencies

Install the core dependencies:
```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

---

## Workflow & Reproduction Guide

### Phase 1: Data Preprocessing & Splitting
Run the dataset split notebooks to partition images into stratified training, validation, and test splits while generating index CSVs:
- `ETI_ERI/ETI_ERI_EgyPLI/egypli-data-split-into-fold.ipynb`
- `ETI_ERI/Alxmamaev_flower_dataset/alxmamaev-flower-dataset-split.ipynb`
- `ETI_ERI/flower_dataset_tensorflow/tf-flowers-split.ipynb`

### Phase 2: Model Training & Evaluation
Navigate to the corresponding `Model_training_validation_testing/` directory for any dataset and execute the notebook for your architecture of choice (e.g., `inceptionresnetv2-egypli.ipynb`). This trains the model, records learning curves, and saves the evaluation checkpoints.

### Phase 3: Trustworthiness (ETI) Evaluation
Open the model's notebook in `Trustworthy/` (or `Trustworthiness/`):
1. Loads the test images and corresponding LabelMe segmentation JSON masks.
2. Extracts Grad-CAM heatmaps from the final convolutional layer.
3. Binarizes the heatmaps at threshold $\tau = 0.70$.
4. Computes per-sample LMOS and aggregate ETI.
5. Run `*-eti-model-wise-comparison.ipynb` to generate cross-model comparative bar plots and ablation CSVs.

### Phase 4: Perturbation Stress-Testing & Robustness (ERI)
Open the model's notebook in `Robustness/`:
1. Generates 9 corrupted variants of each test image using OpenCV transformations.
2. Calculates $S_{\text{acc}}(p)$ and per-sample $S_{\text{LMOS}}(p)$ across each corruption.
3. Synthesizes full ERI scores and ablation breakdowns.
4. Run `*-eri-model-wise-comparison.ipynb` to generate multi-model radar and bar comparison figures.

---

## Citation

If you use the ETI & ERI framework, code, or benchmark results in your research, please cite this work:

```bibtex
@article{eti_eri_framework,
  title={Quantifying Explainable Trustworthiness and Robustness in Deep Convolutional Neural Networks},
  author={Mehedi, Ragib and Contributors},
  journal={GitHub Repository},
  year={2026},
  url={https://github.com/ragib1996/ETI_ERI}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
