# K-Nearest Neighbor Classification — Breast Cancer Diagnosis

## Project Overview
This project implements a complete KNN classification pipeline using the **Breast Cancer Wisconsin (Diagnostic)** dataset. The goal is to classify tumors as **malignant** or **benign** based on 30 numerical features extracted from cell nucleus images.

## Dataset Description
| Property | Detail |
|----------|--------|
| Source | `sklearn.datasets.load_breast_cancer()` |
| Samples | 569 |
| Features | 30 numeric (mean, SE, and worst of 10 cell measurements) |
| Classes | Malignant (212) · Benign (357) |
| Missing Values | None |

## Key Findings
- **Optimal k = 9** identified via 5-fold stratified cross-validation
- **Test Accuracy: 97.37%** on held-out 20% test set
- **F1-Score: 0.9735** (weighted) — strong balance of precision and recall
- **Perfect precision on malignant class** (1.00) — no benign tumors misclassified as malignant
- `StandardScaler` was critical: unscaled features caused ~4% accuracy drop in preliminary tests
- k=1 showed classic overfitting with highest variance (±0.018 vs ±0.015 at k=9)

## Project Structure
```
knn_project/
├── knn_classification.ipynb   # Main notebook (all cells executed)
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── class_distribution.png     # EDA figure
├── correlation_heatmap.png    # Feature correlation matrix
├── pairplot.png               # Feature pairplot
├── k_vs_accuracy.png          # k-tuning plot
└── confusion_matrix.png       # Final model evaluation
```

## Setup Instructions

### Option A — pip
```bash
pip install -r requirements.txt
jupyter notebook knn_classification.ipynb
```

### Option B — conda
```bash
conda create -n knn_env python=3.10
conda activate knn_env
pip install -r requirements.txt
jupyter notebook knn_classification.ipynb
```

## Workflow Summary
1. **Load & Explore** — Dataset shape, class balance, summary statistics
2. **Missing Value Check** — No imputation needed
3. **Visualization** — Class distribution, feature correlations, pairplots
4. **Train/Test Split** — 80/20 stratified split (455 train / 114 test)
5. **Feature Scaling** — `StandardScaler` fit on train only to prevent leakage
6. **k Tuning** — Cross-validation across k ∈ {1, 3, 5, 7, 9, 11}
7. **Final Model** — KNN (k=9, Euclidean distance)
8. **Evaluation** — Accuracy, precision, recall, F1, confusion matrix
9. **New Predictions** — Inference on 3 synthetic patient samples

## Why KNN + Why Scaling?
KNN is a non-parametric, instance-based learner that classifies by majority vote among k nearest neighbors using Euclidean distance. Because distance is scale-sensitive, `StandardScaler` is essential — without it, large-magnitude features (e.g., `mean area` ~143–2501) dominate over small-magnitude ones (e.g., `mean smoothness` ~0.05–0.16), severely biasing predictions.

## Author
**Abubakar Jibrin Gunda**  
AI/Data Professional · Kano State, Nigeria
