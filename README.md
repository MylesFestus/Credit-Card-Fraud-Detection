# Credit Card Fraud Detection — ML Model Comparison

A multi-model machine learning pipeline for detecting credit card fraud.
Trains and evaluates four classifiers with SMOTE oversampling, automatic
threshold tuning (F2-optimised), and 5-fold cross-validation.

---

## Results summary

| Model | Precision | Recall | F1 | AUC-PR | AUC-ROC | FP | FN |
|---|---|---|---|---|---|---|---|
| **Random Forest** ✦ | **0.7395** | 0.8980 | **0.8111** | **0.8765** | 0.9762 | **31** | 10 |
| XGBoost | 0.2005 | 0.9082 | 0.3284 | 0.8493 | 0.9758 | 355 | 9 |
| Logistic Regression | 0.0273 | 0.9184 | 0.0530 | 0.7703 | 0.9698 | 3208 | 8 |
| SVM | 0.0308 | 0.9184 | 0.0596 | 0.6169 | 0.9812 | 2830 | 8 |

### Best model: Random Forest

Random Forest achieves the best balance of precision and recall, with an F1
score of 0.81 and only 31 false positives — compared to thousands for LR and
SVM. It ranks first on AUC-PR (0.88), which is the most reliable metric for
imbalanced datasets like fraud detection.

**Why Random Forest wins:**
- Highest precision (0.74) by a large margin — 31 false alarms vs 355–3,208 for others
- Recall of 0.90 — catches 88 of 98 actual fraud cases
- Only model with a meaningful F1 score (0.81 vs ≤0.33 for all others)
- Highest AUC-PR (0.88) — best overall precision-recall tradeoff
- 5× fewer false positives than XGBoost, 100× fewer than Logistic Regression

---

## Project structure

```
.
├── creditcard.csv                  # Raw dataset (not included — download separately)
├── fraud_detection_models.py       # Basic multi-model script
├── fraud_detection_cv_tuned.py     # Full script with CV + threshold tuning
├── README.md                       # This file
│
└── outputs/
    ├── fraud_model_comparison.png  # Bar chart of all metrics
    ├── fraud_pr_curves.png         # Precision-Recall curves
    ├── fraud_threshold_sweep.png   # F2/Precision/Recall vs threshold per model
    ├── fraud_confusion_matrices.png# 2×2 grid at optimal thresholds
    ├── fraud_cv_scores.png         # Cross-validation AUC-PR and F2 with error bars
    ├── fraud_leaderboard.png       # Composite score ranked chart
    └── fraud_feature_importance.png# Top 15 features from Random Forest
```

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions, 492 fraud cases (0.172% fraud rate)
- Features: `Time`, `Amount`, and `V1`–`V28` (PCA-transformed for privacy)
- Target: `Class` — 1 = fraud, 0 = legitimate

Download `creditcard.csv` and place it in the project root before running.

---

## Setup

### Requirements

```bash
pip install pandas numpy matplotlib scikit-learn xgboost imbalanced-learn
```

### Run

```bash
# Basic comparison
python fraud_detection_models.py

# Full pipeline with cross-validation and threshold tuning (recommended)
python fraud_detection_cv_tuned.py
```

---

## Pipeline overview

### 1. Preprocessing
- `Amount` and `Time` are standardised with `StandardScaler`
- `V1`–`V28` are already PCA-transformed in the source data
- Stratified 80/20 train/test split — preserves the 0.17% fraud ratio

### 2. Class imbalance — SMOTE
Synthetic Minority Oversampling Technique (SMOTE) is applied to the **training
set only** inside each cross-validation fold to prevent data leakage. This
generates synthetic fraud samples by interpolating between existing ones.

### 3. Models trained

| Model | Key hyperparameters |
|---|---|
| Logistic Regression | `class_weight="balanced"`, `max_iter=1000` |
| Random Forest | 200 trees, `class_weight="balanced"`, `max_features="sqrt"` |
| XGBoost | `scale_pos_weight`, `eval_metric="aucpr"`, 200 rounds |
| SVM | RBF kernel, `class_weight="balanced"` (5k-row subsample) |

### 4. Threshold tuning
Each model's decision threshold is tuned independently by sweeping 199 values
from 0.01 to 0.99 and selecting the one that maximises the **F2 score**
(which weights recall twice as heavily as precision — appropriate for fraud).

### 5. Cross-validation
5-fold stratified cross-validation with SMOTE applied inside each fold.
Reports mean ± standard deviation for AUC-PR and F2 across folds.

### 6. Best model selection
A weighted composite score is computed for each model:

```
composite = AUC-PR × 0.40 + F2 × 0.30 + Recall × 0.20 + Precision × 0.10
```

The weights reflect fraud detection priorities: catching fraud (recall) matters
most, but precision is needed to keep false alarms manageable for analysts.

---

## Key concepts

**Why not use accuracy?**
With 99.83% legitimate transactions, a model that predicts "legit" for
everything achieves 99.83% accuracy — while catching zero fraud cases.
Accuracy is meaningless here.

**Why AUC-PR over AUC-ROC?**
AUC-ROC can look flattering on imbalanced data because it accounts for
true negatives, which are abundant. AUC-PR focuses only on the positive
(fraud) class and is a stricter, more honest measure of model quality.

**Why F2 for threshold tuning?**
F2 = (1 + 4) × (Precision × Recall) / (4 × Precision + Recall)

It weights recall twice as heavily as precision, reflecting the real-world
cost asymmetry: missing a fraud (false negative) is far more damaging than
a false alarm (false positive).

**Why SMOTE inside CV folds?**
Applying SMOTE before splitting introduces synthetic samples derived from
the validation set into the training data — a form of data leakage that
inflates validation scores. Correct practice is to fit SMOTE only on each
fold's training portion.

---

## Improving results further

- **Tune Random Forest hyperparameters** with `GridSearchCV` or `Optuna`
  optimising for AUC-PR
- **Try LightGBM** — often faster than XGBoost with comparable accuracy
- **Ensemble** — stack Random Forest + XGBoost predictions as input to a
  meta-learner (logistic regression)
- **Feature engineering** — add rolling transaction velocity, time-since-last
  transaction, and amount deviation from user history
- **Cost-sensitive learning** — assign explicit dollar costs to FP and FN
  and optimise for minimum expected loss rather than F2

---

## Interpretation notes

A false positive (FP) means a legitimate transaction is blocked — customer
friction and potential churn. A false negative (FN) means fraud goes through
— direct financial loss and reputational damage. The right FP/FN tradeoff
depends on your business context; adjust the F-beta `beta` parameter and
composite score weights accordingly.

---

## License

This project is for educational purposes. The dataset is subject to the
[Kaggle terms of use](https://www.kaggle.com/terms).
