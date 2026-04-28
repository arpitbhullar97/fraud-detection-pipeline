# 🏦 Financial Fraud Detection Pipeline

An end-to-end machine learning pipeline to detect fraudulent credit card transactions using Random Forest and XGBoost, with an interactive Tableau dashboard for business stakeholders.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange) ![Tableau](https://img.shields.io/badge/Tableau-Public-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2-green)

---

## 📊 Live Dashboard

🔗 [View the Fraud Detection Dashboard on Tableau Public](https://public.tableau.com/views/FinancialFraudDetectionDashboard/FraudDetectionDashboard)

---

## 🎯 Project Overview

Financial fraud causes billions in losses annually. This project builds a production-style ML pipeline that:

- Ingests and explores 284,807 real credit card transactions
- Handles severe class imbalance (only 0.17% fraud) using SMOTE
- Trains and compares Random Forest vs XGBoost classifiers
- Evaluates using fraud-appropriate metrics (Recall, PR-AUC, ROC-AUC)
- Exports predictions to an interactive Tableau dashboard

---

## 📁 Project Structure

```
fraud-detection-pipeline/
├── data/
│   ├── creditcard.csv           # Raw Kaggle dataset (not tracked in git)
│   ├── train_resampled.csv      # SMOTE-balanced training set
│   └── test.csv                 # Holdout test set
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Scaling, splitting, SMOTE
│   └── 03_modeling.ipynb       # Model training, evaluation, export
├── src/
│   ├── preprocess.py            # Reusable preprocessing functions
│   └── model.py                 # Reusable model training & evaluation
├── outputs/
│   ├── predictions.csv          # Model predictions for Tableau
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

---

## 🔍 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraud cases | 492 (0.17%) |
| Features | 30 (V1–V28 PCA-transformed + Amount + Time) |
| Missing values | None |

> ⚠️ `creditcard.csv` is not tracked in this repo due to file size. Download it from Kaggle and place it in the `data/` folder.

---

## 🧪 Methodology

### 1. Exploratory Data Analysis
- Class imbalance visualization
- Transaction amount and time distributions
- Correlation heatmap
- Top fraud-correlated features

### 2. Preprocessing
- StandardScaler applied to `Amount` and `Time`
- Stratified 80/20 train/test split
- **SMOTE** applied to training set only (prevents data leakage)

### 3. Modeling
Two models trained and compared:

| Model | Strategy |
|---|---|
| Random Forest | `class_weight='balanced'`, 100 trees |
| XGBoost | `scale_pos_weight` tuned to fraud ratio, 200 estimators |

### 4. Evaluation Metrics
Accuracy is misleading on imbalanced data. We use:

| Metric | Why it matters |
|---|---|
| **Recall** | % of actual frauds caught — most critical |
| **Precision** | % of fraud alerts that are real fraud |
| **F1** | Balance between precision and recall |
| **ROC-AUC** | Overall discrimination ability |
| **PR-AUC** | Best metric for imbalanced classification |

---

## 📈 Results

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

> Fill in your actual results after running `03_modeling.ipynb`

---

## 📊 Tableau Dashboard

The dashboard connects to `outputs/predictions.csv` and includes:

- **KPI Cards** — Total transactions, fraud caught, missed, false alarms
- **Outcome Breakdown** — True/False Positive/Negative bar chart
- **Fraud Probability Distribution** — Histogram by class
- **Transaction Amount by Outcome** — Box plot comparison
- **Probability vs Amount** — Scatter with 0.5 decision threshold

🔗 [Live Dashboard](https://public.tableau.com/views/FinancialFraudDetectionDashboard/FraudDetectionDashboard)

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git
cd fraud-detection-pipeline
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`

### 4. Run the notebooks in order
```bash
jupyter notebook
```
- `01_eda.ipynb`
- `02_preprocessing.ipynb`
- `03_modeling.ipynb`

### 5. Or run the src scripts directly
```bash
python src/preprocess.py
python src/model.py
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **pandas, numpy** — data manipulation
- **scikit-learn** — preprocessing, modeling, evaluation
- **XGBoost** — gradient boosted trees
- **imbalanced-learn** — SMOTE oversampling
- **matplotlib, seaborn** — visualization
- **Tableau Public** — interactive dashboard

---

## 👤 Author

**Arpit**  
[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [GitHub](https://github.com/YOUR_USERNAME)
