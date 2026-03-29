# 💳 Loan Default Risk Prediction System (XGBoost)

## 📌 Project Overview

This project develops a machine learning-based credit risk prediction system to identify the likelihood of loan defaults using the U.S. Small Business Administration (SBA) loan dataset.

The primary objective is to assist financial institutions in improving risk assessment, reducing default exposure, and supporting data-driven lending decisions.

---

## 🎯 Problem Statement

Loan default prediction is a critical task in financial risk management. Traditional rule-based systems fail to capture complex nonlinear relationships in borrower behavior.

This project applies advanced machine learning techniques to:
- Predict loan default probability
- Improve risk classification accuracy
- Support proactive financial decision-making

---

## 📊 Dataset

The dataset is derived from SBA loan records and includes:
- Loan characteristics
- Borrower information
- Financial indicators
- Loan outcome (Default / Non-default)

Target variable:
- `CHGOFF` (Loan Charge-off indicator)

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature scaling where necessary
- Removal of leakage variables
- Feature selection

### 2. Handling Class Imbalance
- SMOTE (Synthetic Minority Oversampling Technique)

### 3. Model Development
The following models were evaluated:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (primary model)
- Voting Classifier (ensemble)
- Stacking Classifier

### 4. Hyperparameter Tuning
- RandomizedSearchCV used for XGBoost optimization
- Cross-validation (3-fold) applied

### 5. Final Model Selection
The final model was selected based on:
- ROC-AUC score
- Recall for default class
- Overall generalization performance

---

## 🧠 Final Model: XGBoost

Best Hyperparameters:
```python
{
 'subsample': 0.9,
 'n_estimators': 300,
 'min_child_weight': 5,
 'max_depth': 8,
 'learning_rate': 0.1,
 'gamma': 0,
 'colsample_bytree': 0.8
}