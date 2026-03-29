# 💳 SBA Loan Default Prediction System  
### (Logistic Regression • Random Forest • XGBoost • Voting • Stacking)

---

## 📌 Project Overview

This project presents a complete machine learning pipeline for predicting loan defaults using the **U.S. Small Business Administration (SBA)** dataset.

The objective is to build a robust and interpretable credit risk model capable of identifying high-risk loans using both **individual models** and **ensemble learning techniques**.

The project evaluates multiple machine learning approaches and applies **hyperparameter tuning** to optimize performance.

---

## 🎯 Problem Statement

Loan default prediction is a core challenge in financial risk management. Traditional scoring methods often fail to capture complex nonlinear patterns in borrower behavior.

This project aims to:
- Predict whether a loan will default or not
- Compare multiple machine learning models
- Improve predictive performance using ensemble learning
- Optimize models using hyperparameter tuning

---

## 📊 Dataset

The dataset is sourced from the **Small Business Administration (SBA)** and includes information about:
- Loan amounts and terms
- Business characteristics
- Borrower and financial attributes
- Loan outcome (default / non-default)

### Target Variable:
- **CHGOFF** → Indicates loan default status

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Feature selection
- Removal of data leakage variables

---

### 2. Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset and improve model learning on minority class (defaults)

---

### 3. Models Developed

The following models were trained and evaluated:

#### 🟢 Baseline Model
- Logistic Regression

#### 🔵 Tree-Based Model
- Random Forest Classifier

#### 🟣 Advanced Model
- XGBoost Classifier (primary model)

---

### 4. Hyperparameter Tuning

- RandomizedSearchCV used for optimization
- 3-fold cross-validation
- Tuned parameters for XGBoost to improve generalization and reduce overfitting

---

### 5. Ensemble Learning

To further improve performance, two ensemble methods were implemented:

- **Voting Classifier (Soft Voting)**
- **Stacking Classifier (Meta-learning approach)**

These models combined predictions from multiple base learners to enhance robustness.

---

## 🧠 Final Model Selection

The **XGBoost model with tuned hyperparameters** was selected as the final model due to its superior balance between predictive power and stability.

---

## 📈 Model Performance

### 🏆 Final XGBoost Results

- **Accuracy:** 0.93  
- **ROC-AUC:** 0.973  
- **Recall (Default Class):** 0.90  
- **Precision (Default Class):** 0.76  
- **F1-Score:** 0.82  

---

### 📊 Model Comparison Summary

| Model                | Performance Insight |
|---------------------|---------------------|
| Logistic Regression | Baseline benchmark, lower predictive power |
| Random Forest       | Strong non-linear performance |
| XGBoost             | Best overall performance |
| Voting Classifier   | Improved stability through ensemble averaging |
| Stacking Classifier | Strong performance but slightly below XGBoost |

---

## 📊 Evaluation Visualizations

The following evaluation plots were generated:

- ROC Curve (model discrimination ability)
- Precision-Recall Curve (class imbalance performance)
- Confusion Matrix (prediction outcomes)
- Normalized Confusion Matrix (class-wise performance)
- Feature Importance (XGBoost interpretability)

📁 All visualizations are stored in the `/images` folder.

---

## 🧠 Key Insights

- XGBoost outperformed all individual and ensemble models in terms of ROC-AUC and recall balance
- Ensemble methods (Voting and Stacking) provided competitive but not superior performance
- Hyperparameter tuning significantly improved model generalization
- Recall of 0.90 ensures strong detection of loan defaults, which is critical in credit risk applications

---

## 🚀 Deployment

A **Streamlit web application** was developed to demonstrate real-time prediction of loan default risk.

### Features:
- User input interface for loan and borrower attributes
- Real-time default probability prediction
- Risk classification (High / Low risk)
- Interactive visual feedback

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Streamlit

---

## 📂 Project Structure
