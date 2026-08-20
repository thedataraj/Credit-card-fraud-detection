# Credit Card Fraud Detection Using Machine Learning

A complete Machine Learning project for detecting fraudulent credit card transactions using Python, Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

---

## Project Overview

Credit card fraud is an important problem in the financial industry. Every day, a large number of credit card transactions are processed, and only a very small percentage of them may be fraudulent.

The main challenge is that fraudulent transactions are extremely rare compared with legitimate transactions. Because of this class imbalance, a model can achieve very high accuracy while still failing to identify fraudulent transactions.

The objective of this project is to build a Machine Learning classification model that can identify fraudulent transactions while focusing on metrics such as Precision, Recall, F1-Score, ROC-AUC, and PR-AUC rather than relying only on accuracy.

In this project, I developed an end-to-end fraud detection workflow covering:

- Data loading
- Data understanding
- Data quality checking
- Duplicate removal
- Exploratory Data Analysis
- Class imbalance analysis
- Feature preprocessing
- Outlier handling
- Power transformation
- Machine Learning model development
- Hyperparameter tuning
- Cross-validation
- Model evaluation
- Confusion matrix analysis
- Business interpretation

---

# Project Objectives

The main objectives of this project are:

1. Understand the structure of credit card transaction data.
2. Identify data quality issues.
3. Remove duplicate records.
4. Analyze the distribution of legitimate and fraudulent transactions.
5. Handle highly imbalanced target classes.
6. Apply appropriate preprocessing techniques.
7. Build a Machine Learning classification model.
8. Optimize model hyperparameters.
9. Evaluate the model using fraud-focused metrics.
10. Understand the business impact of false positives and false negatives.

---

# Project Summary

| Category | Details |
|---|---|
| Project Type | Machine Learning Classification |
| Domain | Finance / FinTech |
| Problem | Credit Card Fraud Detection |
| Programming Language | Python |
| Dataset Size | 284,807 transactions |
| Original Columns | 31 |
| Features | 30 |
| Target Variable | Class |
| Model | Logistic Regression |
| Cross-Validation | 5-Fold Stratified Cross-Validation |
| Hyperparameter Tuning | GridSearchCV |
| Optimization Metric | Average Precision / PR-AUC |

---

# Dataset Summary

The dataset contains credit card transaction information with a binary target variable called `Class`.

The target variable represents whether a transaction is legitimate or fraudulent.

### Target Classes

- `Class = 0` → Legitimate transaction
- `Class = 1` → Fraudulent transaction

The original dataset contains:

- **284,807 transactions**
- **31 columns**
- **30 input features**
- **1 target variable**

After checking the dataset, **1,081 duplicate records** were removed.

The final dataset contained:

**283,726 transactions**

---

# Dataset Features

The dataset contains the following major types of variables:

### Time

Represents the time-related information associated with a transaction.

### Amount

Represents the transaction amount.

### V1–V28

These are anonymized PCA-transformed transaction features.

The PCA-based features are already transformed and anonymized, so they were retained as model inputs.

### Class

This is the target variable.

```text
0 = Legitimate
1 = Fraudulent

```
# Data Quality Analysis

Before building the Machine Learning model, I performed several data quality checks.

### Missing Values

No missing values were detected across the dataset.

### Duplicate Records

The dataset initially contained **1,081 duplicate records**.

These duplicate records were removed before model training.

### Final Dataset

After removing duplicates, the dataset contained:

**283,726 transactions**

### Data Validation

The following checks were performed:

- Missing value detection
- Duplicate detection
- Infinite value detection
- Data type inspection
- Statistical summary
- Class distribution analysis


---

# Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the characteristics of the transaction data before building the Machine Learning model.

The analysis focused on:

- Distribution of legitimate and fraudulent transactions
- Transaction amount distribution
- Time-based transaction patterns
- PCA feature distributions
- Feature variance
- Skewness
- Outliers
- Correlation and feature relationships

EDA helped identify the highly imbalanced nature of the target variable and the skewness present in the `Amount` feature.

---

# Class Imbalance

One of the biggest challenges in this project was the severe imbalance between legitimate and fraudulent transactions.

Fraudulent transactions represented only approximately **0.1727%** of the dataset.

This means that the vast majority of transactions were legitimate.

Because of this imbalance, accuracy alone is not a reliable metric for evaluating the fraud detection model.

For example, a model could classify almost every transaction as legitimate and still achieve very high accuracy while failing to detect fraudulent transactions.

Therefore, this project focuses on:

- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC
- Confusion Matrix

---

# Data Preprocessing

A feature-specific preprocessing pipeline was developed using Scikit-learn.

The preprocessing workflow was:

```text
Raw Data
   ↓
Train-Test Split
   ↓
Missing Value Handling
   ↓
IQR-Based Outlier Clipping
   ↓
Yeo-Johnson Transformation
   ↓
Feature Processing
   ↓
Machine Learning Model

IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR

```

# Model Development
---
# Machine Learning Model

The primary Machine Learning algorithm used in this project was:

## Logistic Regression

Logistic Regression was selected as the classification algorithm because the target variable contains two classes:

```text
0 = Legitimate
1 = Fraudulent

class_weight = balanced
penalty = l2
solver = liblinear
max_iter = 500

class_weight="balanced"
```

# Hyperparameter Tuning

GridSearchCV was used to identify the best Logistic Regression configuration.

The following `C` values were evaluated:

```text
0.01
0.1
1
10

n_splits = 5
shuffle = True
random_state = 42

C = 1
class_weight = balanced
penalty = l2
solver = liblinear
max_iter = 500


```
# Model Evaluation

The final model was evaluated on unseen test data.

| Metric | Score |
|---|---:|
| Accuracy | **97.47%** |
| Balanced Accuracy | **92.43%** |
| Precision | **5.51%** |
| Recall | **87.37%** |
| F1-Score | **10.37%** |
| ROC-AUC | **0.9620** |
| PR-AUC | **0.6705** |


# Confusion Matrix

The final confusion matrix was:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| **Actual Legitimate** | **55,229** | **1,422** |
| **Actual Fraud** | **12** | **83** |

### Interpretation

- **True Negative (TN): 55,229** — legitimate transactions correctly classified.
- **False Positive (FP): 1,422** — legitimate transactions incorrectly flagged as fraud.
- **False Negative (FN): 12** — fraudulent transactions missed by the model.
- **True Positive (TP): 83** — fraudulent transactions correctly detected.

The model detected **83 out of 95 fraudulent transactions**, resulting in **87.37% recall**.


# Business Interpretation

The model demonstrates strong fraud detection capability in terms of recall.

The **87.37% recall** means that the model identified most of the fraudulent transactions in the test dataset.

However, the **5.51% precision** indicates that the model also generated a significant number of false alerts.

This creates an important business trade-off:

```text
Higher Recall
     ↓
More Fraud Detected
     ↓
More False Positives
     ↓
More Manual Investigations

```

# Future Improvements

The project can be further improved through:

### 1. Threshold Optimization

Test different probability thresholds to find a better Precision-Recall balance.

### 2. Advanced Machine Learning Models

Compare Logistic Regression with:

- Random Forest
- XGBoost
- LightGBM
- Gradient Boosting
- Ensemble Models

### 3. Explainable AI

Use SHAP or similar techniques to understand the factors influencing fraud predictions.

### 4. Real-Time Fraud Detection

Deploy the model through an API and integrate it with a real-time transaction processing system.

### 5. Model Monitoring

Monitor:

- Data drift
- Model performance
- Fraud detection rate
- Precision
- Recall
- False-positive rate


# Conclusion

This project demonstrates an end-to-end Machine Learning approach to credit card fraud detection.

The final Logistic Regression model achieved:

- **97.47% Accuracy**
- **87.37% Recall**
- **0.9620 ROC-AUC**
- **0.6705 PR-AUC**

The model successfully detected **83 out of 95 fraudulent transactions** in the test dataset.

The project also demonstrates an important lesson in fraud detection: **accuracy alone is not sufficient** when dealing with highly imbalanced data.

Precision, Recall, PR-AUC, ROC-AUC, and the business impact of false positives and false negatives must all be considered when evaluating a fraud detection system.


# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook


# Author

## Raj Singh

Data Analytics | Python | SQL | Power BI | Machine Learning


## Project Keywords

`Python` `Machine Learning` `Data Analytics` `Fraud Detection` `Credit Card Fraud Detection` `Pandas` `NumPy` `Scikit-learn` `Logistic Regression` `GridSearchCV` `EDA` `Data Preprocessing` `Feature Engineering` `Imbalanced Classification` `Cross Validation` `Precision` `Recall` `F1 Score` `ROC-AUC` `PR-AUC` `Confusion Matrix` `FinTech`
