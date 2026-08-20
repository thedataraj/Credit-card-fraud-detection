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
