# Credit-Risk-Prediction-Using-Machine-Learning
Machine learning project for predicting loan default risk using Lending Club data, with SMOTE, Logistic Regression, Random Forest, threshold analysis, and model evaluation.
# Project Overview
This project uses machine learning to predict whether a loan will eventually default based on information available at the time of loan issuance.

The analysis uses Lending Club loan data and compares two classification models: Logistic Regression and Random Forest.
# Objective
The main objective is to answer:
Can we predict whether a loan will eventually default using information available when the loan is issued?
# Dataset
The dataset contains Lending Club loan records from 2007 to 2014.

The original dataset contains approximately 466,000 loan records and 75 variables.

For the machine learning analysis, only loans with resolved outcomes were used:

- **Fully Paid → 0 (Non-default)**
- **Charged Off → 1 (Default)**
- **Default → 1 (Default)**

This resulted in **228,046 resolved loans** being used for modelling.

Ongoing loans were excluded because their final repayment outcome was not yet known.

# Data Leakage Prevention
Variables containing information generated after loan origination were excluded from the model.

Examples include:

- Total payments
- Recoveries
- Recovery fees
- Repayment information
- Last payment information
- Outstanding principal

This was done to ensure that the model only used information that would have been available when the loan was issued.

# Features
The model uses loan and borrower characteristics available at origination, including:

- Loan amount
- Funded amount
- Interest rate
- Installment
- Annual income
- Debt-to-income ratio (DTI)
- Delinquencies
- Credit inquiries
- Open accounts
- Revolving balance
- Revolving utilization
- Total accounts
- Current balance
- Loan term
- Loan grade and sub-grade
- Employment length
- Home ownership
- Verification status
- Loan purpose
- Application type
- Initial listing status

# Methodology
The project follows the following workflow:

1. Data preparation
2. Target variable creation
3. Data leakage prevention
4. Feature selection
5. Time-based train-test split
6. Missing-value imputation
7. Feature scaling
8. One-hot encoding of categorical variables
9. SMOTE for class imbalance
10. Model training
11. Model evaluation
12. Threshold analysis
13. Model interpretation

# Train-Test Split
A time-based split was used to simulate predicting future loans.

- **Training data:** Loans issued before 2014 — 158,739 observations
- **Test data:** Loans issued during 2014 — 69,307 observations

SMOTE was applied only to the training data. The test data retained its original class distribution.
# Models
## 1. Logistic Regression
Logistic Regression was used as an interpretable baseline classification model.

## 2. Random Forest
Random Forest was used to capture potential nonlinear relationships and interactions between features.

# Model Evaluation
Because default loans represent a minority class, accuracy alone was not used as the primary evaluation measure.

The models were evaluated using:
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

# Results
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.697 | 0.369 | 0.324 | 0.646 | 0.431 |
| Random Forest | 0.683 | 0.341 | 0.415 | 0.174 | 0.245 |

At the standard 0.5 classification threshold, Logistic Regression achieved higher ROC-AUC, PR-AUC, recall and F1-score, while Random Forest achieved higher precision.

# Threshold Analysis
Classification thresholds were varied to examine the trade-off between identifying defaults and generating false positives.

For Logistic Regression:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.30 | 0.249 | 0.921 | 0.392 |
| 0.40 | 0.282 | 0.815 | 0.419 |
| 0.50 | 0.324 | 0.646 | 0.431 |
| 0.60 | 0.375 | 0.449 | 0.409 |
| 0.70 | 0.436 | 0.242 | 0.311 |

Lowering the threshold increased recall while reducing precision. At a threshold of 0.30, Logistic Regression identified approximately 92% of actual defaults, but precision decreased to approximately 25%.

For Random Forest, lowering the threshold from 0.50 to 0.30 increased recall from approximately 17% to 70%, while precision decreased from approximately 42% to 30%.

# Model Interpretation

## Random Forest Feature Importance
Among the features identified as important by the Random Forest model were:

- Interest rate
- Annual income
- Total current balance
- DTI
- Revolving utilization
- Recent credit inquiries
- Revolving balance
- Total accounts
- Open accounts
- Employment length

Feature importance indicates how useful a variable was to the model's predictions; it does not imply causation.

## Logistic Regression Coefficients

Logistic Regression coefficients were examined to understand the direction of the model's associations with the default prediction.

Positive coefficients push the model's prediction toward the default class, while negative coefficients push it toward the non-default class, relative to the relevant reference category for categorical variables.

# Key Takeaways
- Logistic Regression showed slightly stronger overall discrimination than Random Forest on the test data.
- Logistic Regression detected a substantially larger share of actual defaults at the standard 0.5 threshold.
- Random Forest had higher precision but substantially lower recall at the 0.5 threshold.
- Classification threshold selection materially affects the trade-off between default detection and false positives.
- Interest rate, income, DTI, credit utilization and other borrower/loan characteristics were among the features used by the models in predicting default.

# Tools & Libraries
- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn

# Project Structure
```text
Credit-Risk-Prediction-Using-Machine-Learning/
│
├── Credit_Risk_Prediction.ipynb
├── README.md
└── data/
    └── README.md
