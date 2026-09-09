# Loan Approval Prediction

This project predicts whether a loan application will be approved or rejected using machine learning and a PyTorch neural network.

The project compares multiple models and evaluates their performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

## Project Overview

The goal of this project is to build and compare machine learning models for predicting loan approval.

The following models are included:

- Logistic Regression
- Random Forest
- XGBoost
- Neural Network (PyTorch)

The final models are compared using validation and test data.

## Dataset

The dataset contains information about loan applicants, including features such as:

- Age
- Income
- Loan Amount
- Credit Score
- Years of Experience
- Gender
- Education
- City
- Employment Type

The target variable is:

- `LoanApproved`
  - `0`: Not Approved
  - `1`: Approved

## Project Workflow

The project follows these main steps:

1. Load and explore the dataset
2. Check data types and missing values
3. Analyze the target distribution
4. Separate features and target
5. Split the data into training, validation, and test sets
6. Preprocess numerical and categorical features
7. Train machine learning baseline models
8. Train a PyTorch neural network
9. Tune the Neural Network classification threshold using validation data
10. Compare all models
11. Evaluate the models on the final test set
12. Analyze feature importance
13. Demonstrate predictions for a new applicant
14. Use SHAP for global and local model explanations

## Data Preprocessing

Numerical features are:

- Imputed using the median
- Standardized using `StandardScaler`

Categorical features are:

- Imputed using the most frequent value
- Encoded using One-Hot Encoding

## Neural Network

The PyTorch neural network uses the following architecture:

- Input layer
- Dense layer with 64 neurons and ReLU activation
- Dropout (0.30)
- Dense layer with 32 neurons and ReLU activation
- Dropout (0.20)
- Output layer

The model is trained using:

- BCEWithLogitsLoss
- Class weighting for class imbalance
- Adam optimizer
- Early stopping based on validation ROC-AUC

## Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The F1-score is used to compare the models during validation.

## Feature Importance and Explainability

Feature importance is analyzed for:

- Random Forest
- XGBoost

SHAP is also used to explain the Random Forest model.

The project includes:

- Global SHAP feature importance
- Local explanations for individual applicants
- SHAP explanation tables for selected approved and rejected applicants

## Project Structure

```text
Loan-Approval-Prediction/
│
├── data/
│   └── loan_risk_prediction_dataset.csv
│
├── notebooks/
│   └── loan_approval_deep_learning_project.ipynb
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── neural_network_checkpoint.pth
│   ├── neural_network_config.json
│   └── neural_network_preprocessor.pkl
│
├── outputs/
│   ├── test_results.csv
│   ├── random_forest_feature_importance.csv
│   ├── xgboost_feature_importance.csv
│   │
│   └── shap/
│       ├── shap_summary_random_forest.png
│       ├── shap_accepted_applicant_explanation.csv
│       └── shap_rejected_applicant_explanation.csv
│
├── requirements.txt
└── README.md
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- PyTorch
- SHAP
- Joblib

## Author

Sara Satariamnab