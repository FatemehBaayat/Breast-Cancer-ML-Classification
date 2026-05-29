# Breast Cancer Classification Using Classical Machine Learning Models

## Project Overview

This project applies classical machine learning models to the Breast Cancer Wisconsin Diagnostic dataset to classify breast cancer samples as malignant or benign.

The task is a supervised binary classification problem. Each sample is described by 30 numerical features extracted from cell nuclei, and the target variable represents the diagnosis:

- 0 = malignant
- 1 = benign

The project follows an applied machine learning workflow including exploratory data analysis, preprocessing, model comparison, cross-validation, hyperparameter tuning, and final evaluation.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin Diagnostic dataset, available through scikit-learn.

It contains:

- 569 samples
- 30 numerical features
- 2 diagnosis classes: malignant and benign

## Methods

The following machine learning models were compared:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest

Model comparison was performed using 5-fold stratified cross-validation on the training data only. Pipelines were used to combine feature scaling and model training, reducing the risk of data leakage.

Hyperparameter tuning was performed using GridSearchCV for the strongest candidate models.

## Final Model

The final selected model was Logistic Regression with the following best parameters:

- C = 0.1
- penalty = l2
- solver = lbfgs

Logistic Regression was selected because it achieved strong cross-validation performance, high malignant recall, and good interpretability.

## Final Test Results

The final model was evaluated on an independent test set.

| Metric | Value |
|---|---:|
| Accuracy | 0.9737 |
| Balanced accuracy | 0.9692 |
| Benign F1-score | 0.9793 |
| Malignant F1-score | 0.9639 |
| Malignant recall | 0.9524 |
| MCC | 0.9433 |
| ROC-AUC | 0.9957 |

## Confusion Matrix

| Actual / Predicted | Predicted malignant | Predicted benign |
|---|---:|---:|
| Actual malignant | 40 | 2 |
| Actual benign | 1 | 71 |

The model produced 2 false negatives for the malignant class, meaning that 2 malignant samples were incorrectly predicted as benign. This is clinically important because false negatives may correspond to missed cancer cases.

## Main Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## LLM Usage Declaration

LLMs were used for code debugging, language polishing, refactoring, and improving the clarity of explanations. All methodological choices, experiments, outputs, and interpretations were reviewed, checked, and validated by the author.
