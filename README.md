# Breast Cancer Wisconsin ML Project

## Project Overview

This project applies classical machine learning models to the **Breast Cancer Wisconsin Diagnostic dataset** to classify breast cancer samples as **malignant** or **benign**.

The task is a **supervised binary classification problem**. Each sample is described by **30 numerical features** extracted from digitized images of cell nuclei, and the target variable represents the diagnosis:

- **0 = malignant**
- **1 = benign**

The main goal is not only to obtain high predictive performance, but also to evaluate the models using clinically meaningful metrics, especially **malignant recall**, because missing malignant cases can be clinically important.

---

## Repository Contents

| File | Description |
|---|---|
| `breast-cancer-wisconsin-ml-project.ipynb` | Main Jupyter notebook containing the full code, EDA, baselines, modelling, evaluation, and outputs |
| `Breast_Cancer_ML_Enhanced_Final_Report.pdf` | Final self-contained report summarizing the project, methods, results, figures, and interpretation |
| `requirements.txt` | Python dependencies needed to run the notebook |
| `.gitignore` | Files and folders excluded from version control |

---

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin Diagnostic dataset**, available through `sklearn.datasets.load_breast_cancer`.

It contains:

- **569 samples**
- **30 numerical features**
- **2 diagnosis classes:** malignant and benign

The class distribution is moderately imbalanced:

| Class | Number of samples | Percentage |
|---|---:|---:|
| Benign | 357 | 62.74% |
| Malignant | 212 | 37.26% |

Because this is a medical classification task, **malignant recall** is especially important. A false negative means that a malignant sample is incorrectly predicted as benign.

---

## Project Workflow

This project follows a complete applied machine learning workflow:

1. **Problem Definition**  
   The task was defined as a supervised binary classification problem: predicting whether a breast cancer sample is malignant or benign.

2. **Dataset Loading**  
   The Breast Cancer Wisconsin Diagnostic dataset was loaded from scikit-learn and converted into a Pandas DataFrame.

3. **Exploratory Data Analysis**  
   Missing values, class balance, feature ranges, selected feature distributions, and feature correlations were analyzed.

4. **Feature and Target Separation**  
   The 30 numerical features were stored in `X`, and the diagnosis labels were stored in `y`. Target-related columns were excluded from the input features to avoid data leakage.

5. **Train-Test Split**  
   The data was split into training and test sets using stratification. The test set was kept unseen until the final evaluation.

6. **Rule-Based Non-ML Baselines**  
   Simple interpretable rule-based baselines were tested before training machine learning models. A one-feature baseline used `mean concave points`, and an optional two-feature exploratory baseline used `mean concave points` and `mean area`. Thresholds were selected using only the training set.

7. **Model Comparison with Cross-Validation**  
   Logistic Regression, KNN, SVM, Decision Tree, and Random Forest were compared using 5-fold stratified cross-validation on the training data only.

8. **Pipeline-Based Preprocessing**  
   Pipelines were used to combine feature scaling and model training. This ensured that scaling was fitted only inside each cross-validation fold and helped avoid data leakage.

9. **Hyperparameter Tuning**  
   GridSearchCV was applied to Logistic Regression and SVM using 5-fold stratified cross-validation.

10. **Final Model Selection**  
    Logistic Regression was selected as the final model because it achieved strong cross-validation performance, high malignant recall, and good interpretability.

11. **Final Test Evaluation**  
    The final model was evaluated once on the unseen test set using accuracy, balanced accuracy, precision, recall, F1-score, MCC, ROC-AUC, and the confusion matrix.

12. **Medical Error Analysis**  
    Special attention was given to malignant recall and false negatives because missing malignant cases can be clinically important.

---

## Rule-Based Non-ML Baselines

Before training machine learning models, simple rule-based baselines were tested as interpretable reference points.

### One-Feature Baseline

The first baseline used only **mean concave points**. A threshold was selected using the training set, and samples above the threshold were predicted as malignant.

| Metric | Value |
|---|---:|
| Accuracy | 0.9035 |
| Balanced accuracy | 0.9137 |
| Malignant precision | 0.8163 |
| Malignant recall | 0.9524 |
| Malignant F1-score | 0.8791 |
| MCC | 0.8062 |

Confusion matrix:

| Actual / Predicted | Predicted malignant | Predicted benign |
|---|---:|---:|
| Actual malignant | 40 | 2 |
| Actual benign | 9 | 63 |

This baseline was simple and interpretable, but it used only one feature and could not combine information from the other 29 variables.

### Optional Two-Feature Exploratory Baseline

An additional exploratory baseline used **mean concave points** and **mean area**. The rule predicted a sample as malignant if either feature was above its selected threshold. The thresholds were selected using only the training set.

This baseline was included only as an interpretable reference and was **not** used for final model selection.

| Metric | Value |
|---|---:|
| Accuracy | 0.9035 |
| Balanced accuracy | 0.9236 |
| Malignant precision | 0.7925 |
| Malignant recall | 1.0000 |
| Malignant F1-score | 0.8842 |
| MCC | 0.8194 |

Confusion matrix:

| Actual / Predicted | Predicted malignant | Predicted benign |
|---|---:|---:|
| Actual malignant | 42 | 0 |
| Actual benign | 11 | 61 |

The two-feature baseline achieved perfect malignant recall on the test set, meaning that it did not miss any malignant samples. However, this came at the cost of more benign samples being incorrectly predicted as malignant. This illustrates the trade-off between reducing false negatives and increasing false positives.

---

## Models Compared

The following machine learning models were compared:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest

Model comparison was performed using **5-fold stratified cross-validation** on the training data only. The test set was kept unseen until the final evaluation.

---

## Final Model

The final selected model was **Logistic Regression** with Pipeline-based preprocessing.

Best parameters:

| Parameter | Value |
|---|---|
| `C` | 0.1 |
| `penalty` | l2 |
| `solver` | lbfgs |

Logistic Regression was selected because it achieved strong cross-validation performance, high malignant recall, good interpretability, and a better overall balance than the simple rule-based baselines.

---

## Final Test Results

The final model was evaluated on the independent test set.

| Metric | Value |
|---|---:|
| Accuracy | 0.9737 |
| Balanced accuracy | 0.9692 |
| Benign precision | 0.9726 |
| Benign recall | 0.9861 |
| Benign F1-score | 0.9793 |
| Malignant precision | 0.9756 |
| Malignant recall | 0.9524 |
| Malignant F1-score | 0.9639 |
| MCC | 0.9433 |
| ROC-AUC | 0.9957 |

---

## Final Confusion Matrix

| Actual / Predicted | Predicted malignant | Predicted benign |
|---|---:|---:|
| Actual malignant | 40 | 2 |
| Actual benign | 1 | 71 |

The final Logistic Regression model produced **2 false negatives** for the malignant class, meaning that **2 malignant samples were incorrectly predicted as benign**. This is clinically important because false negatives may correspond to missed cancer cases.

Compared with the rule-based baselines, the final Logistic Regression model used all **30 numerical features** and achieved a better overall balance between malignant recall, false positives, F1-score, MCC, and interpretability.

---

## Main Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn

---

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook breast-cancer-wisconsin-ml-project.ipynb
```

---

## Report

A complete report is available in:

```text
Breast_Cancer_ML_Enhanced_Final_Report.pdf
```

The report contains the full project explanation, major code outputs, preprocessing details, cross-validation settings, hyperparameter grids, figures, final metrics, classification report, discussion, limitations, future work, and LLM usage declaration.

---

## LLM Usage Declaration

LLMs were used for code debugging, language polishing, refactoring, and improving the clarity of explanations. All methodological choices, experiments, outputs, and interpretations were reviewed, checked, and validated by the author.
