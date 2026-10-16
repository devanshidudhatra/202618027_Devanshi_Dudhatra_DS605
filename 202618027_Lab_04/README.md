# DS605 Lab 04 — Online Shoppers Purchasing Intention

## Overview

This notebook builds a machine-learning pipeline to predict whether an online shopping session will result in a purchase. The target column is `Revenue`.

The notebook uses the Online Shoppers Purchasing Intention dataset and applies scikit-learn preprocessing tools before training a Logistic Regression classifier. It also explores class imbalance, prediction thresholds, and improvements to the pipeline.

## Objectives

- Load and inspect the training and test datasets.
- Check data types, missing values, and target-class distribution.
- Separate the target (`Revenue`) and exclude the identifier (`Session_ID`) from model features.
- Apply suitable preprocessing to numeric and categorical features.
- Build preprocessing and classification pipelines.
- Evaluate Logistic Regression on validation data.
- Compare a standard model with a class-weight-balanced model.
- Explore different probability thresholds.
- Improve the pipeline using cross-validation and hyperparameter tuning.
- Generate `submission.csv` with checkpoint information and test predictions.

## Dataset

The notebook reads two CSV files:

- `train.csv` — training data, including the `Revenue` target column.
- `test.csv` — test data for which predictions are generated.

Place both files in the same directory as the notebook, or update the file paths in the code.

The dataset contains session-related features such as page visits and durations, bounce and exit rates, page values, month, visitor type, weekend status, and other session categories.

## Main Workflow

### 1. Load and inspect data

The notebook checks the dataset shape, data types, missing-value counts, and target-class balance. It also plots the distribution of `Revenue` and explores the relationship between missing values and the target.

### 2. Prepare features

- `Revenue` is the prediction target.
- `Session_ID` is an identifier and is excluded from model features.
- Features are grouped into numeric and categorical columns.
- Integer-coded categories such as `Region`, `TrafficType`, `OperatingSystems`, `Browser`, and `Weekend` are treated as categories rather than continuous measurements.

### 3. Preprocess the data

The notebook uses scikit-learn tools such as:

- `SimpleImputer` to handle missing values.
- `StandardScaler` to scale numeric features.
- `OneHotEncoder` to encode categorical features.
- `ColumnTransformer` to apply transformations to different feature groups.
- `Pipeline` to combine preprocessing and model training.

### 4. Train and evaluate Logistic Regression

The notebook uses a stratified train-validation split with `test_size=0.2` and `random_state=42`.

It evaluates a standard Logistic Regression model and another model using `class_weight="balanced"`.

Evaluation includes accuracy, classification reports, confusion matrices, and experiments with different probability thresholds.

### 5. Improve the pipeline

The later cells explore a more complete pipeline that handles missing values, categorical variables, class imbalance, and hyperparameter tuning with stratified cross-validation and F1 scoring.

### 6. Generate output

The submission cell creates `submission.csv`. It contains checkpoint information such as missing-value counts, row and column counts, row-retention percentage, and the final model name, followed by test-session IDs and predicted `Revenue` labels.

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

## Project Structure

```text
project/
├── 202618027_Lab_04.ipynb
├── train.csv
└── test.csv
```

After running the notebook, `submission.csv` should also appear in the working directory.

## How to Run

1. Install the required libraries.
2. Place `train.csv` and `test.csv` in the notebook directory.
3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `202618027_Lab_04.ipynb`.
5. Run the notebook cells in order, from top to bottom.
6. Review the evaluation metrics and plots.
7. Confirm that `submission.csv` was created.

## Reproducibility

The validation split uses `random_state=42` and stratification to preserve the target-class proportions in both subsets. Scikit-learn pipelines combine preprocessing and model training.

## Expected Outcome

The notebook demonstrates an end-to-end classification workflow for predicting online shopping purchase intention. It shows how preprocessing, correct treatment of categorical features, class-imbalance handling, probability thresholds, and model tuning can affect classification performance.

**Note:** Actual evaluation scores and successful output generation depend on the supplied CSV files and successful execution of the notebook.
