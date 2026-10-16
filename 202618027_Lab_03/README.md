# DS605 Lab 03 — Hotel Booking Cancellation Prediction

## Overview

This lab uses machine learning to predict whether a hotel booking will be canceled. The notebook uses the `hotel_bookings.csv` dataset and compares Logistic Regression and Decision Tree classifiers with two different numerical scaling methods.

## Tasks Covered

### Part A — Data Loading and Preprocessing

- Loads and explores the dataset using `head()`, `shape`, `info()`, `describe()`, and `dtypes`.
- Checks the class distribution of `is_canceled`.
- Separates the target variable and input features.
- Identifies numerical and categorical columns.
- Checks missing values and their percentages.
- Removes selected columns with high missingness.
- Removes `reservation_status` and `reservation_status_date` to prevent target leakage.
- Uses boxplots and the IQR method to inspect outliers.
- Splits the data into training and testing sets using an 80:20 ratio.

### Part B — Preprocessing Pipelines

Two preprocessing pipelines are created.

**Pipeline A — StandardScaler**
- Uses KNN imputation for missing numerical values.
- Applies StandardScaler to numerical features.
- Fills missing categorical values using the most frequent value.
- Converts categorical features using OneHotEncoder.

**Pipeline B — MinMaxScaler**
- Uses KNN imputation for missing numerical values.
- Applies MinMaxScaler to numerical features.
- Uses the same categorical imputation and encoding steps as Pipeline A.

### Model Training and Evaluation

The following four model configurations are compared:

1. Logistic Regression with StandardScaler
2. Logistic Regression with MinMaxScaler
3. Decision Tree with StandardScaler
4. Decision Tree with MinMaxScaler

The evaluation includes:

- Training accuracy
- Testing accuracy
- Precision
- Recall
- F1 score
- Confusion matrices
- Training and testing accuracy differences

The notebook identifies the best configuration for each classifier and the best overall model based on the F1 score.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Installation

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Dataset Setup

The notebook requires `hotel_bookings.csv`.

Place the dataset in the same directory as the notebook, or update the file path in the `pd.read_csv()` statement.

Recommended project structure:

```text
project/
├── 202618027_Lab_03.ipynb
├── README.md
└── hotel_bookings.csv
```

## How to Run

1. Install the required libraries.
2. Place `hotel_bookings.csv` in the project directory.
3. Open a terminal in that directory.
4. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `202618027_Lab_03.ipynb`.
6. Run all cells from top to bottom.

## Reproducibility

The train-test split uses:

- Test size: 20%
- Stratification: `stratify=y`
- Random state: `42`

The same split and model settings are used for all four experiments to make the comparison consistent.

## Conclusion

This lab demonstrates how preprocessing choices can affect machine-learning classification performance. Logistic Regression and Decision Tree models are evaluated using the same dataset split, and their performance is compared using accuracy, precision, recall, F1 score, and confusion matrices.

The best model should be selected based on the evaluation results generated when the notebook is executed.

## Dataset Note

Before uploading the project to a public GitHub repository, check whether the dataset's license permits redistribution. If necessary, provide instructions for obtaining the dataset instead of uploading the CSV file.