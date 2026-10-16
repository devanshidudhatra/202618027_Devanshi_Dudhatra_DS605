# Assignment 07 — Linear Regression and Logistic Regression

## Overview

This assignment implements and compares **Linear Regression** and **Logistic Regression** using a garment worker productivity dataset. Each algorithm is implemented in two ways:

1. Using Scikit-learn libraries.
2. From scratch using basic Python programming.

The assignment compares training time, prediction time, and model performance.

## Dataset

**Dataset:** `garments_worker_productivity.csv`

The dataset contains information about garment production, including employee productivity, working time, incentives, idle time, and number of workers.

### Features Used

- `targeted_productivity`
- `smv`
- `wip`
- `over_time`
- `incentive`
- `idle_time`
- `idle_men`
- `no_of_style_change`
- `no_of_workers`

**Target variable:** `actual_productivity`

## Implementations

### 1. Linear Regression

Linear Regression predicts the continuous value of actual productivity.

**Library implementation**
- Uses Scikit-learn's `LinearRegression`.
- Measures training and prediction time.
- Calculates MAE, RMSE, and R² score.

**From-scratch implementation**
- Standardizes numerical features using the training data.
- Initializes model parameters.
- Uses gradient descent to update weights and bias.
- Calculates MAE, RMSE, and R² score.

### 2. Logistic Regression

Logistic Regression classifies productivity into two categories:

- **Class 1:** Actual productivity is greater than or equal to 0.5.
- **Class 0:** Actual productivity is less than 0.5.

**Library implementation**
- Uses Scikit-learn's `LogisticRegression`.
- Standardizes the features.
- Measures training time, prediction time, and accuracy.

**From-scratch implementation**
- Standardizes numerical features.
- Uses the sigmoid function to calculate probabilities.
- Applies gradient descent to update weights and bias.
- Uses a probability threshold of 0.5 to classify observations.
- Calculates classification accuracy.

## Methodology

1. Load the CSV dataset using Python's `csv` module.
2. Select the input features and target variable.
3. Exclude rows containing missing values in the selected features or target.
4. Split the data into 80% training data and 20% testing data.
5. Use `random.seed(42)` to make the split reproducible.
6. Train and evaluate each model.
7. Compare execution time and performance metrics.

## Results

### Linear Regression

| Metric | Library | From Scratch |
|---|---:|---:|
| Training time (seconds) | 0.06950 | 2.10766 |
| Prediction time (seconds) | 0.00191 | 0.00027 |
| MAE | 0.05001 | 0.05001 |
| RMSE | 0.08709 | 0.08709 |
| R² score | 0.69242 | 0.69242 |

### Logistic Regression

| Metric | Library | From Scratch |
|---|---:|---:|
| Training time (seconds) | 0.09471 | 3.01740 |
| Prediction time (seconds) | 0.00338 | 0.00055 |
| Accuracy | 93.53% | 91.37% |

*Execution times may vary depending on the computer and runtime environment.*

## Conclusion

Both approaches produce similar Linear Regression results, with an R² score of approximately 0.6924. The Scikit-learn implementation trains much faster than the from-scratch implementation in this run.

For Logistic Regression, the library implementation achieves approximately 93.53% accuracy, while the from-scratch implementation achieves approximately 91.37%. This demonstrates how standard ML libraries simplify model training, while a from-scratch implementation helps explain the underlying algorithms.

## Requirements

- Python 3
- Jupyter Notebook
- Scikit-learn (for the library implementations)

The notebook also uses Python's built-in `csv`, `random`, `time`, and `math` modules.

## How to Run

1. Place `202618027_Assignment07.ipynb` and `garments_worker_productivity.csv` in the same directory.
2. Install the required dependency:

   ```bash
   pip install scikit-learn jupyter
   ```

3. Open Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `202618027_Assignment07.ipynb`.
5. Run the cells sequentially to train the models and display the results.

**Note:** The CSV dataset must be available at the path specified in the notebook for execution to succeed.
