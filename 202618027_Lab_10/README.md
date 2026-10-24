# DS605: Fundamentals of Machine Learning
## Lab Assignment 8 — Decision Trees and Support Vector Machines

**Dataset:** UCI Banknote Authentication  
**Dataset size:** 1,372 samples, 4 numerical features, 2 classes

## Overview

This project implements Decision Trees and a linear Support Vector Machine (SVM) from scratch, compares the implementations with Scikit-learn, and explores nonlinear SVM kernels.

Two features, `variance` and `skewness`, are used for 2D decision-boundary visualizations. All four original features are used for the final model evaluation.

## Objectives

- Implement Gini impurity and entropy.
- Build a Decision Tree recursively using feature-threshold splits and stopping rules.
- Visualize the learned tree and its split rules.
- Compare manual and Scikit-learn Decision Trees using different criteria and maximum depths.
- Implement a linear soft-margin SVM using hinge loss and subgradient descent.
- Plot SVM decision boundaries, separating margins, and margin-violating samples.
- Explore the effects of regularization parameter `C`, learning rate, and iterations.
- Compare linear, polynomial, and RBF SVM kernels on nonlinear data.
- Evaluate models using accuracy, precision, recall, F1-score, training time, and prediction time.

## Dataset

The Banknote Authentication dataset is downloaded from the UCI Machine Learning Repository.

| Feature | Description |
|---|---|
| `variance` | Variance of the wavelet-transformed image |
| `skewness` | Skewness of the wavelet-transformed image |
| `curtosis` | Curtosis of the wavelet-transformed image |
| `entropy` | Entropy of the image |
| `class` | Binary target label |

Dataset source: [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

## Project Workflow

### Part A — Decision Tree from Scratch

- Implement Gini impurity and entropy using NumPy.
- Find the best feature-threshold split using impurity reduction.
- Construct the tree recursively with stopping rules.
- Predict class labels using the manual tree.
- Print the learned split rules.
- Visualize decision boundaries for the manual and Scikit-learn trees.
- Explore Gini and entropy with maximum depths of 2, 3, 4, and 5.

### Part B — Linear SVM from Scratch

- Use `variance` and `skewness` for 2D visualization.
- Standardize features using training-set statistics.
- Map class labels from `0` and `1` to `-1` and `+1`.
- Implement a linear soft-margin SVM using hinge loss and subgradient descent.
- Visualize the separating line, positive and negative margins, and margin violations.
- Compare the manual SVM with Scikit-learn's linear SVM.
- Experiment with different values of `C`, learning rates, and iteration counts.

### Part C — Kernel SVM

- Generate a nonlinear dataset using `make_moons`.
- Train Scikit-learn SVMs with linear, polynomial, and RBF kernels.
- Plot and compare their decision boundaries.
- Compare accuracy, F1-score, training time, and prediction time.

### Part D — Final Model Comparison

The final comparison uses all four Banknote features and the same train-test split for all models.

Models evaluated:

1. Manual Decision Tree
2. Scikit-learn Decision Tree
3. Manual Linear SVM
4. Scikit-learn Linear SVM
5. Scikit-learn RBF SVM

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Training time
- Prediction time

## Implementation Details

- Train-test split: 75% training and 25% testing.
- Random state: `42`.
- Stratified splitting preserves class proportions.
- Two features are used for 2D decision-boundary plots.
- All four features are used for final performance evaluation.
- Decision Trees use the original feature scales.
- SVM features are standardized using training-set statistics only.
- The manual Decision Tree and SVM algorithms are implemented without Scikit-learn.
- Scikit-learn is used for comparison models, evaluation metrics, data splitting, and nonlinear-kernel experiments.

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook or JupyterLab

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## How to Run

1. Clone or download the public GitHub repository.
2. Open `DS605_Lab8_Decision_Trees_and_SVMs.ipynb` in Jupyter Notebook or JupyterLab.
3. Install the required dependencies.
4. Run all notebook cells sequentially.
5. Ensure an internet connection is available to download the UCI dataset.

## Expected Outputs

The notebook generates:

- Dataset information and class counts.
- Manual Decision Tree split rules.
- Decision Tree criterion and depth comparison results.
- A visualization of the Scikit-learn Decision Tree.
- Manual and Scikit-learn Decision Tree decision-boundary plots.
- Linear SVM boundary and margin plots.
- SVM hyperparameter experiment results.
- Linear, polynomial, and RBF kernel decision-boundary plots.
- A final comparison table for all five models using the four original features.

## Observations and Discussion

The results can be used to discuss the following:

- **Interpretability:** Decision Trees provide explicit feature-threshold rules, while SVMs are generally less interpretable.
- **Overfitting:** Deeper trees and overly flexible nonlinear boundaries can overfit the training data.
- **Feature scaling:** SVMs are sensitive to feature scales, whereas Decision Trees generally do not require scaling.
- **Regularization:** The SVM parameter `C` controls the trade-off between margin size and training classification errors.
- **Kernel selection:** Polynomial and RBF kernels can capture nonlinear patterns that a linear SVM cannot.
- **Computational efficiency:** Manual implementations demonstrate the underlying algorithms, while Scikit-learn implementations are optimized. Actual execution times depend on the computer and runtime environment.

Use the tables and plots generated by your notebook to report the actual results rather than assuming fixed accuracy values.
