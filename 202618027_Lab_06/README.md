# ML Challenge 2 — Bayesian Optimization for Hyperparameter Tuning

## Overview

This project implements **Bayesian Optimization from scratch** to find effective hyperparameters for an ElasticNet regression model.

The notebook communicates with an external Oracle API to evaluate candidate hyperparameter configurations and obtain their loss values. Instead of testing every possible configuration, Bayesian Optimization uses previous results to choose promising configurations for the next evaluation.

The optimization algorithm is implemented using pure Python, without NumPy, pandas, or external mathematical libraries in the optimizer implementation.

## Objectives

- Retrieve the model's allowed hyperparameter search space from an API.
- Implement Bayesian Optimization using pure Python.
- Build a Gaussian Process surrogate model using a Matérn 5/2 kernel.
- Use Expected Improvement as the acquisition function.
- Select initial configurations using deterministic max-min-distance sampling.
- Query the Oracle API to evaluate candidate configurations.
- Record results and avoid repeating configurations already evaluated.
- Identify the hyperparameter configuration with the lowest observed loss.

## Technologies Used

- Python
- JSON for API communication and saving results
- `urllib.request` for sending HTTP requests
- `os` for checking and managing the query log
- External Oracle API for evaluating hyperparameter configurations

## Hyperparameters

The Oracle API provides the permitted search space for the ElasticNet model.

| Hyperparameter | Purpose |
|---|---|
| `alpha` | Controls the strength of regularization. |
| `l1_ratio` | Controls the balance between L1 and L2 regularization. |
| `tol` | Sets the tolerance used to determine convergence. |
| `positive` | Determines whether model coefficients must be non-negative. |
| `selection` | Selects the coefficient update strategy. |
| `fit_intercept` | Determines whether an intercept is fitted. |

The notebook reports **2,560 possible configurations** in the supplied search space.

## Methodology

### 1. API Communication

The notebook sends requests to the Oracle API to retrieve the model name and allowed hyperparameter values. It then sends candidate configurations to the API and receives their loss values.

### 2. Hyperparameter Encoding

The categorical and numerical parameter values are converted into numerical representations so that the optimizer can compare configurations.

- Numeric values are represented by their ranks in the search space.
- Binary choices are represented numerically.
- Parameters with multiple categorical choices use one-hot encoding.

### 3. Initial Sampling

The optimizer begins with a deterministic space-filling design. It selects an initial configuration and then chooses additional configurations that are far from those already evaluated.

This helps the optimizer explore different parts of the search space before making more informed selections.

### 4. Gaussian Process Surrogate Model

A Gaussian Process is implemented from scratch to estimate the expected loss of unevaluated configurations.

The implementation includes:

- Matérn 5/2 kernel
- Cholesky decomposition
- Forward and backward substitution
- Marginal likelihood calculation
- Length-scale and noise selection

The surrogate model uses previous Oracle results to estimate both the expected loss and uncertainty for untested configurations.

### 5. Expected Improvement

Expected Improvement (EI) is used as the acquisition function. It balances two goals:

- **Exploitation:** testing configurations predicted to have low loss.
- **Exploration:** testing configurations where uncertainty may reveal a better result.

The optimizer selects a promising unevaluated configuration and makes an Oracle call to obtain its actual loss.

### 6. Logging and Stopping Conditions

The notebook saves evaluated configurations and their losses in `oracle_log.json`. The log allows previous results to be reloaded so that already completed evaluations do not need to be repeated.

The optimization may stop when it reaches its evaluation budget, when the expected improvement becomes very small, or when the configured stopping conditions indicate that further progress is unlikely.

### 7. Final Results

The notebook reports the best hyperparameters, the lowest observed loss, the number of Oracle calls, and the best-so-far loss after each evaluation.

The recorded optimization run found a best loss of approximately **8.89035**. Actual results may differ depending on the API response and the evaluations performed.

## Project Structure

```text
project/
├── 202618027_DevanshiDudhatra_MLChallenge2.ipynb
└── oracle_log.json
```

The notebook creates `oracle_log.json` when the optimization runs with logging enabled. The notebook itself contains the optimizer implementation.

## How to Run

1. Open the notebook in Jupyter Notebook or JupyterLab.
2. Ensure that the Oracle API is available and that you have valid team credentials.
3. Configure the team ID, API endpoint, and API key securely.
4. Run the first cell to retrieve the model specification and search space.
5. Run the optimizer-definition cell.
6. Run the optimization cell to evaluate configurations through the Oracle API.
7. Run the reporting cell to display the best parameters, loss, and evaluation history.

**Important:** The optimization cell makes real API requests. Review the evaluation budget before running it because API calls may be limited.

## Expected Outcome

The project demonstrates how Bayesian Optimization can reduce the number of expensive model evaluations compared with exhaustive search. It combines a Gaussian Process surrogate model, Expected Improvement, deterministic initial sampling, and persistent result logging to search for effective ElasticNet hyperparameters.

## Security Note

Do not publish API keys in a public notebook or README. If a real API key has been exposed in a shared notebook, revoke or rotate it and use a secure environment variable or another secret-management method instead.
