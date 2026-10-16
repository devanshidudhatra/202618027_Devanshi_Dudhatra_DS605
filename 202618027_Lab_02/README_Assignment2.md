# DS605 Assignment 2 — NumPy and Pandas Data Wrangling

## Overview

This notebook practices vectorized programming with NumPy and data wrangling with Pandas using the Titanic dataset.

## Contents

### Part A — Vectorized Programming with NumPy

- **Task 1: Arrays, Statistics, and Indexing**
  - Creates a reproducible array of random integers.
  - Calculates minimum, maximum, median, mean, and standard deviation.
  - Uses `np.arange()`, `np.zeros()`, `np.ones()`, and `np.linspace()`.
  - Creates and inspects 2D and 3D arrays.
  - Demonstrates indexing, slicing, reshaping, and flattening.
- **Task 2: Vectorized Arithmetic and Linear Algebra**
  - Performs matrix addition, element-wise multiplication, and matrix multiplication.
  - Calculates a matrix transpose and determinant.
  - Finds the inverse when possible and checks the result with `np.allclose()`.
- **Task 3: Normal Distribution and Histogram**
  - Generates 1,000 values from a normal distribution.
  - Compares the sample mean and sample standard deviation with the chosen values.
  - Displays a histogram.

### Part B — Data Wrangling with Pandas: Titanic Dataset

- **Task 4: Load and Inspect Data**
  - Uses `head()`, `tail()`, `shape`, `columns`, `info()`, and `describe()`.
  - Demonstrates row and column selection with `loc` and `iloc`.
- **Task 5: Filtering and Querying**
  - Filters passengers using conditions involving sex, age, class, fare, survival, family members, and embarkation port.
- **Task 6: Grouping and Aggregation**
  - Calculates survival rates by sex and passenger class.
  - Calculates average age and fare, passenger counts, and grouped survival rates.
- **Task 7: Missing Values and Fare Outliers**
  - Counts and plots missing values.
  - Demonstrates mean, median, mode, and random-value imputation for missing ages.
  - Calculates the IQR bounds and counts fare outliers.
- **Task 8: Features and Pivot Table**
  - Creates `FamilySize` and `IsAlone`.
  - Creates a pivot table of survival rates by sex and passenger class.
- **Task 9: Visualizations and Observations**
  - Displays a correlation heatmap.
  - Plots survival rate by sex and age against fare.
  - Lists observations based on the analysis.

## Requirements

Install the libraries used by the notebook:

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

## Dataset setup

The notebook uses the Titanic `train.csv` dataset. Make sure this file is available before running the Pandas section.

For easier use on another computer, place `train.csv` in the same folder as the notebook and replace the existing absolute Windows paths in the notebook with:

```python
train = pd.read_csv("train.csv")
```

Apply the same path change to each cell that loads `train.csv`.

Example folder structure:

```text
project/
├── 202618027_Assignment2.ipynb
├── README.md
└── train.csv
```

## How to run

1. Install the required libraries.
2. Place `train.csv` in the notebook folder, or update the dataset path in the notebook.
3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `202618027_Assignment2.ipynb`.
5. Run the cells from top to bottom.

## Notes

- NumPy random seeds are set in the notebook to make random outputs reproducible.
- Some Pandas results depend on the contents of `train.csv`.
- The notebook includes code and visualizations for the listed tasks. Run it locally to generate the outputs and verify the observations against the dataset.
