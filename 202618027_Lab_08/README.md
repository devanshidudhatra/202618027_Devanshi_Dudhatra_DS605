# Lab 08 — Image Feature Extraction and Text Classification

## Overview

This lab focuses on converting image and text data into numerical features and using traditional machine-learning algorithms for classification.

The notebook contains two parts:

- **Part A:** Asphalt image feature extraction and crack classification.
- **Part B:** Email spam classification using preprocessed text features.

## Part A: Asphalt Crack Classification

### Objective

Classify asphalt images into two categories:

- `Cracks`
- `NonCracks`

### Dataset

The image dataset is stored in the `448` directory, with separate folders for the two classes.

```text
448/
├── Cracks/
└── NonCracks/
```

The notebook processes **400 images**.

### Image Preprocessing

1. Load images using OpenCV.
2. Inspect the original image dimensions and channels.
3. Resize images to `224 × 224` pixels.
4. Convert the resized images to grayscale.

### Feature Extraction

The notebook extracts numerical features from each image.

**Original Canny features**
- `edge_count`: Number of detected edge pixels.
- `edge_density`: Proportion of pixels identified as edges.

Canny edge detection helps identify boundaries and structural details in asphalt images.

**Improved Canny + Hough features**

Two additional features are extracted using the Hough Line Transform:

- `hough_line_count`: Number of detected lines.
- `average_line_length`: Average length of the detected lines.

These features help represent the line-like structures that may appear in cracks.

### Classification Models

The initial comparison uses four classifiers:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

The final experiment compares two Random Forest models:

1. Original Canny features — 2 features.
2. Improved Canny + Hough features — 4 features.

### Final Feature Comparison Results

| Metric | Original Canny | Canny + Hough |
|---|---:|---:|
| Number of features | 2 | 4 |
| Accuracy | 67.50% | 81.25% |
| Precision | 68.42% | 76.60% |
| Recall | 65.00% | 90.00% |
| F1-score | 66.67% | 82.76% |
| Training time (seconds) | 0.20972 | 0.14377 |

The improved approach increases accuracy by **13.75 percentage points** and recall by 25 percentage points in the recorded experiment.

## Part B: Email Spam Classification

### Objective

Classify emails as spam or non-spam using numerical text features.

### Dataset

**File:** `emails.csv`

The dataset contains:

- 5,172 email records.
- 3,002 columns.
- 3,000 numerical word-frequency features.
- `Email No.` — an email identifier.
- `Prediction` — the target class.

The dataset is already represented numerically using word-frequency features rather than raw email text.

The notebook inspects the dataset, checks for missing values, and examines the distribution of the target classes.

### Preprocessing

1. Load the CSV file using pandas.
2. Inspect the first few records and dataset dimensions.
3. Check missing values and class distribution.
4. Remove `Email No.` because it is an identifier, not a useful predictive feature.
5. Separate the 3,000 input features from the `Prediction` target.
6. Split the data into 80% training data and 20% testing data, using `random_state=42` and stratification.

### Classification Models

The notebook compares three models:

- **Naive Bayes:** Uses word-feature distributions to classify emails.
- **Logistic Regression:** Estimates the probability of an email belonging to a class.
- **Linear SVM:** Learns a decision boundary between the classes.

### Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Naive Bayes | 94.20% | 86.81% | 94.33% | 90.42% |
| Logistic Regression | 98.26% | 95.78% | 98.33% | 97.04% |
| Linear SVM | 98.16% | 96.37% | 97.33% | 96.85% |

Logistic Regression achieves the highest accuracy and F1-score in the recorded results. Linear SVM achieves slightly higher precision.

## Evaluation Metrics

The notebook uses the following metrics to compare model performance:

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predicted positive cases that are actually positive.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Shows correct and incorrect predictions for each class.
- **Training time:** Time taken to fit a model.
- **Prediction time:** Time taken to make predictions.

## Requirements

- Python 3
- Jupyter Notebook
- OpenCV (`opencv-python`)
- NumPy
- pandas
- Scikit-learn

Install the dependencies with:

```bash
pip install jupyter opencv-python numpy pandas scikit-learn
```

## How to Run

1. Keep `202618027_Lab08.ipynb` in your working directory.
2. Place the `448` image directory and `emails.csv` dataset where the notebook expects them.
3. Install the required dependencies.
4. Open the notebook:

   ```bash
   jupyter notebook
   ```

5. Run the cells sequentially.

**Note:** The image-processing section uses OpenCV's `imshow`, which requires a compatible graphical environment. Dataset paths must match the folder and file names used in the notebook.

## Conclusion

This lab demonstrates how traditional machine-learning models can classify images and text using numerical features.

For asphalt images, adding Hough-based line features improves the recorded Random Forest accuracy from 67.50% to 81.25%. For email classification, Logistic Regression achieves the highest recorded accuracy at 98.26%, closely followed by Linear SVM at 98.16%.

The results illustrate the importance of feature representation and model selection in classification tasks.
