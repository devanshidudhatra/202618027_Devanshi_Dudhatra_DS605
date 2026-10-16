# DS605 Lab 7 — Image Representation, PCA, Eigenfaces and Face Recognition

## Dataset
This notebook uses the AT&T Database of Faces (ORL). Download the dataset from the official AT&T/University of Cambridge archive page and extract it.

Arrange the folders like this:

```text
project/
├── DS605_Lab7_PCA_Face_Recognition.ipynb
├── README.md
├── data/
│   └── orl_faces/
│       ├── s1/
│       │   ├── 1.pgm
│       │   ├── 2.pgm
│       │   └── ...
│       ├── s2/
│       └── ...
└── outputs/                 # created automatically by the notebook
```

The notebook expects the subject folders (`s1`, `s2`, etc.) directly inside `data/orl_faces/`. If your folder is elsewhere, change `DATA_DIR` in the notebook.

## Install requirements

```bash
pip install numpy matplotlib pillow scikit-learn notebook
```

Open and run the notebook:

```bash
jupyter notebook DS605_Lab7_PCA_Face_Recognition.ipynb
```

Run all cells from top to bottom.

## What the notebook does

### Part A — Image representation
- Reads each image as a grayscale matrix.
- Prints image dimensions and pixel intensity range.
- Flattens each image into a feature vector and creates the image matrix with subject labels.
- Creates a reproducible train-test split with images from each subject in both sets.

### Part B — PCA, eigenfaces and compression
- Calculates the mean face and centers the training data.
- Uses NumPy eigendecomposition on the sample-space Gram matrix and maps the directions back to pixel space.
- Uses NumPy SVD and compares the leading principal directions.
- Plots cumulative explained variance and selects the minimum number of components needed for at least 90% variance.
- Visualizes eigenfaces, reconstructs sample faces, and compares the number of values needed by original and PCA representations.

**Important:** PCA is implemented with NumPy linear algebra. `sklearn.decomposition.PCA` is not used.

### Part C — Recognition in PCA space
- Matches each test image to its nearest training image using Euclidean distance.
- Compares nearest-neighbor recognition on raw pixels and PCA features.
- Displays examples of test faces and their closest PCA-space matches.

### Part D — Traditional machine learning
- Trains K-Nearest Neighbors (KNN) and a linear Support Vector Machine (SVM) using PCA features.
- Compares their accuracies with the similarity-based nearest-neighbor baseline.
- Tries multiple component counts and prints a compact comparison table.

## Output files
When the notebook runs successfully, it saves these figures in `outputs/`:
- `explained_variance.png`
- `eigenfaces.png`
- `reconstruction_examples.png`
- `recognition_matches.png`

The notebook also prints recognition accuracies, component counts, variance retained, reconstruction error, and approximate storage comparisons. The numerical results are generated from your dataset when you run it.

## Notes
- The train-test split uses a fixed random seed for reproducibility.
- Results may vary if the dataset or split changes.
- This notebook is intended to be a simple, readable lab solution rather than a highly optimized implementation.
- For submission, place the notebook, this README, and generated figures in a **public GitHub repository**, then submit the repository link as required by the assignment.
