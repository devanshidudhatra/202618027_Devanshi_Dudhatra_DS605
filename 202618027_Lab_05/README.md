# Airbnb Nightly Price Prediction

## Overview

This project uses machine learning to estimate the nightly price of an Airbnb listing in New York City based on details such as location, room type, reviews, minimum nights, host listings, and availability.

The project contains two main files:

- `airbnb_task1_task2.py` — prepares the data, trains and compares regression models, tunes a Random Forest model, evaluates performance, and saves the final model.
- `app.py` — runs a Streamlit web application that accepts listing details and displays an estimated nightly price.

## Dataset

The project requires the `AB_NYC_2019.csv` dataset.

Place the dataset in the project directory. The training script expects the filename `AB_NYC_2019.csv` by default.

The dataset includes Airbnb listing information such as neighbourhood, latitude, longitude, room type, price, minimum nights, review counts, host listing counts, and availability.

## Project Workflow

### Task 1 and Task 2: Data Preparation and Model Training

The `airbnb_task1_task2.py` script performs these steps:

1. Loads and inspects the dataset.
2. Removes duplicate rows and unnecessary identifier and text columns.
3. Creates review-related features from `last_review`.
4. Removes invalid prices and caps extreme prices at the 99th percentile.
5. Cleans numerical columns and handles invalid values.
6. Creates features such as log-transformed counts, availability ratio, distance from the approximate NYC centre, and a neighbourhood-room interaction.
7. Selects the input features and target (`price`).
8. Splits the dataset into training and testing sets using an 80:20 split.
9. Builds preprocessing pipelines:
   - Median imputation and standard scaling for numerical features.
   - Most-frequent imputation and one-hot encoding for categorical features.
10. Applies a log transformation to prices during training and converts predictions back to the original price scale.
11. Compares Ridge Regression, Random Forest, and Extra Trees.
12. Evaluates models using MAE, RMSE, and R².
13. Performs five-fold cross-validation.
14. Tunes Random Forest hyperparameters using `RandomizedSearchCV`.
15. Checks for overfitting and generates evaluation plots.
16. Saves the trained model and evaluation results.

### Task 3: Streamlit Application

The `app.py` script provides a web interface where users can enter:

- Neighbourhood group and neighbourhood
- Room type
- Latitude and longitude
- Minimum nights
- Number of reviews and reviews per month
- Host listing count
- Availability during the year
- Whether the listing has a previous review

After the user clicks **Estimate Nightly Price**, the application processes the inputs and displays the estimated nightly price using the saved model.

The prediction is an estimate based on historical data, not a guaranteed market price.

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib scikit-learn joblib streamlit
```

## Project Structure

```text
project/
├── airbnb_task1_task2.py
├── app.py
└── AB_NYC_2019.csv
```

After running the training script, the following output files are generated:

- `airbnb_price_prediction_final.pkl` — saved trained model.
- `airbnb_model_comparison.csv` — comparison of baseline and tuned models.
- `airbnb_final_metrics.csv` — final evaluation metrics.
- `airbnb_test_predictions.csv` — actual prices, predicted prices, and absolute errors for test data.

## How to Run

### Step 1: Prepare the Dataset

Place `AB_NYC_2019.csv` in the same directory as the Python scripts. If the dataset is stored elsewhere, update `DATA_PATH` in `airbnb_task1_task2.py`.

### Step 2: Train the Model

Open a terminal in the project directory and run:

```bash
python airbnb_task1_task2.py
```

Wait for the script to finish and create the trained model and evaluation files.

### Step 3: Start the Web Application

Run:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal. Enter the listing details and click **Estimate Nightly Price** to obtain a prediction.

## Evaluation Metrics

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted prices. Lower values are better.
- **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily. Lower values are better.
- **R² score:** Measures how much variation in prices is explained by the model. Higher values are generally better.

The script compares models using test RMSE and reports the final model's evaluation metrics. Results may change when the model is retrained.

## Notes

- Run `airbnb_task1_task2.py` before starting `app.py`, because the application requires the saved model file.
- The application also reads `AB_NYC_2019.csv` to populate its neighbourhood and room-type dropdowns.
- Keep the model and dataset at the paths expected by the scripts, or update the paths in the code.
- The training script and application must use matching feature-engineering logic.
- The model uses historical listing data and does not account for every factor that may affect current prices.
- Check the dataset's license and usage terms before redistributing it.
