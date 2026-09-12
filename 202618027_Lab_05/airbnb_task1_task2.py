"""
Airbnb Price Prediction - Task 1 + Task 2
-----------------------------------------
This script:
1. Loads the Airbnb dataset.
2. Cleans the data.
3. Creates engineered features.
4. Handles missing values through a preprocessing pipeline.
5. Compares Ridge, Random Forest and Extra Trees regression.
6. Tunes Random Forest using cross-validation.
7. Checks overfitting.
8. Saves the final model for Task 3 (Streamlit/Gradio).

IMPORTANT:
- Change DATA_PATH below to the location of your CSV file.
- If you already have a cleaned dataframe called `df` in a notebook,
  the equivalent Task 2 starts from the "FEATURE ENGINEERING" section.
"""

# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

warnings.filterwarnings("ignore")

# ============================================================
# 1. LOAD DATA
# ============================================================

# CHANGE THIS PATH IF REQUIRED
DATA_PATH = "AB_NYC_2019.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Change DATA_PATH to the location of your Airbnb CSV file."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("ORIGINAL DATASET")
print("=" * 70)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC CLEANING
# ============================================================

# Remove exact duplicate rows
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows:", duplicate_count)

df = df.drop_duplicates().copy()


# ------------------------------------------------------------
# Remove irrelevant identifier / free-text columns
# ------------------------------------------------------------

columns_to_drop = [
    "id",
    "host_id",
    "name",
    "host_name",
    "last_review"
]

# Keep last_review temporarily because we use it to create
# has_review, year and month.
columns_to_drop_without_date = [
    "id",
    "host_id",
    "name",
    "host_name"
]

df = df.drop(
    columns=[
        c for c in columns_to_drop_without_date
        if c in df.columns
    ],
    errors="ignore"
)


# ============================================================
# 3. DATE FEATURE ENGINEERING
# ============================================================

if "last_review" in df.columns:

    df["last_review"] = pd.to_datetime(
        df["last_review"],
        errors="coerce"
    )

    # A listing has a review if last_review is available
    df["has_review"] = (
        df["last_review"].notna()
    ).astype(int)

    # These are optional exploratory features.
    # They are NOT used in the final feature set because
    # they showed very weak relationship with price.
    df["last_review_year"] = (
        df["last_review"].dt.year
    )

    df["last_review_month"] = (
        df["last_review"].dt.month
    )

else:

    df["has_review"] = 0


# ============================================================
# 4. TARGET CLEANING
# ============================================================

# Ensure price is numeric
df["price"] = pd.to_numeric(
    df["price"],
    errors="coerce"
)

# Remove missing and zero/negative prices
before_price_cleaning = len(df)

df = df[
    df["price"].notna() &
    (df["price"] > 0)
].copy()

print("\nRows removed because of invalid price:",
      before_price_cleaning - len(df))


# ------------------------------------------------------------
# IMPORTANT:
# Calculate the percentile ONCE.
# Do not rerun this filtering cell on an already filtered df.
# ------------------------------------------------------------

price_upper_limit = df["price"].quantile(0.99)

print("99th percentile price:", price_upper_limit)

before_outlier = len(df)

df = df[
    df["price"] <= price_upper_limit
].copy()

print(
    "Rows removed as extreme price outliers:",
    before_outlier - len(df)
)

print("Maximum price after treatment:",
      df["price"].max())


# ============================================================
# 5. NUMERICAL CLEANING
# ============================================================

numeric_raw_columns = [
    "latitude",
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365"
]

for col in numeric_raw_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# Remove impossible negative values.
# Missing values are NOT manually filled here;
# the modeling pipeline handles them using median imputation.

for col in [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count"
]:

    if col in df.columns:

        df.loc[
            df[col] < 0,
            col
        ] = np.nan


# Availability must be between 0 and 365
if "availability_365" in df.columns:

    df["availability_365"] = df[
        "availability_365"
    ].clip(
        lower=0,
        upper=365
    )


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

# ------------------------------------------------------------
# Log transformations
# ------------------------------------------------------------
# These variables are strongly right-skewed.
# log1p(x) = log(1+x), so zero values are safe.

log_source_columns = [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count"
]

for col in log_source_columns:

    if col not in df.columns:
        raise ValueError(
            f"Required column missing: {col}"
        )

    # Temporary fill only for creating the log feature.
    # The final raw feature is not used in the model.
    values = df[col].fillna(0).clip(lower=0)

    df[
        "log_" + col
    ] = np.log1p(values)


# ------------------------------------------------------------
# Availability ratio
# ------------------------------------------------------------

if "availability_365" in df.columns:

    df["availability_ratio"] = (
        df["availability_365"] / 365.0
    )


# ------------------------------------------------------------
# Distance from approximate NYC centre
# ------------------------------------------------------------

if (
    "latitude" in df.columns
    and
    "longitude" in df.columns
):

    reference_lat = 40.7128
    reference_lon = -74.0060

    df["distance_from_center"] = np.sqrt(
        (df["latitude"] - reference_lat) ** 2
        +
        (df["longitude"] - reference_lon) ** 2
    )


# ------------------------------------------------------------
# Neighbourhood × room type interaction
# ------------------------------------------------------------

if (
    "neighbourhood_group" in df.columns
    and
    "room_type" in df.columns
):

    df["neighbourhood_room"] = (
        df["neighbourhood_group"].astype(str)
        + "_"
        + df["room_type"].astype(str)
    )


# ============================================================
# 7. FINAL FEATURE SELECTION
# ============================================================

features = [

    # Location
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "distance_from_center",

    # Room information
    "room_type",
    "neighbourhood_room",

    # Skewed numerical variables after log transformation
    "log_minimum_nights",
    "log_number_of_reviews",
    "log_reviews_per_month",
    "log_calculated_host_listings_count",

    # Availability
    "availability_ratio",

    # Review information
    "has_review"
]

target = "price"


missing_features = [
    col
    for col in features
    if col not in df.columns
]

if missing_features:

    raise ValueError(
        "The following selected features are missing:\n"
        + str(missing_features)
    )

X = df[features].copy()
y = df[target].copy()

print("\n" + "=" * 70)
print("FINAL MODELING DATA")
print("=" * 70)

print("Number of observations:", len(X))
print("Number of features:", len(features))

print("\nSelected features:")
for i, feature in enumerate(features, start=1):
    print(f"{i}. {feature}")


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 9. DEFINE FEATURE TYPES
# ============================================================

categorical_features = [
    "neighbourhood_group",
    "neighbourhood",
    "room_type",
    "neighbourhood_room"
]

numerical_features = [
    "latitude",
    "longitude",
    "distance_from_center",
    "log_minimum_nights",
    "log_number_of_reviews",
    "log_reviews_per_month",
    "log_calculated_host_listings_count",
    "availability_ratio",
    "has_review"
]


# ============================================================
# 10. PREPROCESSING PIPELINE
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 11. TARGET TRANSFORMATION
# ============================================================

# Airbnb prices are strongly right-skewed.
# log1p reduces the effect of very large prices.
# Predictions are automatically converted back using expm1.

target_transformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1,
    validate=False
)


# ============================================================
# 12. CREATE BASELINE MODELS
# ============================================================

models = {

    "Ridge Regression": Ridge(
        alpha=10.0
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
}


model_pipelines = {}

for name, model in models.items():

    base_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    model_pipelines[name] = (
        TransformedTargetRegressor(
            regressor=base_pipeline,
            transformer=target_transformer
        )
    )


# ============================================================
# 13. TRAIN AND COMPARE BASELINE MODELS
# ============================================================

results = []

for name, model in model_pipelines.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )

    train_pred = model.predict(
        X_train
    )

    test_pred = model.predict(
        X_test
    )

    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )

    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_pred
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_pred
        )
    )

    train_r2 = r2_score(
        y_train,
        train_pred
    )

    test_r2 = r2_score(
        y_test,
        test_pred
    )

    results.append({
        "Model": name,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train R2": train_r2,
        "Test R2": test_r2
    })


results_df = pd.DataFrame(
    results
).sort_values(
    by="Test RMSE"
).reset_index(drop=True)


print("\n" + "=" * 70)
print("BASELINE MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False))


# ============================================================
# 14. CROSS-VALIDATION
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

for name, model in model_pipelines.items():

    print("\nCross-validating:", name)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "MAE": "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2": "r2"
        },
        n_jobs=-1
    )

    cv_results.append({
        "Model": name,
        "CV MAE": -scores["test_MAE"].mean(),
        "CV RMSE": -scores["test_RMSE"].mean(),
        "CV R2": scores["test_R2"].mean()
    })


cv_results_df = pd.DataFrame(
    cv_results
).sort_values(
    by="CV RMSE"
).reset_index(drop=True)


print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)

print(cv_results_df.to_string(index=False))


# ============================================================
# 15. HYPERPARAMETER TUNING - RANDOM FOREST
# ============================================================

rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


rf_model = TransformedTargetRegressor(
    regressor=rf_pipeline,
    transformer=target_transformer
)


param_distributions = {

    "regressor__model__n_estimators": [
        150,
        200,
        300
    ],

    "regressor__model__max_depth": [
        10,
        15,
        20,
        None
    ],

    "regressor__model__min_samples_split": [
        2,
        5,
        10
    ],

    "regressor__model__min_samples_leaf": [
        1,
        2,
        4
    ],

    "regressor__model__max_features": [
        "sqrt",
        "log2",
        0.5
    ]
}


rf_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_distributions,
    n_iter=15,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)


print("\n" + "=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

rf_search.fit(
    X_train,
    y_train
)


print("\nBest parameters:")

for parameter, value in rf_search.best_params_.items():
    print(
        f"{parameter}: {value}"
    )


print(
    "\nBest CV RMSE:",
    -rf_search.best_score_
)


# ============================================================
# 16. EVALUATE TUNED RANDOM FOREST
# ============================================================

final_model = rf_search.best_estimator_

final_train_pred = final_model.predict(
    X_train
)

final_test_pred = final_model.predict(
    X_test
)


final_train_mae = mean_absolute_error(
    y_train,
    final_train_pred
)

final_test_mae = mean_absolute_error(
    y_test,
    final_test_pred
)


final_train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        final_train_pred
    )
)

final_test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_test_pred
    )
)


final_train_r2 = r2_score(
    y_train,
    final_train_pred
)

final_test_r2 = r2_score(
    y_test,
    final_test_pred
)


print("\n" + "=" * 70)
print("FINAL TUNED RANDOM FOREST")
print("=" * 70)

print(
    f"Train MAE  : ${final_train_mae:.2f}"
)

print(
    f"Test MAE   : ${final_test_mae:.2f}"
)

print(
    f"Train RMSE : ${final_train_rmse:.2f}"
)

print(
    f"Test RMSE  : ${final_test_rmse:.2f}"
)

print(
    f"Train R2   : {final_train_r2:.4f}"
)

print(
    f"Test R2    : {final_test_r2:.4f}"
)


# ============================================================
# 17. OVERFITTING CHECK
# ============================================================

r2_gap = (
    final_train_r2
    -
    final_test_r2
)


print("\n" + "=" * 70)
print("OVERFITTING CHECK")
print("=" * 70)

print(
    f"Train R2 : {final_train_r2:.4f}"
)

print(
    f"Test R2  : {final_test_r2:.4f}"
)

print(
    f"R2 gap   : {r2_gap:.4f}"
)


if r2_gap > 0.20:

    print(
        "Result: Moderate/significant overfitting may be present."
    )

elif r2_gap > 0.10:

    print(
        "Result: Some overfitting may be present."
    )

else:

    print(
        "Result: No major overfitting detected."
    )


# ============================================================
# 18. FINAL MODEL COMPARISON TABLE
# ============================================================

final_comparison = results_df.copy()

tuned_row = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "Train MAE": final_train_mae,
    "Test MAE": final_test_mae,
    "Train RMSE": final_train_rmse,
    "Test RMSE": final_test_rmse,
    "Train R2": final_train_r2,
    "Test R2": final_test_r2
}])


final_comparison = pd.concat(
    [
        final_comparison,
        tuned_row
    ],
    ignore_index=True
)


final_comparison = final_comparison.sort_values(
    by="Test RMSE"
).reset_index(drop=True)


print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    final_comparison.to_string(index=False)
)


best_model_name = (
    final_comparison.iloc[0]["Model"]
)

print(
    "\nBest model based on test RMSE:",
    best_model_name
)


# ============================================================
# 19. ACTUAL VS PREDICTED
# ============================================================

plt.figure(
    figsize=(8, 8)
)

plt.scatter(
    y_test,
    final_test_pred,
    alpha=0.3
)

minimum_value = min(
    y_test.min(),
    final_test_pred.min()
)

maximum_value = max(
    y_test.max(),
    final_test_pred.max()
)

plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--"
)

plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title("Actual vs Predicted Airbnb Prices")
plt.tight_layout()
plt.show()


# ============================================================
# 20. RESIDUAL ANALYSIS
# ============================================================

residuals = (
    y_test -
    final_test_pred
)

plt.figure(
    figsize=(10, 5)
)

plt.scatter(
    final_test_pred,
    residuals,
    alpha=0.3
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Price ($)")
plt.ylabel("Residual ($)")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()


# ============================================================
# 21. SAVE FINAL MODEL
# ============================================================

MODEL_PATH = "airbnb_price_prediction_final.pkl"

joblib.dump(
    final_model,
    MODEL_PATH
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"Saved to: {MODEL_PATH}"
)


# ============================================================
# 22. SAVE MODEL RESULTS
# ============================================================

final_comparison.to_csv(
    "airbnb_model_comparison.csv",
    index=False
)


pd.DataFrame({
    "Metric": [
        "Test MAE",
        "Test RMSE",
        "Test R2",
        "Train R2",
        "R2 Gap"
    ],
    "Value": [
        final_test_mae,
        final_test_rmse,
        final_test_r2,
        final_train_r2,
        r2_gap
    ]
}).to_csv(
    "airbnb_final_metrics.csv",
    index=False
)


pd.DataFrame({
    "Actual_Price": y_test.values,
    "Predicted_Price": final_test_pred
}).assign(
    Absolute_Error=lambda x:
        abs(
            x["Actual_Price"]
            -
            x["Predicted_Price"]
        )
).to_csv(
    "airbnb_test_predictions.csv",
    index=False
)


# ============================================================
# 23. TEST SAVED MODEL
# ============================================================

loaded_model = joblib.load(
    MODEL_PATH
)

saved_model_predictions = loaded_model.predict(
    X_test.iloc[:5]
)

print("\nPredictions from the saved model:")

for prediction in saved_model_predictions:
    print(
        f"${prediction:.2f}"
    )


print("\n" + "=" * 70)
print("TASK 1 + TASK 2 COMPLETED")
print("=" * 70)
print("Final model:", best_model_name)
print(f"Test MAE : ${final_test_mae:.2f}")
print(f"Test RMSE: ${final_test_rmse:.2f}")
print(f"Test R2  : {final_test_r2:.4f}")
print("Model file:", MODEL_PATH)
