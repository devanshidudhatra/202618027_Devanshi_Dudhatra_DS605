# Airbnb Price Prediction - Task 3

## Files required

Keep these files in the same folder:

- `app.py`
- `airbnb_price_prediction_final.pkl`
- `AB_NYC_2019.csv`

The `.pkl` file is created by the Task 1 + Task 2 script.

## Run locally

Open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will open the application in your browser.

## What the application does

The application accepts:

- Neighbourhood group
- Neighbourhood
- Room type
- Latitude
- Longitude
- Minimum nights
- Number of reviews
- Reviews per month
- Host listing count
- Availability in the next 365 days
- Whether the listing has a previous review

It recreates the same engineered features used during training:

- `distance_from_center`
- `neighbourhood_room`
- `log_minimum_nights`
- `log_number_of_reviews`
- `log_reviews_per_month`
- `log_calculated_host_listings_count`
- `availability_ratio`
- `has_review`

The saved pipeline then performs the same preprocessing and returns the estimated nightly price.

## Important

Do not recreate the scaler, one-hot encoder, or Random Forest manually in the app. They are already stored inside the saved `.pkl` pipeline. This prevents training-time and application-time preprocessing from becoming inconsistent.

## Example realistic test

Try:

- Neighbourhood group: Manhattan
- Neighbourhood: Midtown
- Room type: Entire home/apt
- Latitude: 40.7549
- Longitude: -73.9840
- Minimum nights: 2
- Number of reviews: 20
- Reviews per month: 1.5
- Host listings count: 2
- Availability: 180
- Has previous review: Yes

Then click **Estimate Nightly Price**.
