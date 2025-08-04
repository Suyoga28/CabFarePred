import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import radians, cos, sin, asin, sqrt

# -----------------------------------------
st.set_page_config(page_title="🚕 Cab Fare Predictor", layout="wide")
st.title("🚖 Cab Fare Prediction using ML")
st.markdown("Upload your dataset, train the model, and predict cab fares in real-time.")

# -----------------------------------------
# Haversine distance function
def haversine(row):
    try:
        lon1, lat1, lon2, lat2 = map(radians, [row['pickup_longitude'], row['pickup_latitude'],
                                               row['dropoff_longitude'], row['dropoff_latitude']])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        return km
    except:
        return np.nan

# -----------------------------------------
# File Uploads
train_file = st.file_uploader("Upload Training CSV (must include `fare_amount`)", type=["csv"])
test_file = st.file_uploader("Upload Test CSV (optional)", type=["csv"])

# -----------------------------------------
# Main App Logic
if train_file is not None:
    train = pd.read_csv(train_file)

    # Parse datetime
    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'], errors='coerce')
    train = train.dropna(subset=['pickup_datetime'])

    # Extract time features
    train['year'] = train['pickup_datetime'].dt.year
    train['month'] = train['pickup_datetime'].dt.month
    train['day'] = train['pickup_datetime'].dt.day
    train['hour'] = train['pickup_datetime'].dt.hour

    # Convert fare_amount to numeric
    train["fare_amount"] = pd.to_numeric(train["fare_amount"], errors="coerce")

    # Haversine distance
    train['distance'] = train.apply(haversine, axis=1)

    # Clean and filter
    train = train.dropna(subset=["fare_amount", "distance", "passenger_count"])
    train = train[(train['fare_amount'] > 0) & (train['distance'] > 0)]
    train = train[(train['passenger_count'] > 0) & (train['passenger_count'] <= 6)]

    # Fix dtypes
    train['passenger_count'] = train['passenger_count'].astype(int)

    # Feature set
    features = ['passenger_count', 'year', 'month', 'day', 'hour', 'distance']
    X = train[features]
    y = np.log1p(train['fare_amount'])  # log1p to reduce skew

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.success("✅ Model trained successfully!")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # -----------------------------------------
    # Predictions on test.csv
    if test_file is not None:
        test = pd.read_csv(test_file)
        test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], errors='coerce')
        test = test.dropna(subset=['pickup_datetime'])

        test['year'] = test['pickup_datetime'].dt.year
        test['month'] = test['pickup_datetime'].dt.month
        test['day'] = test['pickup_datetime'].dt.day
        test['hour'] = test['pickup_datetime'].dt.hour

        test['distance'] = test.apply(haversine, axis=1)
        test = test.dropna(subset=["distance", "passenger_count"])
        test = test[(test['passenger_count'] > 0) & (test['passenger_count'] <= 6)]
        test['passenger_count'] = test['passenger_count'].astype(int)

        if set(features).issubset(test.columns):
            test_X = test[features]
            test['Predicted_Fare'] = np.expm1(model.predict(test_X))  # inverse log1p
            st.subheader("🧾 Predictions on Test Data")
            st.dataframe(test[features + ['Predicted_Fare']].head())

            csv = test.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Prediction CSV", csv, "cab_fare_predictions.csv", "text/csv")
        else:
            st.warning("❗ Test file is missing required columns.")
else:
    st.info("📂 Please upload a training CSV file to begin.")
