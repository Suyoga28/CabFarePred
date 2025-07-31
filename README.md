# 🚖 Cab Fare Prediction using Machine Learning

This project focuses on predicting New York City cab fares based on historical ride data, using machine learning regression techniques. Implemented in a Jupyter Notebook, it showcases real-world data preprocessing, feature engineering, and regression modeling.

---

## 📌 Project Objective

The goal is to build a predictive model that can estimate the fare amount for NYC taxi rides based on parameters such as:
- Pickup and dropoff locations
- Distance traveled
- Passenger count
- Time of the ride

---

## 🛠️ Tech Stack

- Python
- Jupyter Notebook
- Libraries: `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`, `Datetime`

---

## 🔍 Dataset

The dataset includes:
- `pickup_datetime`
- `pickup_longitude`, `pickup_latitude`
- `dropoff_longitude`, `dropoff_latitude`
- `passenger_count`
- `fare_amount` (target)

> The dataset was cleaned by removing outliers, null values, and inconsistent records.

---

## 🧪 Model Workflow

1. **Data Cleaning**: Handled missing values, removed outliers, and corrected data types.
2. **Feature Engineering**:
   - Calculated Haversine distance between pickup and dropoff points.
   - Extracted features from datetime (hour, day of week, etc.).
3. **Model Used**:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor *(performed best)*
4. **Evaluation**:
   - RMSE (Root Mean Squared Error)
   - R² Score

---

## 📊 Results

- 📈 **Best Model**: Random Forest Regressor  
- ✅ **R² Score**: ~0.83  
- 📉 **RMSE**: Low error indicating strong predictions

---

