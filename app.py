import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Load data once on app start
@st.cache_data
def load_data():
    df = pd.read_excel('data.xlsx')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.rename(columns={'datetime': 'ds', 'demand': 'y'}, inplace=True)
    return df

df = load_data()

# Define feature sets again
feature_sets = {
    'temp_dew': ['temp', 'dew'],
    'temp_dew_precip': ['temp', 'dew', 'precipprob'],
    'temp_dew_precip_wind_cloud': ['temp', 'dew', 'precipprob', 'windspeedmean', 'cloudcover']
}

# Train Prophet models for each feature set once and cache
@st.cache_data
def train_models(df, feature_sets):
    train_df = df.iloc[:-30]
    models = {}
    for name, features in feature_sets.items():
        model = Prophet(daily_seasonality=True)
        for f in features:
            model.add_regressor(f)
        model.fit(train_df[['ds', 'y'] + features])
        models[name] = model
    return models

models = train_models(df, feature_sets)

# Train-test split for evaluation
train_df = df.iloc[:-30]
test_df = df.iloc[-30:]

st.title("Demand Prediction with Prophet")

# Select date for prediction (limit to last 30 days + future dates)
min_date = df['ds'].min()
max_date = df['ds'].max()
selected_date = st.date_input("Select date for prediction", value=max_date, min_value=min_date, max_value=max_date)

# Convert selected_date to Timestamp for indexing
selected_date_ts = pd.Timestamp(selected_date)

# Check if date exists in data
if selected_date_ts not in df['ds'].values:
    st.warning("Selected date not in dataset. Prediction unavailable.")
else:
    # Choose best model based on lowest RMSE from evaluation on test set
    # Evaluate all models
    metrics = {}
    for name, features in feature_sets.items():
        model = models[name]
        future = test_df[['ds']].copy()
        for f in features:
            future[f] = test_df[f].values
        forecast = model.predict(future)
        y_true = test_df['y'].values
        y_pred = forecast['yhat'].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[name] = {'MAE': mae, 'RMSE': rmse}
    # Pick best model by RMSE
    best_model_name = min(metrics, key=lambda k: metrics[k]['RMSE'])
    best_model = models[best_model_name]
    best_features = feature_sets[best_model_name]

    st.write(f"Best Model Selected: **{best_model_name}**")
    st.write(f"MAE: {metrics[best_model_name]['MAE']:.2f}")
    st.write(f"RMSE: {metrics[best_model_name]['RMSE']:.2f}")

    # Prepare single-row dataframe for prediction
    input_row = df[df['ds'] == selected_date_ts][['ds'] + best_features]

    # Predict demand for selected date
    forecast = best_model.predict(input_row)
    predicted_demand = forecast['yhat'].values[0]

    # Get actual demand for selected date
    actual_demand = df[df['ds'] == selected_date_ts]['y'].values[0]

    # Display prediction and actual value
    st.success(f"""
    📅 **Date**: {selected_date}  
    🔮 **Predicted Demand**: {predicted_demand:.2f}  
    📊 **Actual Demand**: {actual_demand:.2f}
    """)

