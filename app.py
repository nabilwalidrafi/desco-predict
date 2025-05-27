import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Convert selected_date to Timestamp
selected_date_ts = pd.Timestamp(selected_date)

# Evaluate all models on last 30 days
metrics = {}
forecasts = {}

for name, features in feature_sets.items():
    model = models[name]
    future = test_df[['ds']].copy()
    for f in features:
        future[f] = test_df[f].values
    forecast = model.predict(future)
    forecasts[name] = forecast
    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics[name] = {'MAE': mae, 'RMSE': rmse}

# Pick best model
best_model_name = min(metrics, key=lambda k: metrics[k]['RMSE'])
best_model = models[best_model_name]
best_features = feature_sets[best_model_name]
best_forecast = forecasts[best_model_name]

st.write(f"Best Model Selected: **{best_model_name}**")
st.write(f"MAE: {metrics[best_model_name]['MAE']:.2f}")
st.write(f"RMSE: {metrics[best_model_name]['RMSE']:.2f}")

# Prediction for selected date
if selected_date_ts not in df['ds'].values:
    st.warning("Selected date not in dataset. Prediction unavailable.")
else:
    input_row = df[df['ds'] == selected_date_ts][['ds'] + best_features]
    forecast_single = best_model.predict(input_row)
    predicted_demand = forecast_single['yhat'].values[0]
    actual_demand = df[df['ds'] == selected_date_ts]['y'].values[0]

    st.success(f"""
    📅 **Date**: {selected_date}  
    🔮 **Predicted Demand**: {predicted_demand:.2f}  
    📊 **Actual Demand**: {actual_demand:.2f}
    """)

# Time series plot for last 30 days
st.subheader("📈 Real vs Predicted Demand (Last 30 Days)")

plot_df = test_df[['ds', 'y']].copy()
plot_df['Predicted'] = best_forecast['yhat'].values
plot_df.rename(columns={'y': 'Actual'}, inplace=True)

st.line_chart(plot_df.set_index('ds'))
