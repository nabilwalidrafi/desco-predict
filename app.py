import streamlit as st
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel('data.xlsx')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.rename(columns={'datetime': 'ds', 'demand': 'y'}, inplace=True)
    return df

df = load_data()

# Train Prophet model with temp and dew only
@st.cache_data
def train_model(df):
    train_df = df.iloc[:-30]
    model = Prophet(daily_seasonality=True)
    model.add_regressor('temp')
    model.add_regressor('dew')
    model.fit(train_df[['ds', 'y', 'temp', 'dew']])
    return model

model = train_model(df)

# Train-test split for evaluation
train_df = df.iloc[:-30]
test_df = df.iloc[-30:]

# Evaluate model on last 30 days
future = test_df[['ds']].copy()
future['temp'] = test_df['temp'].values
future['dew'] = test_df['dew'].values
forecast = model.predict(future)

mae = mean_absolute_error(test_df['y'], forecast['yhat'])
rmse = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat']))

st.title("Demand Prediction with Prophet (temp + dew)")

st.write(f"Model Performance on Last 30 Days: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

# User input for prediction
st.subheader("Input Features for Prediction")
min_date = df['ds'].min().date()
max_date = df['ds'].max().date()
user_date = st.date_input("Select Date", value=max_date, min_value=min_date)
user_temp = st.number_input("Temperature (Celsius)", format="%.2f")
user_dew = st.number_input("Dew Point (°C Td)", format="%.2f")

input_df = pd.DataFrame({
    'ds': [pd.Timestamp(user_date)],
    'temp': [user_temp],
    'dew': [user_dew]
})

# Predict demand for user input
forecast_input = model.predict(input_df)
predicted_demand = forecast_input['yhat'].values[0]

st.success(f"""
📅 Date: {user_date}  
🌡 Temperature: {user_temp:.2f}  
💧 Dew Point: {user_dew:.2f}  
🔮 Predicted Demand: {predicted_demand:.2f}
""")

# Plot actual vs predicted for last 30 days
st.subheader("Real vs Predicted Demand (Last 30 Days)")
plot_df = test_df[['ds', 'y']].copy()
plot_df['Predicted'] = forecast['yhat'].values
plot_df.rename(columns={'y': 'Actual'}, inplace=True)
st.line_chart(plot_df.set_index('ds'))
