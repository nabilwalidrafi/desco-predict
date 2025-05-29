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
    df['cap'] = 900  # Saturation cap for logistic growth
    return df

df = load_data()

# Train Prophet model with logistic growth and raw regressors
@st.cache_data
def train_model(df):
    train_df = df.iloc[:-30].copy()
    
    model = Prophet(growth='logistic', daily_seasonality=True)
    model.add_regressor('temp')
    model.add_regressor('dew')
    model.fit(train_df[['ds', 'y', 'cap', 'temp', 'dew']])
    
    return model

model = train_model(df)

# Prepare test data
test_df = df.iloc[-30:].copy()

future = test_df[['ds']].copy()
future['temp'] = test_df['temp'].values
future['dew'] = test_df['dew'].values
future['cap'] = 840

# Forecast
forecast = model.predict(future)

# Evaluate
mae = mean_absolute_error(test_df['y'], forecast['yhat'])
rmse = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat']))

st.title("📈 Demand Prediction with Prophet (Logistic Growth + temp/dew)")
st.write(f"### 📊 Model Performance (Last 30 Days)\n- MAE = {mae:.2f}\n- RMSE = {rmse:.2f}")

# User input
st.subheader("🔍 Predict Demand for a Custom Day")
min_date = df['ds'].min().date()
max_date = df['ds'].max().date()
user_date = st.date_input("Select Date", value=max_date, min_value=min_date)
user_temp = st.number_input("Temperature (°C)", format="%.2f")
user_dew = st.number_input("Dew Point (°C Td)", format="%.2f")

# Create input for prediction (no scaling needed)
input_df = pd.DataFrame({
    'ds': [pd.Timestamp(user_date)],
    'temp': [user_temp],
    'dew': [user_dew],
    'cap': [900]
})

# Predict
forecast_input = model.predict(input_df)
predicted_demand = forecast_input['yhat'].values[0]

st.success(f"""
📅 Date: {user_date}  
🌡 Temperature: {user_temp:.2f}  
💧 Dew Point: {user_dew:.2f}  
🔮 **Predicted Demand:** {predicted_demand:.2f}
""")

# Plot: Actual vs Predicted
st.subheader("📉 Actual vs Predicted Demand (Last 30 Days)")
plot_df = test_df[['ds', 'y']].copy()
plot_df['Predicted'] = forecast['yhat'].values
plot_df.rename(columns={'y': 'Actual'}, inplace=True)
st.line_chart(plot_df.set_index('ds'))

# Plot: Residuals
st.subheader("📉 Residuals (Prediction Error)")

# Align indices before subtraction
residuals = forecast['yhat'].reset_index(drop=True) - test_df['y'].reset_index(drop=True)

# Build residual DataFrame
residual_df = pd.DataFrame({
    'ds': test_df['ds'].reset_index(drop=True),
    'Residuals': residuals
})

st.line_chart(residual_df.set_index('ds'))

