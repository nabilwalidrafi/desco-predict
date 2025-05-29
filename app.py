import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

# Function to load and preprocess data
def load_data():
    try:
        df = pd.read_excel('data.xlsx')
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime', 'demand'])
        df.rename(columns={'datetime': 'ds', 'demand': 'y'}, inplace=True)
        # Clip outliers (1st–99th percentiles)
        demand_lower, demand_upper = np.percentile(df['y'], [1, 99])
        df['y'] = df['y'].clip(lower=demand_lower, upper=demand_upper)
        # Set logistic growth parameters
        df['cap'] = np.percentile(df['y'], 97.5)  # ~1450
        df['floor'] = np.percentile(df['y'], 2.5)  # ~550
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_data()
if df.empty:
    st.stop()

# Function to train Prophet model
def train_model(df, growth='linear'):
    train_df = df.iloc[:-30].copy()
    model = Prophet(
        growth=growth,
        changepoints=['2023-06-01', '2024-01-01', '2024-06-01'],
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=15.0
    )
    model.add_regressor('temp', prior_scale=1.0, standardize=True)
    model.add_regressor('dew', prior_scale=1.0, standardize=True)
    if growth == 'logistic':
        model.fit(train_df[['ds', 'y', 'cap', 'floor', 'temp', 'dew']])
    else:
        model.fit(train_df[['ds', 'y', 'temp', 'dew']])
    return model, train_df

# Train models
model_linear, train_df = train_model(df, growth='linear')
model_logistic, _ = train_model(df, growth='logistic')

# Prepare test data and forecast
test_df = df.iloc[-30:].copy().reset_index(drop=True)  # Reset index for alignment
future_linear = test_df[['ds', 'temp', 'dew']].copy()
future_logistic = test_df[['ds', 'temp', 'dew', 'cap', 'floor']].copy()
forecast_linear = model_linear.predict(future_linear).reset_index(drop=True)  # Reset index
forecast_logistic = model_logistic.predict(future_logistic).reset_index(drop=True)

# Calculate performance metrics with safe MAPE
def calculate_metrics(y_true, y_pred):
    y_true = y_true.reset_index(drop=True)  # Ensure index alignment
    y_pred = y_pred.reset_index(drop=True)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Safe MAPE calculation
    mask = y_true > 1e-6
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return mae, rmse, mape

mae_linear, rmse_linear, mape_linear = calculate_metrics(test_df['y'], forecast_linear['yhat'])
mae_logistic, rmse_logistic, mape_logistic = calculate_metrics(test_df['y'], forecast_logistic['yhat'])

# Select better model
best_model = model_linear if mae_linear < mae_logistic else model_logistic
best_forecast = forecast_linear if mae_linear < mae_logistic else forecast_logistic
best_mae, best_rmse, best_mape = (mae_linear, rmse_linear, mape_linear) if mae_linear < mae_logistic else (mae_logistic, rmse_logistic, mape_logistic)
growth_type = 'linear' if mae_linear < mae_logistic else 'logistic'

# Cross-validation
cv_df = cross_validation(best_model, initial='365 days', period='90 days', horizon='30 days')
cv_metrics = performance_metrics(cv_df)

# Streamlit UI
st.title("📈 Demand Forecasting Dashboard")
st.write(f"Predicting daily energy demand using temperature and dew point with a Prophet model ({growth_type} growth).")

# Display model performance
st.subheader("Model Performance (Last 30 Days)")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{best_mae:.2f}")
col2.metric("RMSE", f"{best_rmse:.2f}")
col3.metric("MAPE", f"{best_mape:.2f}%" if not np.isnan(best_mape) else "N/A")

# Cross-validation results
st.subheader("Cross-Validation Performance")
st.write("Average performance across multiple test periods:")
st.dataframe(cv_metrics[['horizon', 'mae', 'rmse', 'mape']].round(2))

# User input for prediction
st.subheader("🔮 Predict Demand")
min_date = df['ds'].min().date()
max_date = pd.Timestamp('2025-12-31').date()
temp_min, temp_max = df['temp'].min(), df['temp'].max()
dew_min, dew_max = df['dew'].min(), df['dew'].max()

col1, col2, col3 = st.columns(3)
with col1:
    user_date = st.date_input("Select Date", value=max_date, min_value=min_date, max_value=max_date)
with col2:
    user_temp = st.number_input("Temperature (°C)", min_value=float(temp_min), max_value=float(temp_max), value=float(df['temp'].mean()), format="%.2f")
with col3:
    user_dew = st.number_input("Dew Point (°C)", min_value=float(dew_min), max_value=float(dew_max), value=float(df['dew'].mean()), format="%.2f")

# Predict for user input
input_df = pd.DataFrame({
    'ds': [pd.Timestamp(user_date)],
    'temp': [user_temp],
    'dew': [user_dew]
})
if growth_type == 'logistic':
    input_df['cap'] = np.percentile(df['y'], 97.5)
    input_df['floor'] = np.percentile(df['y'], 2.5)
forecast_input = best_model.predict(input_df)
predicted_demand = forecast_input['yhat'].values[0]

st.success(f"""
📅 Date: {user_date}  
🌡 Temperature: {user_temp:.2f} °C  
💧 Dew Point: {user_dew:.2f} °C  
🔮 Predicted Demand: {predicted_demand:.2f}
""")

# Plot actual vs predicted
st.subheader("📊 Actual vs Predicted Demand (Last 30 Days)")
plot_df = test_df[['ds', 'y']].copy().reset_index(drop=True)
plot_df['Predicted'] = best_forecast['yhat'].values
plot_df.rename(columns={'y': 'Actual'}, inplace=True)
fig = px.line(plot_df, x='ds', y=['Actual', 'Predicted'], title="Actual vs Predicted Demand",
              labels={'value': 'Demand', 'ds': 'Date', 'variable': 'Type'})
fig.update_layout(legend_title="Type", font_size=12, template="plotly_white", showlegend=True)
fig.update_xaxes(tickangle=45, tickformat="%b %d, %Y")
st.plotly_chart(fig, use_container_width=True)

# Residual plot
st.subheader("📉 Residuals (Prediction Errors)")
residuals = best_forecast['yhat'] - test_df['y'].reset_index(drop=True)
residual_df = pd.DataFrame({'ds': test_df['ds'], 'Residuals': residuals}).reset_index(drop=True)
fig_residuals = px.line(residual_df, x='ds', y='Residuals', title="Prediction Residuals",
                        labels={'Residuals': 'Error (Predicted - Actual)', 'ds': 'Date'})
fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
fig_residuals.update_layout(font_size=12, template="plotly_white")
fig_residuals.update_xaxes(tickangle=45, tickformat="%b %d, %Y")
st.plotly_chart(fig_residuals, use_container_width=True)

# Model diagnostics
st.subheader("🔍 Model Diagnostics")
st.write("Correlation between variables and demand:")
corr_temp = df['y'].corr(df['temp'])
corr_dew = df['y'].corr(df['dew'])
st.write(f"- Temperature vs Demand: {corr_temp:.2f}")
st.write(f"- Dew Point vs Demand: {corr_dew:.2f}")
st.write(f"Note: Using {growth_type} growth. Linear growth may perform better for colder months like January 2025.")