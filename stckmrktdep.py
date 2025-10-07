import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Apple Stock Forecast (SARIMA)", layout="wide")
st.title("🍎 Apple Stock Forecasting using SARIMA")

# --- Load the CSV file ---
@st.cache_data
def load_data():
    df = pd.read_csv("D:\Data science April2025\P584 Stock Market Analytics project3\AAPL (4) (3).csv")
    # ✅ Parse date in DD-MM-YYYY format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])  # remove any unparsable rows
    df.set_index('Date', inplace=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")
    st.stop()

st.subheader("📊 Apple Stock Data Preview")
st.dataframe(df.tail())

# --- Use Close column ---
if 'Close' not in df.columns:
    st.error("The CSV file must contain a 'Close' column.")
    st.stop()

series = df['Close']

# --- Train/Test Split ---
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# --- Fit SARIMA Model (update with your best params if available) ---
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

st.subheader("🚀 Training SARIMA Model...")
model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)
st.success("✅ Model trained successfully!")

# --- Forecast ---
forecast_steps = len(test)
forecast = results.forecast(steps=forecast_steps)
forecast.index = test.index

# --- Plot Forecast vs Actual ---
st.subheader("📈 Forecast vs Actual")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(train.index, train, label='Train')
ax.plot(test.index, test, label='Actual', color='blue')
ax.plot(forecast.index, forecast, label='Forecast', color='red')
ax.legend()
st.pyplot(fig)

# --- Metrics ---
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")

# --- Future Forecast ---
st.subheader("🔮 30-Day Future Forecast")
steps = st.slider("Select forecast days", min_value=7, max_value=90, value=30)
future_forecast = results.forecast(steps=steps)
future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1),
                             periods=steps, freq='D')
future_df = pd.DataFrame({'Date': future_dates, 'Forecast': future_forecast})

# --- Plot Future Forecast ---
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(series.index, series, label='Historical')
ax2.plot(future_df['Date'], future_df['Forecast'], label='Future Forecast', color='red')
ax2.legend()
st.pyplot(fig2)

# --- Show Forecast Table ---
st.dataframe(future_df)

# --- Download Forecast CSV ---
csv = future_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Forecast CSV", data=csv,
                   file_name='apple_sarima_forecast.csv', mime='text/csv')
