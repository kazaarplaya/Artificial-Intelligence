# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab06_puja0009/Source/question4.py
# Date: 18-09-2026
# Description: Perform 7-step rolling forecast 
# Usage: python Lab06_puja0009/Source/question4.py

# %%
import matplotlib.pyplot as plt
import pandas as pd
import requests
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

# %%
# Define the endpoint and parameters
endpoint = 'https://archive-api.open-meteo.com/v1/archive'
params = {
    'latitude': -34.9285,       # Latitude for Adelaide
    'longitude': 138.6007,      # Longitude for Adelaide
    'start_date': datetime(2026, 6, 1).strftime('%Y-%m-%d'),
    'end_date': datetime(2026, 8, 31).strftime('%Y-%m-%d'),
    'daily': 'temperature_2m_mean',
    'timezone': 'Australia/Adelaide'
}

# %%
# Make the API call
response = requests.get(endpoint, params=params)
wdata = response.json()

# Extract the daily temperature data
timestamps = [
    datetime.fromisoformat(item)
    for item in wdata['daily']['time']
]

temperatures = wdata['daily']['temperature_2m_mean']

# Create a DataFrame
wdf = pd.DataFrame({
    'Date': timestamps,
    'Temp': temperatures
})

# %%
# Plot the daily temperature data
plt.figure(figsize=(12, 5))

plt.plot(wdf.index, wdf['Temp'])

plt.title("Adelaide Daily Mean Temperature - Winter 2026")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.grid(True)

plt.tight_layout()
plt.show()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %%
# Split data into training and test sets
train_size = int(len(wdf) * 0.8)
train, test = wdf['Temp'][:train_size], wdf['Temp'][train_size:]
order_510 = (5,1,0)
history = train.copy()
forecast_steps = 7

# %%
rolling_preds = []
for start in range(0, len(test), forecast_steps):
    model = ARIMA(history, order=order_510)
    model_fit = model.fit()
    
    remaining = len(test) - start
    # Forecast either 7 steps or however many are left
    steps = min(forecast_steps, remaining)
    # Forecast next 7 observations
    forecast = model_fit.forecast(steps=steps)

    # Store predictions
    rolling_preds.extend(forecast)

    # Get the actual observations for this 7-step period
    actual_values = test.iloc[start:start + steps]
    history = pd.concat([history, actual_values])

# %%
rolling_predictions = np.array(rolling_preds)

mae = mean_absolute_error(test, rolling_predictions)
mse = mean_squared_error(test, rolling_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(test, rolling_predictions)

print(f"MAE:  {mae:.3f}")
print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")


