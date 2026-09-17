# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab06_puja0009/Source/question3.py
# Date: 18-09-2026
# Description: Compare different ARIMA models with different orders
# Usage: python Lab06_puja0009/Source/question3.py

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
response = requests.get(endpoint, params=params)
response.raise_for_status()

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
# Fit the model on the training set
arimamodel_510 = ARIMA(train, order=(5, 1, 0))
arimamodel_fit_510 = arimamodel_510.fit()
# Make predictions on the test set
predictions_510 = arimamodel_fit_510.forecast(steps=len(test))

# %%
# Split data into training and test sets
train_size = int(len(wdf) * 0.8)
train, test = wdf['Temp'][:train_size], wdf['Temp'][train_size:]
# Fit the model on the training set
arimamodel_111 = ARIMA(train, order=(1, 1, 1))
arimamodel_fit_111 = arimamodel_111.fit()
# Make predictions on the test set
predictions_111 = arimamodel_fit_111.forecast(steps=len(test))

# %%
# Split data into training and test sets
train_size = int(len(wdf) * 0.8)
train, test = wdf['Temp'][:train_size], wdf['Temp'][train_size:]
# Fit the model on the training set
arimamodel_212 = ARIMA(train, order=(2, 1, 2))
arimamodel_fit_212 = arimamodel_212.fit()
# Make predictions on the test set
predictions_212 = arimamodel_fit_212.forecast(steps=len(test))

# %%
import numpy as np
from sklearn.metrics import mean_squared_error

# Assuming 'actual' and 'predictions' are arrays/lists of equal length
rmse_510 = np.sqrt(mean_squared_error(test, predictions_510))
rmse_111 = np.sqrt(mean_squared_error(test, predictions_111))
rmse_212 = np.sqrt(mean_squared_error(test, predictions_212))
print(f"RMSE (5_1_0): {rmse_510}")
print(f"RMSE (1_1_1): {rmse_111}")
print(f"RMSE (2_1_2): {rmse_212}")


