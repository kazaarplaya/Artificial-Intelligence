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
'latitude': -37.8136, # Latitude for Melbourne
'longitude': 144.9631, # Longitude for Melbourne
'start_date': (datetime(2026,6,1)).strftime('%Y-%m-%d'),
'end_date': (datetime(2026,8,31)).strftime('%Y-%m-%d'),
'hourly': 'temperature_2m',
'timezone': 'Australia/Melbourne'
}

# %%
# Make the API call
response = requests.get(endpoint, params=params)
wdata = response.json()
# Extract the relevant data
timestamps = [datetime.fromisoformat(item) for item in wdata['hourly']['time']]
temperatures = wdata['hourly']['temperature_2m']
# Create a DataFrame
wdf = pd.DataFrame({
'Date': timestamps,
'Temp': temperatures
})

# %%
# Set the Date column as the index
wdf.set_index('Date', inplace=True)
# Perform seasonal decomposition
decomp = seasonal_decompose(wdf['Temp'], model='additive', period=24)
# Assuming hourly data with daily seasonality
# Plot the decomposed components
decomp.plot()
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
order_212 = (2,1,2)
history = train.copy()
forecast_steps = 7

# %%
rolling_preds = []
for start in range(0, len(test), forecast_steps):
    model = ARIMA(history, order=order_212)
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


