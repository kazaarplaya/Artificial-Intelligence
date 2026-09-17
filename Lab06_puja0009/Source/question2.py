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
# Fit the model on the training set
arimamodel = ARIMA(train, order=(5, 1, 0))
arimamodel_fit = arimamodel.fit()
# Make predictions on the test set
predictions = arimamodel_fit.forecast(steps=len(test))

# %%
import numpy as np
from sklearn.metrics import mean_squared_error

# Assuming 'actual' and 'predictions' are arrays/lists of equal length
rmse = np.sqrt(mean_squared_error(test, predictions))
print(f"RMSE: {rmse}")


