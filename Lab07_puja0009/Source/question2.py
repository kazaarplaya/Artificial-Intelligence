# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab07_puja0009\Source\question2.py
# Date: 18-10-2026
# Description: Evaluate the change in RMSE and MAE as the number of nodes in each layer is increased from 10 to 300
# Usage: python Lab07_puja0009\Source\question2.py

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


# %%
data = sm.datasets.sunspots.load_pandas().data
sunspots = data["SUNACTIVITY"].values

# Create lagged features
def add_lagged_features(data, lag=12):
    X, y = [], []

    for i in range(lag, len(data)):
        X.append(data[i-lag:i])
        y.append(data[i])

    return np.array(X), np.array(y)

X, y = add_lagged_features(sunspots, lag=12)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# Scale inputs
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)

# Scale targets
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
).ravel()

# %%
neuron_counts = range(10, 301, 10)

results = []
predictions = {}

for neurons in neuron_counts:

    mlp = MLPRegressor(
        hidden_layer_sizes=(neurons, neurons),
        activation="relu",
        solver="adam",
        max_iter=2500,
        random_state=42
    )

    # Train model
    mlp.fit(X_train_scaled, y_train_scaled)

    # Make predictions
    y_pred_scaled = mlp.predict(X_test_scaled)

    # Convert predictions to original scale
    y_pred = y_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).ravel()

    # Evaluate
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "Neurons": neurons,
        "RMSE": rmse,
        "MAE": mae
    })

    predictions[neurons] = y_pred

results_df = pd.DataFrame(results)

print(results_df)

# %%
plt.figure(figsize=(10, 5))

plt.plot(
    results_df["Neurons"],
    results_df["RMSE"],
    marker="o",
    label="RMSE"
)

plt.plot(
    results_df["Neurons"],
    results_df["MAE"],
    marker="s",
    label="MAE"
)

plt.xlabel("Number of Neurons per Hidden Layer")
plt.ylabel("Prediction Error")
plt.title("MLP Performance vs Number of Neurons")

plt.legend()
plt.grid(True)
plt.show()

# %%
plt.figure(figsize=(12, 5))

plt.plot(
    y_test,
    label="Actual Sunspots",
    color="black",
    linewidth=2
)

for neurons in [10, 50, 100, 200, 300]:
    plt.plot(
        predictions[neurons],
        label=f"{neurons} Neurons",
        alpha=0.7
    )

plt.xlabel("Test Observation")
plt.ylabel("Sunspot Activity")
plt.title("Actual vs Predicted Sunspot Activity")

plt.legend()
plt.show()


