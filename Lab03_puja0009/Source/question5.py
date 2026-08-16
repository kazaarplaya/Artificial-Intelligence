# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab03_puja0009/Source/question5.py
# Date: 16-08-2026
# Description: KMeans clustering + experimented with different scaling methods and initialisers
# Usage: python Lab03_puja0009/Source/question5.py

# %%
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# %%
wine = load_wine()
X = wine.data
y = wine.target

wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
print(wine_df.head())

# %%
scalers = {
    'Standard': StandardScaler(), 
    'MinMax': MinMaxScaler(),
    'Robust': RobustScaler(),
}

initialisations = [
    "k-means++", 
    "random"
]

results = []

# %%
import matplotlib.pyplot as plt

for scaler_name, scaler in scalers.items():

    X_scaled_version = scaler.fit_transform(X)

    for init_method in initialisations:

        model = KMeans(
            n_clusters=3,
            init=init_method,
            n_init=20,
            random_state=42
        )

        labels = model.fit_predict(
            X_scaled_version
        )

        score = silhouette_score(
            X_scaled_version,
            labels
        )

        results.append({
            "Scaler": scaler_name,
            "Initialization": init_method,
            "Silhouette Score": score
        })

# %%
results_df = pd.DataFrame(results)

print(results_df)

# %%
selected_features = [
    "alcohol",
    "malic_acid",
    "flavanoids",
    "color_intensity",
    "proline"
]


for scaler_name, scaler in scalers.items():

    X_scaled_version = scaler.fit_transform(X)

    for init_method in initialisations:

        model = KMeans(
            n_clusters=3,
            init=init_method,
            n_init=20,
            random_state=42
        )

        model.fit(X)

        labels = model.labels_

        plot_df = pd.DataFrame(
            X_scaled_version,
            columns=wine.feature_names
        )

        plot_df["Cluster"] = labels.astype(str)

        print(
            f"{scaler_name} - {init_method}"
        )

        sns.pairplot(
            plot_df[
                selected_features +
                ["Cluster"]
            ],
            hue="Cluster"
        )

        plt.show()


