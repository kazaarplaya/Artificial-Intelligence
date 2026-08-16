# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab03_puja0009/Source/question4.py
# Date: 16-08-2026
# Description: KMeans clustering + evaluation accuracy
# Usage: python Lab03_puja0009/Source/question4.py

# %%
from sklearn.datasets import load_wine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# %%
wine = load_wine()
X = wine.data
y = wine.target

wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
print(wine_df.head())

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# %%
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# %%
inertias = []

k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_train_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(k_values, inertias, marker="o")

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Wine Dataset")

plt.show()

# %%
optimal_k = 3 # Identified using elbow method

# %%
kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=20
)

train_clusters = kmeans.fit_predict(
    X_train_scaled
)

# %%
print(kmeans.cluster_centers_)

# %%
cluster_to_class = {}

for cluster in range(optimal_k):

    actual_classes = y_train[
        train_clusters == cluster
    ]

    majority_class = np.bincount(
        actual_classes
    ).argmax()

    cluster_to_class[cluster] = majority_class


print(cluster_to_class)

# %%
test_clusters = kmeans.predict(
    X_test_scaled
)

y_pred = np.array([
    cluster_to_class[cluster]
    for cluster in test_clusters
])

# %%
accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"Accuracy: {accuracy:.2%}")


