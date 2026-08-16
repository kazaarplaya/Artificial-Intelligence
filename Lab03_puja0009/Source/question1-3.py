# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab03_puja0009/Source/question1-3.py
# Date: 16-08-2026
# Description: Applied K-means clustering to the wine dataset and determined optimal number of clusters using Elbow method. NOTE: DATA NOT SCALED
# Usage: python Lab03_puja0009/Source/question1-3.py

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# %%
wine = load_wine()
X = wine.data
y = wine.target

wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
print(wine_df.head())

# %%
print(wine_df.describe())

# %%
kmeans = KMeans(n_clusters=3, random_state=1)
kmeans.fit(X)
labels = kmeans.labels_

wine_df['Cluster'] = labels
sns.pairplot(wine_df, hue='Cluster', palette='viridis', markers=["o", "s", "D"])
plt.suptitle('Pair Plot of Wine Dataset with K-Means Clusters', y=1.02)
plt.show()

# %%
silhouette_avg = silhouette_score(X, labels)
print(f'Silhouette Score: {silhouette_avg}')

# %%
silhouette_scores = []
cluster_range = range(2,11)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker="o")
plt.title("Silhouette Scores for Different Numbers of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(cluster_range)
plt.grid(True)
plt.show()

# %%
print(f'Silhouette Score: {silhouette_avg}')
print(f'Best number of clusters using Elbow method: 2 ({silhouette_scores[0]})')
