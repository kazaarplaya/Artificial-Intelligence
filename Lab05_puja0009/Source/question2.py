# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab05_puja0009\Source\question2.py
# Date: 10-09-2026
# Description: Calculate and interpret the Precision and Recall across the three classes for a Decision Tree Classifier on the Iris dataset
# Usage: python Lab05_puja0009\Source\question2.py

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score, classification_report
# Additional libraries for handling imbalanced datasets
from sklearn.utils.class_weight import compute_class_weight

# %%
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# %%
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

# %%
print("Per-Class Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


