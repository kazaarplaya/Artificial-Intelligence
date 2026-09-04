# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
 accuracy_score,
 confusion_matrix,
 classification_report,
 roc_auc_score,
 log_loss,
 RocCurveDisplay,
)

# %%
from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features
y = wine_quality.data.targets["quality"]

# Defining good quality as anything 7 or above for wine quality
y_binary = (y >= 7).astype(int)


# %%
X_train, X_test, y_train, y_test = train_test_split(
 X, y_binary, test_size=0.2, random_state=4
)

scaler = StandardScaler()
# Scale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
CWmodel = LogisticRegression(class_weight="balanced")
# Train the model
CWmodel.fit(X_train_scaled, y_train)

# %%
log_loss_value = log_loss(y_test, CWmodel.predict_proba(X_test_scaled))
print("Log Loss:", log_loss_value)

# %%
PrecisionRecallDisplay.from_estimator(
    CWmodel,
    X_test_scaled,
    y_test
)

plt.title("Precision-Recall Curve")
plt.show()


