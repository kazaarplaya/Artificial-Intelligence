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
  
# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
X = ionosphere.data.features 
y = LabelEncoder().fit_transform(ionosphere.data.targets)

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=4
)
# y = ionosphere.data.targets 

# %%
scaler = StandardScaler()
# Scale the features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
model = LogisticRegression(l1_ratio=1, solver="saga", max_iter=3000)
model.fit(X_train_scaled, y_train)

# %%
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
print("ROC AUC Score:", roc_auc)
# Calculate Log Loss
log_loss_value = log_loss(y_test, model.predict_proba(X_test_scaled))
print("Log Loss:", log_loss_value)

# %%
RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
plt.show()


