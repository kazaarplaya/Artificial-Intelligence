# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab05_puja0009\Source\question5.py
# Date: 10-09-2026
# Description: Full DTR vs pruned DTR
# Usage: python Lab05_puja0009\Source\question5.py

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
# Additional libraries for handling imbalanced datasets
from sklearn.utils.class_weight import compute_class_weight

# %%
california = fetch_california_housing()

X = california.data
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# %%
full_tree = DecisionTreeRegressor(random_state=42)
full_tree.fit(X_train, y_train)


# %%
path = full_tree.cost_complexity_pruning_path(X_train,y_train)
ccp_alphas = path.ccp_alphas

# %%
# Training was taking way too long so will just sample the alpha values instead
step = max(1, len(ccp_alphas) // 50)
ccp_alphas_sampled = ccp_alphas[::step]

param_grid = {
    "ccp_alpha": ccp_alphas_sampled
}

grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_["ccp_alpha"]
print("Best ccp_alpha:", best_alpha)

# %%
pruned_tree = DecisionTreeRegressor(
    random_state=42,
    ccp_alpha=best_alpha
)
pruned_tree.fit(X_train, y_train)

# %%
def eval_model(model, X_train, y_train, X_test, y_test):

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_r2 = r2_score(y_train, train_predictions)
    train_rmse = root_mean_squared_error(
        y_train,
        train_predictions
    )

    test_r2 = r2_score(y_test, test_predictions)
    test_rmse = root_mean_squared_error(
        y_test,
        test_predictions
    )

    return train_r2, train_rmse, test_r2, test_rmse

# %%
full_results = eval_model(
    full_tree,
    X_train,
    y_train,
    X_test,
    y_test
)

pruned_results = eval_model(
    pruned_tree,
    X_train,
    y_train,
    X_test,
    y_test
)

# %%
results = pd.DataFrame({
    "Model": [
        "Fully Fitted Decision Tree",
        "Pruned Decision Tree"
    ],

    "Train R²": [
        full_results[0],
        pruned_results[0]
    ],

    "Train RMSE": [
        full_results[1],
        pruned_results[1]
    ],

    "Test R²": [
        full_results[2],
        pruned_results[2]
    ],

    "Test RMSE": [
        full_results[3],
        pruned_results[3]
    ]
})

# %%
print("Model Performance:")
print(results)


