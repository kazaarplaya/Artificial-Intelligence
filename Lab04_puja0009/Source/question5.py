# Student Name: Hans Pujalte
# Student FAN: puja0009
# File: Lab04_puja0009/Source/question5.py
# Date: 04-09-2026
# Description: Implement Logistic Classification with different solvers (liblinear, lbfgs, saga) on the MNIST dataset
# Usage: python Lab04_puja0009/Source/question5.py

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
 accuracy_score,
 confusion_matrix,
 classification_report,
 roc_auc_score,
 log_loss,
 RocCurveDisplay,
)

# %%
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

X = X / 255.0

X = X[:10000]
y = y[:10000]

# %%
models = {
    "liblinear": OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            max_iter=1000
        )
    ),
    "lbfgs": LogisticRegression(
        solver="lbfgs",
        max_iter=1000
    ),
    "saga": LogisticRegression(
        solver="saga",
        max_iter=1000
    )
}

# %%
results = {}
for solver_name, model in models.items():
    print(solver_name)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    results[solver_name] = scores

    print("Cross-validation scores:", scores)
    print("Mean accuracy:", scores.mean())
    print("Standard deviation:", scores.std())
    print()


# %%
solver_names = list(results.keys())

mean_scores = [
    results[name].mean()
    for name in solver_names
]

plt.bar(
    solver_names,
    mean_scores
)

plt.xlabel("Solver")
plt.ylabel("Mean Cross-Validation Accuracy")
plt.title("Logistic Regression Solvers on MNIST")

plt.ylim(0.90, .902)

plt.show()

# %%
print("Final Results:")
for solver in solver_names:
    print(
        f"{solver}: "
        f"{results[solver].mean():.4f} "
        f"std: {results[solver].std():.4f}"
    )


