from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Data/ncaa_data.csv")

# Fix target if needed
if df["Drafted"].dtype == "object":
    df["Drafted"] = df["Drafted"].map({"Yes": 1, "No": 0})

y = df["Drafted"]

# Features (same as before)
features = [
    "GP", "GS", "MP", "FG", "FGA", "2P", "2PA",
    "3P", "3PA", "FT", "FTA", "ORB", "DRB", "TRB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "FG%", "2P%", "3P%", "FT%", "TS%", "eFG%"
]
X = df[features]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=67,
    stratify=y
)

# Calculate scale_pos_weight
num_negative = (y_train == 0).sum()
num_positive = (y_train == 1).sum()

scale_pos_weight = num_negative / num_positive

print(scale_pos_weight)
print(df.head())
print(df.shape)
from sklearn.model_selection import GridSearchCV
import numpy as np


param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 4, 5],
}

search = GridSearchCV(
    estimator=XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=67,
    ),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best val accuracy", search.best_score_)
search.best_params_
search.best_estimator_

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("Data/ncaa_data.csv")

# Fix target if needed
if df["Drafted"].dtype == "object":
    df["Drafted"] = df["Drafted"].map({"Yes": 1, "No": 0})

# Fix column name if needed
df.columns = df.columns.str.strip()

# Features
features = [
    "GP", "GS", "MP", "FG", "FGA", "2P", "2PA", "3P", "3PA", "FT", "FTA",
    "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "FG%", "2P%", "3P%", "FT%", "TS%", "eFG%"
]

X = df[features]
y = df["Drafted"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)

# Scale positive weight
num_negative = (y_train == 0).sum()
num_positive = (y_train == 1).sum()
scale_pos_weight = num_negative / num_positive

# Grid search
param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4, 5],
}

search = GridSearchCV(
    estimator=XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=67,
    ),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best val accuracy:", search.best_score_)

# Final tuned model
final_model = XGBClassifier(
    n_estimators=search.best_params_["n_estimators"],
    max_depth=search.best_params_["max_depth"],
    scale_pos_weight=scale_pos_weight,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=67,
)

final_model.fit(X_train, y_train)

# Predictions
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\nTuned Model Test Metrics:")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("ROC-AUC:", roc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(cm)

import matplotlib.pyplot as plt

importances = final_model.get_booster().get_score(importance_type='weight')
importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

plt.figure(figsize=(10, 6))
plt.barh(list(importances.keys())[:15], list(importances.values())[:15])
plt.xlabel('Weight')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()