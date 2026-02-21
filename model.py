# ==========================================
# Feature Selection - Filter vs Wrapper
# Breast Cancer Dataset
# ==========================================

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = load_breast_cancer()

X = data.data
y = data.target
feature_names = data.feature_names

# -----------------------------
# 2. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Scale Features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4. Baseline Model (No FS)
# -----------------------------
base_model = LogisticRegression(max_iter=5000)
base_model.fit(X_train_scaled, y_train)

base_pred = base_model.predict(X_test_scaled)
base_acc = accuracy_score(y_test, base_pred)

print("Baseline Accuracy:", base_acc)

# -----------------------------
# 5. Filter Method (SelectKBest)
# -----------------------------
filter_selector = SelectKBest(score_func=f_classif, k=10)
X_train_filter = filter_selector.fit_transform(X_train_scaled, y_train)
X_test_filter = filter_selector.transform(X_test_scaled)

filter_model = LogisticRegression(max_iter=5000)
filter_model.fit(X_train_filter, y_train)

filter_pred = filter_model.predict(X_test_filter)
filter_acc = accuracy_score(y_test, filter_pred)

selected_filter_features = feature_names[filter_selector.get_support()]

print("Filter Accuracy:", filter_acc)
print("Selected Filter Features:", selected_filter_features)

# -----------------------------
# 6. Wrapper Method (RFE)
# -----------------------------
wrapper_selector = RFE(
    estimator=LogisticRegression(max_iter=5000),
    n_features_to_select=10
)

X_train_wrapper = wrapper_selector.fit_transform(X_train_scaled, y_train)
X_test_wrapper = wrapper_selector.transform(X_test_scaled)

wrapper_model = LogisticRegression(max_iter=5000)
wrapper_model.fit(X_train_wrapper, y_train)

wrapper_pred = wrapper_model.predict(X_test_wrapper)
wrapper_acc = accuracy_score(y_test, wrapper_pred)

selected_wrapper_features = feature_names[wrapper_selector.get_support()]

print("Wrapper Accuracy:", wrapper_acc)
print("Selected Wrapper Features:", selected_wrapper_features)

# -----------------------------
# 7. Save Everything
# -----------------------------
joblib.dump(base_model, "baseline_model.pkl")
joblib.dump(filter_model, "filter_model.pkl")
joblib.dump(wrapper_model, "wrapper_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(filter_selector, "filter_selector.pkl")
joblib.dump(wrapper_selector, "wrapper_selector.pkl")

print("\nModels and selectors saved successfully!")