# ==========================================
# Feature Selection Comparison App
# Breast Cancer Dataset
# ==========================================

import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Feature Selection Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Feature Selection: Filter vs Wrapper")
st.write("Comparison using Breast Cancer Dataset")

# -----------------------------
# Load Dataset
# -----------------------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# -----------------------------
# Load Models & Objects
# -----------------------------
baseline_model = joblib.load("baseline_model.pkl")
filter_model = joblib.load("filter_model.pkl")
wrapper_model = joblib.load("wrapper_model.pkl")

scaler = joblib.load("scaler.pkl")
filter_selector = joblib.load("filter_selector.pkl")
wrapper_selector = joblib.load("wrapper_selector.pkl")

# -----------------------------
# Accuracy Calculation
# -----------------------------
X_scaled = scaler.transform(X)

baseline_acc = accuracy_score(y, baseline_model.predict(X_scaled))
filter_acc = accuracy_score(
    y,
    filter_model.predict(filter_selector.transform(X_scaled))
)
wrapper_acc = accuracy_score(
    y,
    wrapper_model.predict(wrapper_selector.transform(X_scaled))
)

st.subheader("Model Accuracy Comparison")

col1, col2, col3 = st.columns(3)

col1.metric("Baseline (All Features)", round(baseline_acc, 4))
col2.metric("Filter Method (SelectKBest)", round(filter_acc, 4))
col3.metric("Wrapper Method (RFE)", round(wrapper_acc, 4))

st.divider()

# -----------------------------
# Show Selected Features
# -----------------------------
st.subheader("Selected Features")

filter_features = feature_names[filter_selector.get_support()]
wrapper_features = feature_names[wrapper_selector.get_support()]

col4, col5 = st.columns(2)

with col4:
    st.write("### Filter Method Features")
    for f in filter_features:
        st.write("-", f)

with col5:
    st.write("### Wrapper Method Features")
    for f in wrapper_features:
        st.write("-", f)

st.divider()

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔍 Try Prediction")

method = st.selectbox(
    "Choose Model",
    ["Baseline", "Filter Method", "Wrapper Method"]
)

inputs = []

st.write("Enter first 10 feature values (for demo):")

for i in range(10):
    value = st.number_input(feature_names[i], value=0.0)
    inputs.append(value)

if st.button("Predict"):

    input_array = np.array([inputs])

    # Pad to full 30 features for baseline
    if method == "Baseline":
        padded = np.zeros((1, 30))
        padded[0, :10] = inputs
        scaled = scaler.transform(padded)
        prediction = baseline_model.predict(scaled)

    elif method == "Filter Method":
        padded = np.zeros((1, 30))
        padded[0, :10] = inputs
        scaled = scaler.transform(padded)
        selected = filter_selector.transform(scaled)
        prediction = filter_model.predict(selected)

    else:
        padded = np.zeros((1, 30))
        padded[0, :10] = inputs
        scaled = scaler.transform(padded)
        selected = wrapper_selector.transform(scaled)
        prediction = wrapper_model.predict(selected)

    if prediction[0] == 1:
        st.success("✅ Benign Tumor")
    else:
        st.error("⚠️ Malignant Tumor")