<div align="left">

# 📊 Feature Selection Comparison: Filter vs Wrapper

A professional Streamlit web application for predicting breast cancer classification (Benign or Malignant) using Logistic Regression along with varying feature selection techniques. This interactive app allows users to predict cancer malignancy based on numeric features and explore model performance between Baseline, Filter Method (SelectKBest), and Wrapper Method (RFE).

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🌟 Features

### 📋 Feature Selection Comparison
- **Baseline Model:** Logistic Regression using all 30 features from the Breast Cancer dataset.
- **Filter Method (SelectKBest):** Evaluates features statistically (ANOVA F-value) and selects the 10 most prominent features.
- **Wrapper Method (RFE):** Uses Recursive Feature Elimination wrapped around Logistic Regression to incrementally select the optimal 10 features.
- **Real-Time Accuracy Metric:** Visual representation of accuracy differences between all three models directly on the interface.

### 🔍 Interactive Output & Prediction
- **Interactive Input Form:** Manually enter specific values for 10 prominent features on the sidebar or via numeric inputs.
- **Instant Predictions:** Predict whether the tumor is **Benign** or **Malignant**.
- **Model Toggling:** Easily switch between the Baseline, Filter Method, or Wrapper Method models to see how feature selection alters prediction results.
- **Feature Review:** Quick checklist showcasing which exact features were preserved by both the Filter and Wrapper selection methods.

---

## 🛠️ How to Run

1. **Clone the repository**
```bash
git clone https://github.com/ayyapparaja227-arch/Feature-Selection-Filter-Wrapper.git
cd Feature-Selection-Filter-Wrapper
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app.py
```

---

## 🧩 Model Training Script
A dedicated `model.py` script is included if you want to rebuild and export the models and selectors yourself. It handles data loading, feature scaling, model training, and exporting pipelines as `.pkl` objects to be loaded instantaneously by the frontend.
</div>