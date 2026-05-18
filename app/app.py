import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load(r"D:\Projects\ML PROJECT 1\models\best_model.joblib")

st.title("Customer Churn Prediction")

# User Inputs

gender = st.selectbox("Gender", ["Male", "Female"])

SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

Partner = st.selectbox("Partner", ["Yes", "No"])

Dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.slider("Tenure", 0, 72)

PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

Contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0
)

# Prediction

if st.button("Predict"):

    data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "tenure": [tenure],
        "PhoneService": [PhoneService],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges]
    })

    prediction = model.predict(data)

    # Prediction Result

    if prediction[0] == "Yes":
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")

    # SHAP Explainability

    try:

        transformed_data = model.named_steps["preprocessor"].transform(data)

        explainer = shap.Explainer(model.named_steps["model"])

        shap_values = explainer(transformed_data)

        st.subheader("SHAP Explanation")

        fig, ax = plt.subplots()

        shap.plots.waterfall(shap_values[0], show=False)

        st.pyplot(fig)

    except Exception as e:

        st.warning(f"SHAP visualization could not load: {e}")