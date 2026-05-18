import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load model
model = joblib.load(r"D:\Projects\ML PROJECT 1\models\best_model.joblib")

st.title("Customer Churn Prediction")

st.write("Fill customer details to predict churn probability.")

st.sidebar.title("Customer Churn Prediction")
st.sidebar.write("ML Powered Prediction System")

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

st.divider()

if st.button("Predict"):

    with st.spinner("Predicting..."):
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

        try:

            prediction = model.predict(data)

            if prediction[0] == "Yes":
                st.error("Customer is likely to Churn")
            else:
                st.success("Customer is likely to Stay")

        except Exception as e:

            st.error(f"Error occurred: {e}")

    # SHAP Explainability

    try:

        st.subheader("SHAP Feature Importance")

        transformed_data = model.named_steps["preprocessor"].transform(data)

        explainer = shap.LinearExplainer(
         model.named_steps["model"],
            transformed_data
        )

        shap_values = explainer.shap_values(transformed_data)

        feature_names = model.named_steps[
            "preprocessor"
        ].get_feature_names_out()

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values[0]
        })

        st.dataframe(shap_df)

    except Exception as e:

        st.warning(f"SHAP visualization could not load: {e}")