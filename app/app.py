import streamlit as st
import pandas as pd
import joblib

model = joblib.load(r"D:\Projects\ML PROJECT 1\models\best_model.joblib")

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure", 0, 72)
MonthlyCharges = st.number_input("Monthly Charges")

if st.button("Predict"):

    data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [SeniorCitizen],
        "tenure": [tenure],
        "MonthlyCharges": [MonthlyCharges]
    })

    prediction = model.predict(data)

    st.sucess(f"Prediction: {prediction[0]}")