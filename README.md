# Customer Churn Prediction

## Project Overview

This project is an End-to-End Machine Learning Pipeline with Streamlit Deployment for predicting customer churn in a telecom company.

The application predicts whether a customer is likely to leave the company based on customer details, billing information, and service usage.

---

## Features

- Data preprocessing pipeline
- Machine Learning model training
- Logistic Regression model
- Model serialization using Joblib
- Streamlit web application
- Real-time prediction
- SHAP explainability
- Error handling
- Interactive UI

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- SHAP
- Joblib
- Matplotlib
- Seaborn

---

## Dataset

Dataset Used:
Telco Customer Churn Dataset

Source:
Kaggle

---

## Project Structure

ML PROJECT 1/

│

├── app/

│   └── app.py

│

├── data/

│   └── customer_churn.csv

│

├── models/

│   └── best_model.joblib

│

├── notebooks/

│   └── experimentation.ipynb

│

├── requirements.txt

│

└── README.md

---

## Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Feature Engineering
4. Data Preprocessing
5. Model Training
6. Model Evaluation
7. Model Serialization
8. Streamlit Deployment
9. SHAP Explainability

---

## Model Accuracy

- Logistic Regression Accuracy: 82%
- Random Forest Accuracy: 78%

Selected Model:
Logistic Regression

---

## How to Run the Project

### Step 1

Clone repository

```bash
git clone https://github.com/DRPatel75/End-to-End-Predictive-Modeling-Pipeline-with-Streamlit-Deployment.git
```

Step 2

Install dependencies

pip install -r requirements.txt
Step 3

Run Streamlit app

streamlit run app.py
SHAP Explainability

SHAP is used to explain model predictions and feature importance.

Future Improvements
Better UI design
Database integration
User authentication
Deployment on cloud
Advanced visualization

Author
D.R.Patel

Developed as a Machine Learning Deployment Project.
