import streamlit as st
import pandas as pd
import pickle

st.title("🔮 Customer Churn Prediction")

# Load model
with open("models/decision.pkl","rb") as f:
    model = pickle.load(f)

with open("models/features.pkl","rb") as f:
    feature_names = pickle.load(f)

# Inputs

contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

internet = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

# Predict Button

if st.button("Predict"):

    input_df = pd.DataFrame({
        "Contract":[contract],
        "tenure":[tenure],
        "InternetService":[internet],
        "MonthlyCharges":[monthly]
    })

    input_df = pd.get_dummies(input_df)

    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("Customer Likely To Churn ❌")
    else:
        st.success("Customer Likely To Stay ✅")