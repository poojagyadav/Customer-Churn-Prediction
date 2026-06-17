import streamlit as st
import pandas as pd
import pickle


st.title("Customer Churn Prediction ")


with open("models/decision.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/features.pkl", "rb") as f:
    feature_names = pickle.load(f)


# INPUT UI

st.header("Enter Customer Details")

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

tenure = st.slider(
    "Tenure (Months)",
    0, 72, 12
)

internet = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    # value=50.0
)


# PREDICTION
if st.button("Predict Churn"):

    try:
        # IMPORTANT FEATURE ENGINEERING (MUST MATCH TRAINING)
        charge_per_month = monthly_charges / (tenure)

        # Create input dataframe
        input_df = pd.DataFrame({
            "Contract": [contract],
            "tenure": [tenure],
            "InternetService": [internet],
            "MonthlyCharges": [monthly_charges],
            "ChargePerMonth": [charge_per_month]   # ✅ FIX ADDED
        })

        # One-hot encoding
        input_df = pd.get_dummies(input_df)

        
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Prediction
        prediction = model.predict(input_df)[0]

        # Probability
        probability = model.predict_proba(input_df)[0][1]

       
        st.subheader("Result")

        

        if prediction == 1:
            st.error(" CHURN (Customer will leave service)")
        else:
            st.success("Not Churn (Customer will stay)")

    except Exception as e:
        st.error(f"Error: {e}")