import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")

st.markdown("""
### Welcome

This project predicts whether a customer is likely to leave the telecom company.

### Model Used
- Decision Tree Classifier

### Features Used
- Contract
- Tenure
- Internet Service
- Monthly Charges
""")