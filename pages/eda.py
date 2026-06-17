import streamlit as st
import pandas as pd

st.title("Exploratory Data Analysis")

df = pd.read_csv("data/Customer_Churn.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Shape")

col1,col2=st.columns(2)

with col1:
    st.metric("Rows",df.shape[0])

with col2:
    st.metric("Columns",df.shape[1])

st.subheader("Missing Values")
st.dataframe(df.isnull().sum())

st.subheader("Churn Distribution")
st.bar_chart(df["Churn"].value_counts())

st.subheader("Monthly Charges")
st.bar_chart(df["MonthlyCharges"])