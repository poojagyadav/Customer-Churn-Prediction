import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- 1. PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn System",
    page_icon="📊",
    layout="wide"
)


sns.set_style("whitegrid")


st.markdown("<h1 style='text-align: center;'>📊 Customer Churn Prediction System</h1>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; color: #555; font-size: 1.1rem; margin-bottom: 20px;'>
        Welcome to the Churn Analysis Dashboard. This tool helps you visualize customer behavior 
        patterns and use Machine Learning to predict the likelihood of customer turnover. 
        Analyze the distribution charts below or use the predictor at the bottom.
    </div>
""", unsafe_allow_html=True)

st.divider()

# DATA LOADING 
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/Customer_Churn.csv")
        return df
    except:
        return pd.DataFrame({
            "Contract": ["Month-to-month", "One year", "Two year"] * 30,
            "InternetService": ["Fiber optic", "DSL", "No"] * 30,
            "tenure": [1, 5, 10, 24, 48, 72] * 15,
            "MonthlyCharges": [20.5, 70.0, 110.0, 50.0, 85.0, 25.0] * 15
        })

df = load_data()

# Load models
try:
    with open("models/svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/features.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    model, feature_names, scaler = None, None, None

#plots
# Standardized size for all plots
FIG_SIZE = (6, 4) 

def plot_pie(column, data_frame):
    data_ = data_frame[column].value_counts()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.pie(data_.values, labels=data_.index, autopct="%1.1f%%", startangle=90, colors=sns.color_palette("Pastel1"))
    ax.axis('equal') 
    plt.tight_layout(pad=3.0) 
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def plot_count(column, data_frame):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.countplot(data=data_frame, x=column, palette="Pastel1", ax=ax)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def plot_hist(column, data_frame):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.histplot(data_frame[column], bins=20, color="skyblue", kde=True, ax=ax)
    plt.tight_layout(pad=3.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def plot_box(column, data_frame):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(data=data_frame, x=column, color="lightcoral", ax=ax)
    # This empty label ensures the plot area width matches the histogram next to it
    ax.set_ylabel("Count", color="white", alpha=0) 
    plt.tight_layout(pad=3.0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

#EDA
st.markdown("<h2 style='text-align: center;'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
st.write("<br>", unsafe_allow_html=True)

# Row 1
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.markdown("<h4 style='text-align: center;'>Contract Distribution</h4>", unsafe_allow_html=True)
    plot_pie("Contract", df)

with row1_col2:
    st.markdown("<h4 style='text-align: center;'>Internet Service Usage</h4>", unsafe_allow_html=True)
    plot_count("InternetService", df)

# Row 2
row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.markdown("<h4 style='text-align: center;'>Customer Tenure (Months)</h4>", unsafe_allow_html=True)
    plot_hist("tenure", df)

with row2_col2:
    st.markdown("<h4 style='text-align: center;'>Monthly Charges Spread</h4>", unsafe_allow_html=True)
    plot_box("MonthlyCharges", df)

st.divider()

# Prediction
st.markdown("<h2 style='text-align: center;'>🔮 Predict Customer Churn</h2>", unsafe_allow_html=True)

with st.container():
    c1, c2 = st.columns(2)
    with c1:
        contract = st.selectbox("Select Contract Type", ["Month-to-month", "One year", "Two year"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
    
    with c2:
        internet = st.selectbox("Select Internet Service", ["DSL", "Fiber optic", "No"])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, value=50.0)

st.write("<br>", unsafe_allow_html=True)

_, btn_center, _ = st.columns([2, 1, 2])

with btn_center:
    predict_btn = st.button("🚀 Run Prediction", use_container_width=True)

# Result Output
if predict_btn:
    st.write("<br>", unsafe_allow_html=True)
    if model is not None:
        try:
            input_df = pd.DataFrame({
                "Contract": [contract], "tenure": [tenure],
                "InternetService": [internet], "MonthlyCharges": [monthly_charges]
            })
            input_df = pd.get_dummies(input_df).reindex(columns=feature_names, fill_value=0)
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]

            if prediction == 1:
                st.error("### ⚠️ Result: The customer is likely to CHURN")
            else:
                st.success("### ✅ Result: The customer is likely to STAY")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        # Dummy Logic
        if contract == "Month-to-month" and monthly_charges > 65:
            st.error("Customer likely to CHURN")
        else:
            st.success("Customer likely to STAY")