import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn System",
    page_icon="📊",
    layout="wide"
)

# Set a clean style
sns.set_style("whitegrid")

# 1. SET PAGE CONFIG (Must be the very first Streamlit command)
st.set_page_config(
    page_title="Customer Churn System",
    page_icon="📊",
    layout="wide"
)

# 2. APP TITLE & HEADER (Placed here so they always show up first)
st.title("📊 Customer Churn Prediction System")

st.divider()

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("data/Customer_Churn.csv")
    with open("models/svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/features.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    pass # Local dev bypass

# =====================================================
# THE FIX: HARD-CODED DIMENSIONS & MARGINS
# =====================================================
FIG_WIDTH = 5
FIG_HEIGHT = 4

def setup_standard_fig():
    """Creates a figure and forces fixed margins so they align perfectly."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    # This is the secret: it locks the 'box' of the graph to specific coordinates
    # left, bottom, right, top are percentages of the figure size (0 to 1)
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.85)
    return fig, ax

def plot_pie(column, data_frame):
    data_ = data_frame[column].value_counts()
    fig, ax = setup_standard_fig()
    
    # Use a smaller radius to ensure labels don't get cut off
    ax.pie(
        data_.values,
        labels=data_.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set2"),
        radius=1.1
    )
    ax.set_title(f"Distribution of {column}", fontweight='bold', pad=10)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def plot_count(column, data_frame):
    fig, ax = setup_standard_fig()
    sns.countplot(data=data_frame, x=column, palette="Set2", ax=ax)
    ax.set_title(f"Count of {column}", fontweight='bold', pad=10)
    ax.set_xlabel("") # Remove x-label to prevent it from pushing the graph up
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def plot_hist(column, data_frame):
    fig, ax = setup_standard_fig()
    sns.histplot(data_frame[column], bins=30, color="skyblue", kde=True, ax=ax)
    ax.set_title(f"Frequency of {column}", fontweight='bold', pad=10)
    ax.set_xlabel("") 
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def plot_box(column, data_frame):
    fig, ax = setup_standard_fig()
    sns.boxplot(data=data_frame, y=column, color="orange", ax=ax)
    ax.set_title(f"Spread of {column}", fontweight='bold', pad=10)
    ax.set_ylabel("") # Remove y-label to match the others
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# =====================================================
st.header("📊 EDA Section")

# Row 1
col1, col2 = st.columns(2)
with col1:
    plot_pie("Contract", df)
with col2:
    plot_count("InternetService", df)

# Row 2
col3, col4 = st.columns(2)
with col3:
    plot_hist("tenure", df)
with col4:
    plot_box("MonthlyCharges", df)

st.divider()

# =====================================================
st.header("🔮 Customer Churn Prediction")

col_inp1, col_inp2 = st.columns(2)

with col_inp1:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col_inp2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    monthly_charges = st.number_input("Monthly Charges", 0.0, value=50.0)

if st.button("🚀 Predict Churn"):
    try:
        input_df = pd.DataFrame({
            "Contract": [contract],
            "tenure": [tenure],
            "InternetService": [internet],
            "MonthlyCharges": [monthly_charges]
        })
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer is likely to STAY")
    except Exception as e:
        st.error(f"Error: {e}")




