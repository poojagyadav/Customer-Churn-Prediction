import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- 1. CONFIG (Must be the VERY FIRST command) ----------------
st.set_page_config(
    page_title="Customer Churn System",
    page_icon="📊",
    layout="wide"
)

# Global Style Settings
sns.set_style("whitegrid")
# DO NOT use plt.rcParams.update({'figure.autolayout': True}) as it breaks manual alignment

# ---------------- 2. APP TITLE ----------------
st.title("📊 Customer Churn Prediction System")
st.divider()

# ---------------- 3. LOAD DATA ----------------
@st.cache_data
def load_data():
    try:
        # Replace with your actual file path
        df = pd.read_csv("data/Customer_Churn.csv")
        return df
    except:
        # Dummy data for UI testing if file is missing
        return pd.DataFrame({
            "Contract": ["Month-to-month", "One year", "Two year"] * 20,
            "InternetService": ["Fiber optic", "DSL", "No"] * 20,
            "tenure": [1, 24, 72, 12, 5, 40] * 10,
            "MonthlyCharges": [20.5, 70.0, 110.0, 50.0, 85.0, 25.0] * 10
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

# ---------------- 4. ALIGNMENT ENGINE (THE FIX) ----------------
FIG_SIZE = (6, 4) # Fixed Aspect Ratio

def apply_fixed_layout(fig, ax, title):
    """
    FORCES exact pixel-perfect margins for every plot.
    This prevents the 'shifting' effect between different plot types.
    """
    ax.set_title(title, fontweight='bold', fontsize=13, pad=15)
    
    # Force identical margins: 
    # Left 15% for labels, Top 15% for title, Bottom 20% for X-axis labels
    fig.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.2)
    
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def plot_pie(column, data_frame):
    data_ = data_frame[column].value_counts()
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # Wedgeprops adds a thin border to make colors pop
    ax.pie(
        data_.values,
        labels=data_.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("Set2"),
        textprops={'fontsize': 10}
    )
    ax.axis('equal') # Standardizes circular shape
    apply_fixed_layout(fig, ax, f"Distribution: {column}")

def plot_count(column, data_frame):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.countplot(data=data_frame, x=column, palette="Set2", ax=ax)
    ax.set_xlabel(column, fontsize=10)
    ax.set_ylabel("Quantity", fontsize=10)
    ax.tick_params(labelsize=9)
    apply_fixed_layout(fig, ax, f"Count: {column}")

def plot_hist(column, data_frame):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.histplot(data_frame[column], bins=20, color="skyblue", kde=True, ax=ax)
    ax.set_xlabel(column, fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.tick_params(labelsize=9)
    apply_fixed_layout(fig, ax, f"Frequency: {column}")

def plot_box(column, data_frame):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(data=data_frame, x=column, color="orange", ax=ax)
    ax.set_xlabel(column, fontsize=10)
    ax.tick_params(labelsize=9)
    apply_fixed_layout(fig, ax, f"Spread: {column}")

# ---------------- 5. EDA SECTION ----------------
st.header("📊 Exploratory Data Analysis")

# Use st.columns to create the layout grid
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    plot_pie("Contract", df)
with row1_col2:
    plot_count("InternetService", df)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    plot_hist("tenure", df)
with row2_col2:
    plot_box("MonthlyCharges", df)

st.divider()

# ---------------- 6. PREDICTION SECTION ----------------
st.header("🔮 Customer Churn Prediction")

if model is None:
    st.warning("Model files not found. Using UI preview mode.")

col_inp1, col_inp2 = st.columns(2)

with col_inp1:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col_inp2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    monthly_charges = st.number_input("Monthly Charges", 0.0, value=50.0)

if st.button("Predict Churn"):
    if model is not None:
        try:
            # Preprocessing
            input_df = pd.DataFrame({
                "Contract": [contract],
                "tenure": [tenure],
                "InternetService": [internet],
                "MonthlyCharges": [monthly_charges]
            })
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=feature_names, fill_value=0)
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)[0]

            if prediction == 1:
                st.error("⚠️ Customer is likely to CHURN")
            else:
                st.success("Customer is likely to STAY")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        # Mock logic if model missing
        if contract == "Month-to-month" and monthly_charges > 70:
            st.error("⚠️Customer likely to CHURN")
        else:
            st.success("Customer likely to STAY")