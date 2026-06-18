# Customer Churn Prediction System

## Overview

The Customer Churn Prediction System is a Machine Learning and Streamlit-based application designed to predict whether a telecom customer is likely to churn (leave the service) or stay with the company.

The project uses customer subscription information such as tenure, monthly charges, contract type, and internet service details to make predictions using trained machine learning models.

---

## Features

* Customer churn prediction
* Interactive Streamlit dashboard
* Real-time prediction
* Data preprocessing and feature engineering
* Model evaluation and comparison
* Visual analytics and insights
* Pickle-based model deployment

---

## Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── telecom_customer_churn.csv
│
├── models/
│   ├── decision.pkl
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── metrics.pkl
│
├── notebooks/
│   ├── random.ipynb
│   ├── svm.ipynb
│   └── XGboost.ipynb
│
├── main.py
├── requirements.txt
├── README.md
├── pyproject.toml
└── .gitignore
```

---

## System Architecture

```text
Customer Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
 ┌───────────────┐
 │ Decision Tree │
 ├───────────────┤
 │ Random Forest │
 ├───────────────┤
 │ SVM           │
 ├───────────────┤
 │ XGBoost       │
 └───────────────┘
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
        │
        ▼
Pickle Serialization
        │
        ▼
Streamlit Deployment
        │
        ▼
Customer Churn Prediction
```

---

## Technologies Used

### Programming Language

* Python 3.x

### Libraries

* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib
* Seaborn
* Pickle

---

## Machine Learning Pipeline

### Data Preprocessing

* Handling missing values
* Encoding categorical features
* Feature scaling using StandardScaler
* One-Hot Encoding using `pd.get_dummies()`

### Feature Engineering

Additional features were created to improve model performance.

Example:

```python
X["ChargePerMonth"] = (
    X["MonthlyCharges"] /
    (X["tenure"] + 1)
)
```

### Model Training

The following algorithms were explored:

1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. XGBoost Classifier

### Model Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project folder:

```bash
cd Customer-Churn-Prediction
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit application:

```bash
streamlit run main.py
```

The application will open automatically in your browser.

---

## Input Features

The model uses features such as:

* Tenure
* Monthly Charges
* Contract Type
* Internet Service

---

## Prediction Output

### Churn Customer

```text
⚠️ Result: The customer is likely to CHURN
```

### Retained Customer

```text
✅ Result: The customer is likely to STAY
```

---

## Model Files

| File          | Description                 |
| ------------- | --------------------------- | 
| svm_model.pkl | Trained SVM Model           |
| scaler.pkl    | Feature Scaling Object      |
| features.pkl  | Feature Names               |
| metrics.pkl   | Evaluation Metrics          |

---

## Future Enhancements

* Hyperparameter tuning
* SMOTE for class balancing
* Model explainability using SHAP
* Cloud deployment
* Customer segmentation
* Automated retraining pipeline

---

## Author

Pooja Yadav

## License

This project is developed for educational and academic purposes.
