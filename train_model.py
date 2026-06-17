import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data/Customer_Churn.csv.csv")

# Keep only 4 features
X = df[
    [
        "Contract",
        "tenure",
        "InternetService",
        "MonthlyCharges"
    ]
]

# Target
y = df["Churn"].map({
    "Yes":1,
    "No":0
})

# Encode categorical features
X = pd.get_dummies(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
with open("models/decision.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature names
with open("models/features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model trained successfully")

