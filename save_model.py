import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/lung_cancer.csv")

# Remove duplicate rows
df = df.drop_duplicates()

# Convert GENDER
df["GENDER"] = df["GENDER"].map({
    "M": 1,
    "F": 0
})

# Convert target
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({
    "YES": 1,
    "NO": 0
})

# Features and target
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/lung_cancer_model.pkl")

# Save feature names
joblib.dump(
    list(X.columns),
    "models/feature_names.pkl"
)

print("\n===== MODEL SAVED SUCCESSFULLY =====")
print("Model: models/lung_cancer_model.pkl")
print("Features: models/feature_names.pkl")