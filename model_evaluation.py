import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

# Load dataset
df = pd.read_csv("data/lung_cancer.csv")

# Remove duplicates
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

# Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Prediction probabilities
y_probability = model.predict_proba(X_test)[:, 1]

# Accuracy
print("\n===== ACCURACY =====")
print(accuracy_score(y_test, y_pred))

# Classification report
print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["NO", "YES"]
    )
)

# Confusion matrix
print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# ROC-AUC
auc = roc_auc_score(y_test, y_probability)

print("\n===== ROC-AUC =====")
print(auc)

# Confusion matrix graph
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NO", "YES"]
).plot()

plt.title("Random Forest Confusion Matrix")
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.legend()
plt.show()