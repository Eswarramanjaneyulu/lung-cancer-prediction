import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

print("\n===== DATASET INFORMATION =====")
print(df.info())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== TARGET DISTRIBUTION =====")
print(df["LUNG_CANCER"].value_counts())

# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="LUNG_CANCER", data=df)
plt.title("Lung Cancer Distribution")
plt.xlabel("Lung Cancer (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.show()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["AGE"], bins=10, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.show()