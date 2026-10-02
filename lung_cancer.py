import pandas as pd

# Load dataset
df = pd.read_csv("data/lung_cancer.csv")

print("\n===== ORIGINAL DATA =====")
print(df.head())

# Remove duplicate rows
df = df.drop_duplicates()

# Convert GENDER into numbers
df["GENDER"] = df["GENDER"].map({
    "M": 1,
    "F": 0
})

# Convert target into numbers
df["LUNG_CANCER"] = df["LUNG_CANCER"].map({
    "YES": 1,
    "NO": 0
})

print("\n===== AFTER PREPROCESSING =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== TARGET VALUES =====")
print(df["LUNG_CANCER"].value_counts())