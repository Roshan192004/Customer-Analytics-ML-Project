import pandas as pd
import numpy as np

# Load the dataset
try:
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 1: Fix TotalCharges (IMPORTANT BUG)
# It is stored as string and contains empty values " "
print("Fixing TotalCharges...")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values again
print("\nMissing values after conversion:")
print(df.isnull().sum())

# Handle missing values by dropping them
df = df.dropna()
print(f"Missing values after dropping: {df.isnull().sum().sum()}")
print(f"New shape: {df.shape}")

# Step 2: Convert Target Variable
print("\nConverting Churn to numeric...")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Step 3: Drop Unnecessary Columns
print("Dropping customerID column...")
df = df.drop("customerID", axis=1)

print(f"Final columns: {df.columns.tolist()}")
