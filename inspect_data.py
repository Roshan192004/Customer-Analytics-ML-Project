import pandas as pd
import numpy as np

# Load the dataset
try:
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 4. Basic Data Inspection
print("\n--- Shape ---")
print(df.shape)

print("\n--- Columns ---")
print(df.columns)

print("\n--- Data Info ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Basic Statistics ---")
print(df.describe())

# 5. Target Variable
print("\n--- Target Variable Distribution ---")
print(df["Churn"].value_counts())

# 7. TotalCharges issue check
print("\n--- TotalCharges Data Type ---")
print(df["TotalCharges"].dtype)
print("First 5 values of TotalCharges:")
print(df["TotalCharges"].head())
