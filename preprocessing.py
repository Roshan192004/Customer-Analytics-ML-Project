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

# Step 4: Handle Categorical Data
print("\nChecking categorical columns...")
cat_cols = df.select_dtypes(include=["object"]).columns
print(f"Categorical columns: {cat_cols.tolist()}")

print("Applying One-Hot Encoding...")
df = pd.get_dummies(df, drop_first=True)

print(f"New shape after encoding: {df.shape}")
print(f"First 5 columns of encoded data: {df.columns[:5].tolist()}")

# Step 5: Feature Scaling
from sklearn.preprocessing import StandardScaler

print("\nApplying feature scaling...")
scaler = StandardScaler()

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Keep track of column names as scaling returns a numpy array
X_scaled = scaler.fit_transform(X)
print("Scaling complete.")
print(f"Shape of scaled features: {X_scaled.shape}")

# Step 6: Train-Test Split
from sklearn.model_selection import train_test_split

print("\nPerforming train-test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Train set shape: {X_train.shape}")
print(f"Test set shape: {y_test.shape}")

# Step 7: Save Clean Data (Optional but Pro 🔥)
print("\nSaving cleaned data to data/cleaned_data.csv...")
df.to_csv("data/cleaned_data.csv", index=False)
print("Data saved successfully!")

print("\n--- Phase 2 Complete ---")
