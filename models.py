import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Classification imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Regression imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_models():
    print("="*50)
    print("PHASE 5: MACHINE LEARNING MODELS")
    print("="*50)

    # Load cleaned data
    try:
        df = pd.read_csv('data/cleaned_data.csv')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading cleaned_data.csv: {e}")
        print("Please ensure you have run the preprocessing script first.")
        return

    # ---------------------------------------------------------
    # 5.1 Classification (Churn)
    # ---------------------------------------------------------
    print("\n" + "*"*40)
    print("5.1 Classification (Churn Prediction)")
    print("*"*40)

    X_class = df.drop('Churn', axis=1)
    y_class = df['Churn']

    # Split data
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, random_state=42
    )

    # Scale data 
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)

    # --- Logistic Regression ---
    print("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_c_scaled, y_train_c)
    lr_preds = lr_model.predict(X_test_c_scaled)

    print(f"Accuracy:  {accuracy_score(y_test_c, lr_preds):.4f}")
    print(f"Precision: {precision_score(y_test_c, lr_preds):.4f}")
    print(f"Recall:    {recall_score(y_test_c, lr_preds):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_c, lr_preds))

    # --- Random Forest Classifier ---
    print("\n--- Random Forest Classifier ---")
    rf_class_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_class_model.fit(X_train_c_scaled, y_train_c)
    rf_class_preds = rf_class_model.predict(X_test_c_scaled)

    print(f"Accuracy:  {accuracy_score(y_test_c, rf_class_preds):.4f}")
    print(f"Precision: {precision_score(y_test_c, rf_class_preds):.4f}")
    print(f"Recall:    {recall_score(y_test_c, rf_class_preds):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_c, rf_class_preds))

    # ---------------------------------------------------------
    # 5.2 Regression (Spending)
    # ---------------------------------------------------------
    print("\n" + "*"*40)
    print("5.2 Regression (Predicting TotalCharges)")
    print("*"*40)

    # We use TotalCharges as the spending target.
    # We include Churn as a feature, as we are analyzing overall customer spending behavior.
    X_reg = df.drop('TotalCharges', axis=1)
    y_reg = df['TotalCharges']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Scale data
    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)

    # --- Linear Regression ---
    print("\n--- Linear Regression ---")
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train_r_scaled, y_train_r)
    lin_reg_preds = lin_reg_model.predict(X_test_r_scaled)

    lin_rmse = np.sqrt(mean_squared_error(y_test_r, lin_reg_preds))
    lin_r2 = r2_score(y_test_r, lin_reg_preds)
    print(f"RMSE:     {lin_rmse:.2f}")
    print(f"R² Score: {lin_r2:.4f}")

    # --- Random Forest Regressor ---
    print("\n--- Random Forest Regressor ---")
    rf_reg_model = RandomForestRegressor(random_state=42, n_estimators=100)
    rf_reg_model.fit(X_train_r_scaled, y_train_r)
    rf_reg_preds = rf_reg_model.predict(X_test_r_scaled)

    rf_rmse = np.sqrt(mean_squared_error(y_test_r, rf_reg_preds))
    rf_r2 = r2_score(y_test_r, rf_reg_preds)
    print(f"RMSE:     {rf_rmse:.2f}")
    print(f"R² Score: {rf_r2:.4f}")

if __name__ == "__main__":
    run_models()
