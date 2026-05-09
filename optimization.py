import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA

def run_optimization_phase():
    print("="*50)
    print("PHASE 9: MODEL EVALUATION & OPTIMIZATION")
    print("="*50)

    # 1. Load Data
    try:
        df = pd.read_csv('data/cleaned_data.csv')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Create visualizations directory if not exists
    os.makedirs('visualizations', exist_ok=True)

    # ---------------------------------------------------------
    # 9.1 Cross-Validation (Classification - Churn)
    # ---------------------------------------------------------
    print("\n" + "*"*40)
    print("9.1 Cross-Validation (Churn Prediction)")
    print("*"*40)

    X_class = df.drop('Churn', axis=1)
    y_class = df['Churn']

    # Scale data 
    scaler_c = StandardScaler()
    X_class_scaled = scaler_c.fit_transform(X_class)

    rf_class = RandomForestClassifier(random_state=42, n_estimators=100)
    
    print("Performing 5-fold cross-validation for Random Forest Classifier...")
    cv_scores_class = cross_val_score(rf_class, X_class_scaled, y_class, cv=5)
    
    print(f"CV Scores: {cv_scores_class}")
    print(f"Mean Accuracy: {cv_scores_class.mean():.4f} (+/- {cv_scores_class.std() * 2:.4f})")

    # ---------------------------------------------------------
    # 9.2 Cross-Validation (Regression - Spending)
    # ---------------------------------------------------------
    print("\n" + "*"*40)
    print("9.2 Cross-Validation (Spending Prediction)")
    print("*"*40)

    X_reg = df.drop('TotalCharges', axis=1)
    y_reg = df['TotalCharges']

    # Scale data
    scaler_r = StandardScaler()
    X_reg_scaled = scaler_r.fit_transform(X_reg)

    rf_reg = RandomForestRegressor(random_state=42, n_estimators=100)
    
    print("Performing 5-fold cross-validation for Random Forest Regressor...")
    # Using negative MSE for CV scores (standard in sklearn)
    cv_scores_reg = cross_val_score(rf_reg, X_reg_scaled, y_reg, cv=5, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_scores_reg)
    
    print(f"CV RMSE Scores: {cv_rmse_scores}")
    print(f"Mean RMSE: {cv_rmse_scores.mean():.2f} (+/- {cv_rmse_scores.std() * 2:.2f})")

    # ---------------------------------------------------------
    # 9.3 Hyperparameter Tuning (GridSearchCV)
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("9.3 Hyperparameter Tuning (GridSearchCV)")
    print("="*40)

    # Tuning Random Forest Classifier
    print("\nTuning Random Forest Classifier for Churn...")
    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    grid_rf_class = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)
    grid_rf_class.fit(X_class_scaled, y_class)

    print(f"Best Parameters (Classification): {grid_rf_class.best_params_}")
    print(f"Best Score (Accuracy): {grid_rf_class.best_score_:.4f}")

    # Tuning Random Forest Regressor
    print("\nTuning Random Forest Regressor for Spending...")
    grid_rf_reg = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_rf_reg.fit(X_reg_scaled, y_reg)

    print(f"Best Parameters (Regression): {grid_rf_reg.best_params_}")
    best_rmse = np.sqrt(-grid_rf_reg.best_score_)
    print(f"Best Score (RMSE): {best_rmse:.2f}")

    # Save the best models for next steps
    best_rf_class = grid_rf_class.best_estimator_
    best_rf_reg = grid_rf_reg.best_estimator_

    print("\n--- Hyperparameter Tuning Step Complete ---")

    # ---------------------------------------------------------
    # 9.4 Feature Importance
    # ---------------------------------------------------------
    print("\n" + "*"*40)
    print("9.4 Feature Importance Analysis")
    print("*"*40)

    # Classification Feature Importance
    importances = best_rf_class.feature_importances_
    features = X_class.columns
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("\nTop 10 Features for Churn Prediction:")
    print(feature_importance_df.head(10).to_string(index=False))

    # Visualize Feature Importance
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(15), x='Importance', y='Feature', palette='viridis')
    plt.title('Top 15 Features for Churn Prediction (Random Forest)')
    plt.tight_layout()
    
    viz_path = 'visualizations/feature_importance.png'
    plt.savefig(viz_path)
    print(f"\nFeature importance visualization saved to {viz_path}.")
    plt.close()

    print("\n--- Feature Importance Step Complete ---")

    # ---------------------------------------------------------
    # 9.5 PCA (Principal Component Analysis)
    # ---------------------------------------------------------
    print("\n" + "*"*40)
    print("9.5 PCA (Dimensionality Reduction)")
    print("*"*40)

    # We'll use the scaled classification data
    pca = PCA()
    X_pca = pca.fit_transform(X_class_scaled)

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    print("\nCumulative Explained Variance by Component:")
    for i, var in enumerate(cumulative_variance[:10]):
        print(f"Component {i+1}: {var:.4f}")

    # Visualize Explained Variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance (Churn Features)')
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Variance')
    plt.legend()
    plt.grid(True)
    
    viz_path = 'visualizations/pca_variance.png'
    plt.savefig(viz_path)
    print(f"\nPCA variance visualization saved to {viz_path}.")
    plt.close()

    # Find number of components for 95% variance
    n_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nNumber of components explaining 95% variance: {n_95}")

    # Run RF on PCA-reduced data
    pca_final = PCA(n_components=n_95)
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        pca_final.fit_transform(X_class_scaled), y_class, test_size=0.2, random_state=42
    )

    rf_pca = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_pca.fit(X_train_pca, y_train_pca)
    pca_preds = rf_pca.predict(X_test_pca)

    print(f"\nAccuracy with {n_95} PCA Components: {accuracy_score(y_test_pca, pca_preds):.4f}")

    print("\n--- PCA Step Complete ---")
    print("\n" + "="*50)
    print("PHASE 9 COMPLETE")
    print("="*50)

    print("\n--- Cross-Validation Step Complete ---")

if __name__ == "__main__":
    run_optimization_phase()
