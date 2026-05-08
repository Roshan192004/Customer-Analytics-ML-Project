import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def run_neural_network_phase():
    print("="*50)
    print("PHASE 8: DEEP LEARNING (NEURAL NETWORKS)")
    print("="*50)

    # 1. Load Data
    try:
        df = pd.read_csv('data/cleaned_data.csv')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Prepare Data
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data split and scaled (80/20 split).")

    # 3. Build and Train MLP (Neural Network)
    print("\nTraining Multi-Layer Perceptron (Neural Network)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), 
        activation='relu', 
        solver='adam', 
        max_iter=500, 
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    mlp_preds = mlp.predict(X_test_scaled)
    print("Neural Network training complete.")

    # 4. Compare with ML Models (LR and RF)
    print("\nEvaluating and comparing models...")
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)

    # Random Forest
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)

    # 5. Calculate Metrics
    models = {
        'Neural Network (MLP)': mlp_preds,
        'Logistic Regression': lr_preds,
        'Random Forest': rf_preds
    }

    results = []
    for name, preds in models.items():
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1-Score': f1_score(y_test, preds)
        })

    df_results = pd.DataFrame(results)
    
    # 6. Display Comparison
    print("\nModel Performance Comparison:")
    print("-" * 65)
    print(df_results.to_string(index=False))
    print("-" * 65)

    # 7. Save Results
    os.makedirs('data', exist_ok=True)
    comparison_path = 'data/model_comparison_results.csv'
    df_results.to_csv(comparison_path, index=False)
    print(f"\nComparison results saved to {comparison_path}.")

    # 8. Visualize Comparison
    plt.figure(figsize=(10, 6))
    df_melted = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model')
    plt.title('Model Performance Comparison (Churn Prediction)')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs('visualizations', exist_ok=True)
    viz_path = 'visualizations/model_comparison_nn.png'
    plt.savefig(viz_path)
    print(f"Comparison visualization saved to {viz_path}.")
    plt.close()

    print("\n--- Phase 8 Complete ---")

if __name__ == "__main__":
    run_neural_network_phase()
