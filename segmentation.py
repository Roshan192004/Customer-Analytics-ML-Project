import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def run_segmentation():
    print("="*50)
    print("PHASE 6: UNSUPERVISED LEARNING (CUSTOMER SEGMENTATION)")
    print("="*50)

    # Load cleaned data
    try:
        df = pd.read_csv('data/cleaned_data.csv')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading cleaned_data.csv: {e}")
        return

    # Select features for segmentation
    # We will use continuous variables for clear customer profiles
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    X_seg = df[features]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)
    print("Features selected and scaled: tenure, MonthlyCharges, TotalCharges.")

    # ---------------------------------------------------------
    # 6.1 Find Optimal Clusters (Elbow Method)
    # ---------------------------------------------------------
    print("\nCalculating WCSS for Elbow Method...")
    wcss = []
    max_clusters = 10
    
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
    # Plot the Elbow Graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    
    # Save the plot
    os.makedirs('visualizations', exist_ok=True)
    elbow_path = 'visualizations/elbow_method.png'
    plt.savefig(elbow_path)
    print(f"Elbow curve saved to {elbow_path}.")
    plt.close()

if __name__ == "__main__":
    run_segmentation()
