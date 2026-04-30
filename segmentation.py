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

    # ---------------------------------------------------------
    # 6.2 Apply K-Means (Assuming k=3 based on elbow curve)
    # ---------------------------------------------------------
    optimal_k = 3
    print(f"\nApplying K-Means with k={optimal_k}...")
    kmeans_optimal = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans_optimal.fit_predict(X_scaled)
    
    # Add clusters back to the original subset for interpretation
    df_segmented = X_seg.copy()
    df_segmented['Cluster'] = cluster_labels
    
    print("\nCluster Distribution:")
    print(df_segmented['Cluster'].value_counts().sort_index())
    
    print("\nCluster Profiles (Mean values):")
    cluster_profiles = df_segmented.groupby('Cluster').mean()
    print(cluster_profiles)

    # ---------------------------------------------------------
    # 6.3 Visualize Clusters (3D Scatter)
    # ---------------------------------------------------------
    print("\nGenerating 3D Cluster Visualization...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    scatter = ax.scatter(
        df_segmented['tenure'], 
        df_segmented['MonthlyCharges'], 
        df_segmented['TotalCharges'], 
        c=df_segmented['Cluster'], 
        cmap='viridis', 
        alpha=0.6,
        s=20
    )
    
    ax.set_title(f'Customer Segments (k={optimal_k})')
    ax.set_xlabel('Tenure (Months)')
    ax.set_ylabel('Monthly Charges ($)')
    ax.set_zlabel('Total Charges ($)')
    
    # Legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    
    # Save the 3D plot
    clusters_path = 'visualizations/cluster_segments_3d.png'
    plt.savefig(clusters_path)
    print(f"Cluster visualization saved to {clusters_path}.")
    plt.close()
    
    # Optional: save the segmented data
    df_segmented.to_csv('data/segmented_customers.csv', index=False)
    print("\nSegmented customer data saved to data/segmented_customers.csv.")
    
    print("\n--- Phase 6 Complete ---")

if __name__ == "__main__":
    run_segmentation()
