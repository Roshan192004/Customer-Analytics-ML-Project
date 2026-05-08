import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os

def run_recommendation_system():
    print("="*50)
    print("PHASE 7: AI RECOMMENDATION SYSTEM (KNN)")
    print("="*50)

    # 1. Load Cleaned Data
    try:
        df = pd.read_csv('data/cleaned_data.csv')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. Define Service Columns (Products to Recommend)
    # These are the 'Yes' columns for various services
    service_mapping = {
        'PhoneService_Yes': 'Phone Service',
        'MultipleLines_Yes': 'Multiple Lines',
        'OnlineSecurity_Yes': 'Online Security',
        'OnlineBackup_Yes': 'Online Backup',
        'DeviceProtection_Yes': 'Device Protection',
        'TechSupport_Yes': 'Tech Support',
        'StreamingTV_Yes': 'Streaming TV',
        'StreamingMovies_Yes': 'Streaming Movies'
    }
    
    service_cols = list(service_mapping.keys())
    
    # 3. Prepare Features for Similarity
    # Use all features except 'Churn' to find similar customers
    X = df.drop('Churn', axis=1)
    
    # Scale features for KNN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled for similarity analysis.")

    # 4. Build KNN Model
    # We use 6 neighbors (1 is the customer themselves, so we get 5 peers)
    n_neighbors = 6
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(X_scaled)
    print(f"KNN model fitted (k={n_neighbors-1} peers, metric='cosine').")

    # 5. Recommendation Logic
    print("\nGenerating recommendations...")
    
    # Find neighbors for all customers
    distances, indices = knn.kneighbors(X_scaled)
    
    recommendations = []
    
    for i in range(len(df)):
        # Get current customer's services
        current_services = df.iloc[i][service_cols]
        
        # Identify services they DON'T have (value is 0)
        missing_services = [col for col in service_cols if current_services[col] == 0]
        
        if not missing_services:
            recommendations.append("All services already active")
            continue
            
        # Check adoption of missing services among 5 nearest neighbors
        # indices[i][0] is the customer themselves, so we look at indices[i][1:]
        neighbor_indices = indices[i][1:]
        neighbor_data = df.iloc[neighbor_indices][missing_services]
        
        # Calculate frequency of each missing service among neighbors
        service_scores = neighbor_data.mean()
        
        # Sort by score (descending) and get top recommendations
        top_recs = service_scores[service_scores > 0].sort_values(ascending=False)
        
        if top_recs.empty:
            recommendations.append("No common services among neighbors")
        else:
            # Map back to human-readable names
            rec_names = [service_mapping[s] for s in top_recs.index[:3]]
            recommendations.append(", ".join(rec_names))

    # 6. Save Results
    df_results = df.copy()
    df_results['Recommendations'] = recommendations
    
    os.makedirs('data', exist_ok=True)
    output_path = 'data/customer_recommendations.csv'
    df_results.to_csv(output_path, index=False)
    print(f"Recommendations saved to {output_path}.")

    # 7. Show Sample Recommendations
    print("\nSample Recommendations:")
    sample = df_results[['tenure', 'MonthlyCharges', 'Recommendations']].sample(5, random_state=42)
    print(sample.to_string())

    print("\n--- Phase 7 Complete ---")

if __name__ == "__main__":
    run_recommendation_system()
