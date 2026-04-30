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

if __name__ == "__main__":
    run_segmentation()
