import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create visualizations directory
os.makedirs('visualizations', exist_ok=True)

# 1. Load Data
try:
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. Basic Preprocessing for EDA
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# 3. Summary Statistics
print("\n--- Summary Statistics ---")
summary_stats = df.describe()
print(summary_stats)
with open('visualizations/summary_statistics.txt', 'w') as f:
    f.write(summary_stats.to_string())

# Set style for plots
sns.set_theme(style="whitegrid")

# 4. Distribution Plots (Spending & Tenure)
print("\nGenerating Distribution Plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['tenure'], bins=30, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Tenure (Months)')
sns.histplot(df['MonthlyCharges'], bins=30, kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Distribution of Monthly Charges')
sns.histplot(df['TotalCharges'], bins=30, kde=True, ax=axes[2], color='lightgreen')
axes[2].set_title('Distribution of Total Charges')
plt.tight_layout()
plt.savefig('visualizations/spending_distribution.png')
plt.close()

# 5. Correlation Heatmap
print("Generating Correlation Heatmap...")
numeric_df = df.select_dtypes(include=['int64', 'float64'])
# Convert Churn to numeric just for correlation if possible, but let's keep it simple
df_corr = df.copy()
df_corr['Churn'] = df_corr['Churn'].map({'Yes': 1, 'No': 0})
numeric_df_with_churn = df_corr.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(8, 6))
correlation_matrix = numeric_df_with_churn.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png')
plt.close()

# 6. Outlier Detection (Box Plots)
print("Generating Outlier Box Plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(y=df['tenure'], ax=axes[0], color='skyblue')
axes[0].set_title('Boxplot of Tenure')
sns.boxplot(y=df['MonthlyCharges'], ax=axes[1], color='salmon')
axes[1].set_title('Boxplot of Monthly Charges')
sns.boxplot(y=df['TotalCharges'], ax=axes[2], color='lightgreen')
axes[2].set_title('Boxplot of Total Charges')
plt.tight_layout()
plt.savefig('visualizations/outliers_boxplot.png')
plt.close()

# 7. Churn vs Features
print("Generating Churn vs Features Plots...")
categorical_features = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, feature in enumerate(categorical_features):
    sns.countplot(data=df, x=feature, hue='Churn', ax=axes[i], palette='Set2')
    axes[i].set_title(f'Churn by {feature}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('visualizations/churn_vs_features.png')
plt.close()

print("\nEDA completed. Visualizations saved to 'visualizations/' directory.")
