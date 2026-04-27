import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import plotext as plt

# Load Data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Plot Histogram of Tenure
plt.hist(df['tenure'].dropna(), bins=30, color='blue')
plt.title("Distribution of Customer Tenure (Months)")
plt.xlabel("Months")
plt.ylabel("Frequency")

plt.show()

# Show average Monthly Charges by Churn
print("\n" + "-"*40)
print("Average Monthly Charges by Churn:")
churn_charges = df.groupby('Churn')['MonthlyCharges'].mean()
plt.clear_figure()
plt.bar(['No Churn', 'Churn'], [churn_charges['No'], churn_charges['Yes']], color=['green', 'red'])
plt.title("Average Monthly Charges vs Churn")
plt.ylabel("$")
plt.show()
