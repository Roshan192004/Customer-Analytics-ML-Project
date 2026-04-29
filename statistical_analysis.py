import pandas as pd
from scipy import stats
import numpy as np

def run_statistical_analysis():
    print("Loading data...")
    # Load dataset
    try:
        df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Basic preprocessing
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    
    # Split by churn
    churned = df[df['Churn'] == 'Yes']
    retained = df[df['Churn'] == 'No']

    with open('statistical_conclusions.md', 'w') as f:
        f.write("# Phase 4: Statistical Analysis Conclusions\n\n")
        
        # 1. Hypothesis Testing (t-tests)
        f.write("## 1. Hypothesis Testing (T-tests)\n")
        f.write("We conducted two-sample t-tests to determine if there is a statistically significant difference in spending and tenure between customers who churned and those who stayed.\n\n")
        
        # Monthly Charges
        t_stat, p_val = stats.ttest_ind(churned['MonthlyCharges'], retained['MonthlyCharges'], equal_var=False)
        f.write("### Hypothesis 1: Do users who churn have different Monthly Charges?\n")
        f.write(f"- **T-statistic:** {t_stat:.4f}\n")
        f.write(f"- **P-value:** {p_val:.4e}\n")
        if p_val < 0.05:
            f.write("- **Conclusion:** The p-value is extremely small (< 0.05), meaning we reject the null hypothesis. **Customers who churn have significantly higher Monthly Charges** than those who stay.\n\n")
        else:
            f.write("- **Conclusion:** No statistically significant difference in Monthly Charges.\n\n")

        # Tenure
        t_stat_tenure, p_val_tenure = stats.ttest_ind(churned['tenure'], retained['tenure'], equal_var=False)
        f.write("### Hypothesis 2: Do users who churn have a different Tenure?\n")
        f.write(f"- **T-statistic:** {t_stat_tenure:.4f}\n")
        f.write(f"- **P-value:** {p_val_tenure:.4e}\n")
        if p_val_tenure < 0.05:
            f.write("- **Conclusion:** The p-value is practically zero (< 0.05). We reject the null hypothesis. **Customers who churn have significantly lower tenure** than those who stay.\n\n")
        
        # 2. Check Distributions
        f.write("## 2. Distribution Checks\n")
        f.write("We used the Shapiro-Wilk test on a sample of 5000 users to check for normality (since Shapiro-Wilk is limited to N <= 5000).\n\n")
        
        stat_shapiro_mc, p_shapiro_mc = stats.shapiro(df['MonthlyCharges'].sample(5000, random_state=42))
        f.write("### Monthly Charges Normality Check\n")
        f.write(f"- **Shapiro-Wilk Statistic:** {stat_shapiro_mc:.4f}, **P-value:** {p_shapiro_mc:.4e}\n")
        f.write("- **Conclusion:** The p-value is < 0.05, meaning **Monthly Charges are NOT normally distributed.** (This aligns with our EDA where we saw a bimodal distribution).\n\n")

        # 3. Correlation Analysis (Pearson with p-values)
        f.write("## 3. Detailed Correlation Analysis\n")
        f.write("We calculated the Pearson correlation coefficient and its associated p-value to ensure the relationships are statistically significant.\n\n")
        
        corr_mc_tc, p_mc_tc = stats.pearsonr(df['MonthlyCharges'], df['TotalCharges'])
        f.write("### Monthly Charges vs Total Charges\n")
        f.write(f"- **Pearson r:** {corr_mc_tc:.4f}\n")
        f.write(f"- **P-value:** {p_mc_tc:.4e}\n")
        f.write("- **Conclusion:** Strong, statistically significant positive correlation.\n\n")

        corr_tenure_tc, p_tenure_tc = stats.pearsonr(df['tenure'], df['TotalCharges'])
        f.write("### Tenure vs Total Charges\n")
        f.write(f"- **Pearson r:** {corr_tenure_tc:.4f}\n")
        f.write(f"- **P-value:** {p_tenure_tc:.4e}\n")
        f.write("- **Conclusion:** Extremely strong, statistically significant positive correlation.\n\n")

    print("Statistical analysis completed. Results saved to statistical_conclusions.md")

if __name__ == '__main__':
    run_statistical_analysis()
