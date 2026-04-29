# Phase 4: Statistical Analysis Conclusions

## 1. Hypothesis Testing (T-tests)
We conducted two-sample t-tests to determine if there is a statistically significant difference in spending and tenure between customers who churned and those who stayed.

### Hypothesis 1: Do users who churn have different Monthly Charges?
- **T-statistic:** 18.3409
- **P-value:** 2.6574e-72
- **Conclusion:** The p-value is extremely small (< 0.05), meaning we reject the null hypothesis. **Customers who churn have significantly higher Monthly Charges** than those who stay.

### Hypothesis 2: Do users who churn have a different Tenure?
- **T-statistic:** -34.9719
- **P-value:** 2.3471e-234
- **Conclusion:** The p-value is practically zero (< 0.05). We reject the null hypothesis. **Customers who churn have significantly lower tenure** than those who stay.

## 2. Distribution Checks
We used the Shapiro-Wilk test on a sample of 5000 users to check for normality (since Shapiro-Wilk is limited to N <= 5000).

### Monthly Charges Normality Check
- **Shapiro-Wilk Statistic:** 0.9221, **P-value:** 4.3042e-45
- **Conclusion:** The p-value is < 0.05, meaning **Monthly Charges are NOT normally distributed.** (This aligns with our EDA where we saw a bimodal distribution).

## 3. Detailed Correlation Analysis
We calculated the Pearson correlation coefficient and its associated p-value to ensure the relationships are statistically significant.

### Monthly Charges vs Total Charges
- **Pearson r:** 0.6511
- **P-value:** 0.0000e+00
- **Conclusion:** Strong, statistically significant positive correlation.

### Tenure vs Total Charges
- **Pearson r:** 0.8259
- **P-value:** 0.0000e+00
- **Conclusion:** Extremely strong, statistically significant positive correlation.

