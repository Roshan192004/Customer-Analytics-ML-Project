# Phase 3: Exploratory Data Analysis (EDA) Insights 🔥

Based on the statistical analysis and visualizations generated during Phase 3, here are the key insights into customer behavior and churn patterns:

## 1. Summary Statistics & Spending
- **Tenure:** The average customer stays for about 32 months. However, the distribution is highly polarized. There is a massive spike of customers at `tenure = 1` month (new customers), and another spike at `tenure = 72` months (very loyal, long-term customers).
- **Monthly Charges:** Customers pay an average of $64.80 per month. The distribution of monthly charges shows a large concentration of customers at the lower end (~$20/month, likely basic phone service only) and a wide spread between $70-$100/month (likely bundled internet and TV services).
- **Outliers:** Based on the box plots generated, there are no significant statistical outliers in `tenure`, `MonthlyCharges`, or `TotalCharges`.

## 2. Correlation Analysis
- `TotalCharges` is highly positively correlated with `tenure` (as expected: longer tenure = more money spent over time).
- `TotalCharges` also has a strong positive correlation with `MonthlyCharges`.
- `tenure` has a negative correlation with `Churn` (longer tenure means lower likelihood to churn).
- `MonthlyCharges` has a slightly positive correlation with `Churn` (higher monthly bills might contribute to churn).

## 3. Churn vs Categorical Features (The "Why")
The `churn_vs_features.png` plot reveals the most actionable insights:
- **Contract Type (CRITICAL):** Customers on a **Month-to-month** contract have a drastically higher churn rate compared to those on One-year or Two-year contracts. Long-term contracts heavily lock in loyalty.
- **Internet Service:** Customers using **Fiber optic** internet churn at a noticeably higher rate than those using DSL. This could point to issues with fiber optic pricing, reliability, or competitor offers.
- **Tech Support:** Customers **without Tech Support** are significantly more likely to churn. Proactive support seems to be a strong retention tool.
- **Payment Method:** Customers using **Electronic checks** have the highest churn rate among all payment methods.

## 🌟 Recommendations for Business Action:
1. **Target new customers (Month 1):** The highest drop-off is in the first month. An onboarding or discount program for month-to-month users could retain them longer.
2. **Upsell Long-Term Contracts:** Incentivize users to switch from month-to-month to 1-year contracts (e.g., offer a free month of service).
3. **Investigate Fiber Optic:** Look into the quality, pricing, or customer service specifically related to the Fiber Optic offering.
4. **Bundle Tech Support:** Offer tech support at a discount or bundled to decrease churn.

---
*Note: All corresponding graphs are saved in the `visualizations/` folder.*
