# Bank-Customer-Churn-Prediction

## Executive Summary & Business Value

Customer acquisition is roughly 5 to 25 times more expensive than customer retention. In retail banking, losing a high-net-worth individual doesn't just impact current account fees; it drains the bank's lending capital and destroys lifetime value (LTV).

This project bridges the gap between predictive analytics and proactive customer success. By deploying a highly tuned Random Forest classifier, this project shifts the retention strategy from a **reactive** approach (trying to save a customer after they initiate an account closure) to a **proactive** one (intervening before the customer makes the decision to leave).

### Model ROI & Financial Impact
Instead of applying blanket retention discounts to the entire customer base which wastes budget on loyal customers, this model enables targeted interventions. 

* **High-Precision Targeting:** By adjusting the model's decision threshold to optimize for a balanced F1-Score, the retention team can confidently allocate expensive perks (e.g., fee waivers, dedicated account managers) strictly to the high-risk segment identified by the algorithm, rather than wasting budget on the ~80% of the customer base that the historical data proves is naturally loyal.
* **Capital Preservation:** Integrating the model's risk probabilities with Power BI allows stakeholders to instantly quantify **"Total Revenue at Risk."** This empowers leadership to prioritize outreach not just by churn probability, but by the highest account balances, actively protecting the bank's liquidity.
* **Strategic Resource Allocation:** Extracting the Random Forest's feature importance metrics revealed that account activity and geographical location are critical drivers of churn. By isolating inactive customers in specific European markets (e.g., Germany) as a top-risk segment, the marketing department can deploy highly specific, localized re-engagement campaigns rather than generic, bank-wide blasts.

### The Final Deliverable
This repository contains the end-to-end pipeline: from raw data ingestion and feature engineering to model hyperparameter tuning. The final output is an exported dataset containing precise churn probability scores, designed specifically to feed directly into a BI tool for daily executive monitoring.

## Methodology

1. **Exploratory Data Analysis:** Investigated the 79/21 class imbalance and visualized the relationship between demographics (Geography, Gender) and financial behavior (Balance, Active Status) with churn.
2. **Feature Engineering:** Created new logical variables, such as `Balance_per_Age` ratios and a `High_Risk_Profile` flag for inactive customers in high-churn regions.
3. **Data Preprocessing:** Utilized `scikit-learn`'s `ColumnTransformer` to apply `StandardScaler` to numerical features and `OneHotEncoder` to categorical features strictly after the train-test split to prevent data leakage.
4. **Model Selection:** Trained and tuned multiple classification algorithms:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest Classifier (Winner)
   - XGBoost
5. **Evaluation:** Evaluated models using F1-Score, Precision, Recall, and AUC-ROC to handle the imbalanced nature of the dataset.

## Key Insights & Actionable Recommendations
*Interpretability is just as important as accuracy. Using EDA and Feature Importances, the model revealed:*

1. **The "Wealth" Flight Risk:** Customers with unusually high account balances are churning at a higher rate than low-balance customers. *Recommendation: Implement targeted premium perk programs or highly competitive interest rate matching for the top 10% of balance holders.*
2. **The Geography Factor:** Customers located in Germany are significantly more likely to exit compared to those in France or Spain. *Recommendation: Investigate regional competitor offerings or localized customer service issues in the German market.*
3. **Age Correlation:** Feature importance highlights Age as a primary driver of account closures, indicating that churn risk fluctuates significantly across different life stages. *Recommendation: Transition away from a one-size-fits-all retention strategy by segmenting the customer base into life-stage profiles, ensuring that proactive retention offers match their immediate financial needs.*
4. **Activity Status:** Inactive members are a strong leading indicator of churn. *Recommendation: Deploy automated re-engagement campaigns before an account officially closes.*

## Interactive Power BI Dashboard: From Predictions to Business Action

While machine learning models are built in Python, business decisions are driven by interactive BI tools. To bridge the gap between predictive analytics and operational strategy, the final predictions and risk probabilities were exported from the Random Forest model and visualized in **Power BI**. 

**The Purpose:** To transform abstract machine learning probabilities (e.g., "0.82 risk") into quantifiable financial metrics (e.g., "$125,000 at risk") and empower the retention team to operationalize the model without needing to read code.

### Dashboard Highlights & Functionality:
* **Dynamic 'Revenue at Risk' (What-If Parameter):** Built a custom DAX numeric range parameter that allows stakeholders to dynamically adjust the model's risk threshold (e.g., sliding from a 50% risk cutoff to an 80% risk cutoff). The dashboard instantly recalculates the total capital at risk, enabling leaders to balance their retention budget against the most critical flight risks.
* **The Scatter Plot:** Engineered custom DAX conditional formatting to visually isolate high-net-worth individuals who also exhibit a high probability of churning. High-balance, high-risk targets automatically highlight in red, giving the VIP banking team an immediate priority matrix.
* **Operational 'Hit List':** A dynamically sorted matrix table that maps the ML predictions back to the original `CustomerID`. This provides customer success managers with a daily, pre-ranked list of exactly who to call, complete with demographic context like geography and active status.

*(Below is a screenshot of the interactive dashboard in action)*
![Power BI Dashboard Screenshot](figures/powerbi_dashboard_screenshot.png)
## Technologies Used
* **Python** (Pandas, NumPy)
* **Machine Learning:** Scikit-Learn, XGBoost
* **Visualization:** Matplotlib, Seaborn, PowerBI
