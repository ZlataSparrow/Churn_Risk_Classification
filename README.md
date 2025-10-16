# 🌀 Predicting Customer Churn Using Behavioral Data: ML Models, SHAP Insights, and Retention Simulations (In Progress)

## Behavioral modeling of user engagement, churn risk, and retention strategies

This project explores and models user churn using a customer behavior dataset. By leveraging both interpretable and powerful machine learning models, the goal is to predict the likelihood of churn and uncover the key behavioral signals driving customer retention.

⸻

### Strategic Churn Questions:

1. What behaviors signal a user is about to churn?
2. Can we accurately predict churn before it happens?
3. Are there user segments that are more prone to churn?
4. When in the customer lifecycle is churn most likely?


### Data

This project is based on the Customer Log Dataset published on IEEE DataPort, which captures detailed user interaction logs from a digital music streaming service. The raw dataset contains over 26 million records across 18 fields, recording user behavior at a fine-grained level.

* Source: IEEE DataPort – Customer Churn Dataset (https://ieee-dataport.org/documents/customer-churn-dataset#files)
* Size: 12.5 GB (JSON format)
* Records: 26,259,199 user activity logs
* Columns: 18 total (12 string, 6 numeric)
* Granularity: One row per user interaction (e.g., play song, login, logout, thumbs up)

### Methods
* Churn Labeling — Identified cancellations from activity logs and aligned with final user sessions to create a realistic churn flag.
* Feature Engineering — Built user-level metrics from raw JSON logs, including sessions, engagement actions, registration age, and activity frequency.
* Modeling — Applied Logistic Regression (baseline), Random Forest, and XGBoost to predict churn based on behavioral features.
* Explainability — Used SHAP to interpret model outputs globally and per user, enabling transparent and actionable insights.
* Simulation — Ran counterfactual experiments (e.g., +5 active days) to estimate the causal effect of engagement interventions on churn probability.
* Validation — Evaluated models using ROC-AUC, precision, recall, and test-set performance, including threshold-based targeting for high-risk users.

### Key Outcomes
1. Low engagement behaviors (e.g., few sessions, low playlist activity) strongly predict churn across all models.
2. User interactions like thumbs_up and add_to_playlist are top signals of retention, confirmed by both logistic regression coefficients and SHAP values.
3. Free-tier users show significantly higher churn risk, even after controlling for activity — indicating a pricing-access dynamic.
4. XGBoost achieved the best performance (ROC-AUC ~0.89), outperforming logistic regression while maintaining stable feature rankings.
5. Counterfactual simulations show that increasing active days by just 5 can lower churn probability by up to 40 percentage points for high-risk users.

### What This Shows

**Behavioral data can powerfully predict churn — with interpretable models confirming intuitive insights and complex models refining them**
**Explainable AI techniques (SHAP) help bridge the gap between data science and business decisions**
**Counterfactual simulations offer a lightweight alternative to A/B testing — enabling proactive retention strategies without full deployments**
**The project demonstrates end-to-end ML fluency, from raw logs to interpretable models, simulations, and actionable recommendations**

⸻

### Structure:


```
 EDICTING_CUSTOMER_CHURN/
├── data/
│   ├── User_Records.json               # https://ieee-dataport.org/documents/customer-churn-dataset#files
│ 
│
├── notebooks/
│   ├── 01_Data_Prep.ipynb                     
│   ├── 02_EDA.ipynb                           
│   ├── 03_Logistic_Regression_Classifier.ipynb
│   ├── 04_Random_Forest_&_XGBoost.ipynb
│
└── README.md                                  # Project documentation
```
