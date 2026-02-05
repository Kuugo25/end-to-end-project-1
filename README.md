# Customer Churn Prediction (Retail Transactions)

End-to-end data science project using transactional retail data to build an early-warning customer churn model, with SQL-based feature engineering and interpretable machine learning.

## Problem Statement

Customer churn is a key challenge for retail businesses, as retaining existing customers is often more cost-effective than acquiring new ones.  
The goal of this project is to build an early-warning churn prediction model that identifies customers at risk of becoming inactive, allowing businesses to target retention efforts proactively.

## Dataset

The project uses an online retail transactional dataset containing individual purchase records, including:
- Customer ID
- Invoice ID and Date
- Product Quantities and Prices
- Country Information

The raw data is transactional (multiple rows per customer) and spans multiple purchase dates, making it suitable for customer-level aggregation and churn analysis.

## Churn definition

For this project, churn is defined as a customer having no recorded purchases for more than 90 days relative to the latest transaction date in the dataset. This definition reflects a practical retail scenario where prolonged inactivity is treated as customer attrition.

## Feature engineering (SQL + RFM)

Customer-level features were created using SQL queries on a SQL database. Each customer was represented by the following RFM-style features:

- Frequency: Number of distinct purchase invoices
- Monetary: Total money spent across all transactions
- Recency: Days since last purchase (used for label construction)

SQL was used to aggregate transaction-level data into a single customer-level table, ensuring a clear separation between data preparation and modelling.

## Modelling Approach

### Baseline model (with recency)
A baseline logistic regression model was trained using frequency, monetary value, and recency. As expected, this model achieved near-perfect performance due to the churn label being directly defined using inactivity duration. This step served as a validation of the data pipeline and churn definition.

### Early-warning model (without recency)
To avoid label leakage and evaluate true predictive capability, recency was excluded from the feature set. A logistic regression model was trained using frequency and monetary value only, representing behavioural signals available before prolonged inactivity occurs. This model achieved moderate recall for churners, realistically reflecting the difficulty of early churn prediction.

## Decision Threshold Tuning

The default 0.5 classification threshold was adjusted to prioritize recall of churners. Lowering the threshold to approximately 0.42 increased churn recall to around 65%, at the cost of reduced precision. This trade-off is appropriate for low-cost retention interventions, where missing at-risk customers is more costly than contacting some customers who would not churn.

## Model Comparison

A random forest model was also evaluated using the same behavioural geatures. Despite its ability to capture non-linear relationships, it did not improve churn recall and relied heavily on monetary value alone. Given its comparable or better performance, interpretability, and stability, logistic regression was selected as the final model.

## Results Summary

- Final model: Logistic Regression (frequency + monetary)
- Optimised for: Recall of churned customers
- Approximately 65% of churners identified in advance
- Trade-off: Increased false positives (misidentified as churners), acceptable for early-warning use cases.

## Limitations and Future Work

- The model relies on limited behavioural features, incorporating temporal patterns or product-level behaviour may improve performance.
- The chrun definition is rule-based and could be refined using survival analysis or time-to-event modelling.
- Future work could include deploying the model as an API for real-time predictions.
