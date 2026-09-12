# Bank Customer Churn Prediction

A simple, from-scratch churn prediction project — built to practice the full ML lifecycle end-to-end: EDA, preprocessing, model comparison, and evaluation with a business-first lens on metrics.

## Problem Statement

Banks lose significant revenue when customers leave (churn). This project predicts which customers are likely to churn, so a business could intervene (retention offers, outreach) before they actually leave.

## Dataset

[Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers) — 10,000 customers, 14 columns including credit score, geography, gender, age, tenure, balance, number of products, activity status, and estimated salary. Target column: `Exited` (1 = churned, 0 = stayed).

No missing values, no duplicate customers.

## Exploratory Data Analysis

Key findings, each checked against sample size before trusting the pattern:

- **Geography**: Germany churns at ~32%, roughly double France (~16%) and Spain (~17%). Verified this isn't a small-sample artifact — Germany and Spain have nearly identical customer counts (~2,500 each), yet very different churn rates.
- **Gender**: Female customers churn more (~25%) than male customers (~16%).
- **Age**: Churn rises sharply through middle age, peaking around 45–65 (~50% churn), then drops for older customers. Raw per-year age plotting was noisy due to very few customers at extreme ages; binning into ranges revealed the real trend.
- **Number of Products**: A U-shaped pattern — customers with 2 products churn least (~8%), while 1 product (~28%) and 3+ products churn more. The 3–4 product groups are based on small samples (266 and 60 customers respectively) and should be treated as suggestive, not conclusive.
- **Active Membership**: Inactive members churn roughly twice as often (~27%) as active members (~14%).
- **Balance**: Zero-balance customers churn least (~14%); churn is fairly flat (~20–26%) across mid-range balances. Apparent spikes at very low and very high balance ranges are based on small samples (75 and 33 customers) and are not reliable.

## Preprocessing

- Dropped non-predictive identifier columns (`RowNumber`, `CustomerId`, `Surname`)
- One-hot encoded `Geography` and `Gender` (with `drop_first=True` to avoid redundant columns)
- 80/20 train-test split (`random_state=42` for reproducibility)
- Scaled features with `StandardScaler` for Logistic Regression (fit on train only, applied to test, to avoid data leakage)
- Addressed class imbalance (~80% stayed, ~20% churned) using `class_weight='balanced'` rather than SMOTE, to keep the pipeline simple and interpretable

## Models & Results

| Metric | Logistic Regression | Random Forest (default) | Random Forest (threshold=0.3) |
|---|---|---|---|
| Accuracy | 0.72 | 0.87 | — |
| Precision (churn) | 0.38 | 0.79 | 0.56 |
| Recall (churn) | 0.71 | 0.46 | 0.66 |

**Interpretation**: Logistic Regression casts a wide net, catching more real churners but with more false alarms. Random Forest is more conservative — high confidence when it flags churn, but misses more actual churners at the default threshold. Lowering Random Forest's decision threshold to 0.3 pushes it toward a middle ground.

**Business take**: for churn, missing a real churner (lost customer, lost revenue) is usually costlier than a false alarm (a low-cost retention email to someone who wasn't leaving). This generally favors optimizing for recall over precision — which would favor Logistic Regression or a lowered-threshold Random Forest over the default Random Forest, depending on the actual cost tradeoff a business defines.

## Feature Importance (Random Forest)

Top drivers: **Age**, **Balance**, **Estimated Salary**, **Number of Products**, **Credit Score**.

Age, Balance, and Number of Products align directly with the EDA findings above. Estimated Salary and Credit Score were not explicitly explored during EDA — a gap worth investigating further. Interestingly, Geography and Active Membership — which showed strong separation in EDA bar charts — rank lower in feature importance, likely because their signal overlaps with what Age and Balance already capture.

## What I'd Do With More Time

- Explore Estimated Salary and Credit Score against churn in EDA, since the model flagged them as important but they weren't part of the original analysis
- Try SMOTE as an alternative to class weighting, and compare results
- Hyperparameter tuning (GridSearch/RandomSearch) on Random Forest
- Try XGBoost as a third model for comparison
- Simple Streamlit app for live churn probability lookup

## Tech Stack

Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter (VS Code)