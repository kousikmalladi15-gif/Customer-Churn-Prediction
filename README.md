# Customer Churn Prediction

A churn prediction system built on the IBM Telco customer dataset, with a twist most churn projects skip: predictions are weighted by customer lifetime value, not just probability of leaving.

## Why this project exists

Most churn projects you see online stop at "will this customer leave, yes or no." That framing is fine for a classroom exercise but it is not how a business actually prioritizes retention spend. Losing a customer who pays 30 dollars a month is not the same problem as losing one who pays 80 dollars a month, even if both have identical churn probability. This project tries to close that gap by combining churn probability with customer lifetime value (CLV) so the output ranks customers by expected revenue at risk, not just likelihood of leaving.

## Dataset

IBM Telco Customer Churn dataset, around [PLACEHOLDER] rows and [PLACEHOLDER] features covering tenure, contract type, monthly charges, internet and phone service add ons, and payment method. Target variable is binary churn (yes or no).

## Approach

1. Exploratory analysis to understand which features actually separate churners from non churners (tenure and contract type turned out to matter far more than most demographic columns).
2. Preprocessing: encoding categorical variables, handling the class imbalance in the target, scaling where needed.
3. Model: XGBoost, tuned with grid search. Chosen over simpler baselines like logistic regression because it handled the mixed categorical and numeric feature set better without heavy manual feature engineering.
4. CLV scoring layer on top of the raw churn probability, so the final ranking reflects "probability of churn multiplied by what this customer is worth," not churn probability alone.
5. Threshold and scoring choices were validated against business reasoning, not just statistical metrics. A model that is 90 percent accurate but flags low value customers first is not actually useful to a retention team.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.81 |
| Precision | 0.52 |
| Recall | 0.83 |
| F1 score | 0.64 |


Numbers above are on a held out test split of 25% percent. Recall was treated as more important than raw accuracy here, since missing an actual churner costs the business more than a false alarm, leading to a smaller value of precision. 

## Tech stack

Python, pandas, scikit learn, XGBoost, Flask for the serving layer, deployed on Hugging Face Spaces. Frontend is a simple Flask templated UI, nothing fancy, built to demonstrate the model rather than win design awards.

## Project structure
