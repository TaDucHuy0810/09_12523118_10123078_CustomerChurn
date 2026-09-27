# Dataset and Data Dictionary

## Source and preparation

- Dataset: IBM Telco Customer Churn sample distributed through [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
- Local file: `data/telco_churn.csv`, 7,043 source rows, semicolon-separated.
- Course archive: `ai-models/data/dataset.zip` contains the same CSV for the notebook workflow.
- The current pipeline retains 7,032 rows after converting `Monthly Charges` and `Total Charges` to numeric and dropping rows with invalid target or charge values.
- Target: `Churn Label` (`No` = 0, `Yes` = 1). In the cleaned dataset: 5,163 `No` and 1,869 `Yes`.
- Model inputs: 20 features listed below. `CustomerID`, `Count`, location fields (`Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`) and target-derived fields (`Churn Value`, `Churn Score`, `Churn Reason`) are excluded.
- Charge values are parsed after trimming whitespace and replacing decimal commas with decimal points. Numeric imputation, scaling and categorical encoding are fit inside the sklearn pipeline after the stratified split.
- License/source note: Kaggle currently displays “Data files © Original Authors”; an open redistribution license was not confirmed. Verify the current terms and obtain permission before redistributing beyond the coursework. This repository does not claim ownership of the raw data.

Numeric ranges below are observed in the cleaned local dataset, not policy or health limits.

## Feature dictionary

| Feature             | Type     | Description                                    | Values or observed range                                                                   |
| ------------------- | -------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `Gender`            | Category | Customer gender                                | `Female`, `Male`                                                                           |
| `Senior Citizen`    | Category | Whether the customer is a senior citizen       | `No`, `Yes`                                                                                |
| `Partner`           | Category | Whether the customer has a partner             | `No`, `Yes`                                                                                |
| `Dependents`        | Category | Whether the customer has dependents            | `No`, `Yes`                                                                                |
| `Tenure Months`     | Number   | Months the customer has had service            | 1 to 72 months                                                                             |
| `Phone Service`     | Category | Whether phone service is active                | `No`, `Yes`                                                                                |
| `Multiple Lines`    | Category | Phone line status                              | `No`, `No phone service`, `Yes`                                                            |
| `Internet Service`  | Category | Internet service type                          | `DSL`, `Fiber optic`, `No`                                                                 |
| `Online Security`   | Category | Online security subscription status            | `No`, `No internet service`, `Yes`                                                         |
| `Online Backup`     | Category | Online backup subscription status              | `No`, `No internet service`, `Yes`                                                         |
| `Device Protection` | Category | Device protection subscription status          | `No`, `No internet service`, `Yes`                                                         |
| `Tech Support`      | Category | Tech support subscription status               | `No`, `No internet service`, `Yes`                                                         |
| `Streaming TV`      | Category | Streaming TV subscription status               | `No`, `No internet service`, `Yes`                                                         |
| `Streaming Movies`  | Category | Streaming movies subscription status           | `No`, `No internet service`, `Yes`                                                         |
| `Contract`          | Category | Contract term                                  | `Month-to-month`, `One year`, `Two year`                                                   |
| `Paperless Billing` | Category | Whether paperless billing is enabled           | `No`, `Yes`                                                                                |
| `Payment Method`    | Category | Payment method                                 | `Bank transfer (automatic)`, `Credit card (automatic)`, `Electronic check`, `Mailed check` |
| `Monthly Charges`   | Number   | Current monthly service charges                | 18.25 to 118.75                                                                            |
| `Total Charges`     | Number   | Total charges over the customer's tenure       | 18.80 to 8,684.80                                                                          |
| `CLTV`              | Number   | Dataset-provided customer lifetime value field | 2,003 to 6,500; source unit is unspecified                                                 |

## Reproducibility

From the repository root, run `python src/train_model.py`. The script fixes
`random_state=42`, uses an 80/20 stratified split and 5-fold stratified CV,
selects the packaged model by CV F1, and writes the model comparison and
serving artifacts described in the root README.
