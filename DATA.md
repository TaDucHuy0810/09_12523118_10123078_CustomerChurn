# DATA.md

- Dataset: Telco Customer Churn.
- Local file: `data/telco_churn.csv` (7,043 rows, semicolon separated).
- Source: IBM Telco Customer Churn sample, distributed through Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- License/source note: the dataset is a public sample dataset. Verify the current Kaggle page terms before redistribution; this repository keeps the source link and does not claim ownership of the raw data.
- Target: `Churn Label` (`No` = 0, `Yes` = 1).
- Cleaning: `Total Charges` is converted to numeric; rows that cannot be converted are removed. Identifiers, geographic columns and target-derived churn columns are excluded to prevent leakage.
- Reproducibility: `src/train_model.py` fixes `random_state=42`, uses an 80/20 stratified split, and fits imputation/encoding/scaling inside the sklearn pipeline.
