from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ai-models"

knn_model = joblib.load(MODEL_DIR / "knn_model.pkl")
decision_tree_model = joblib.load(
    MODEL_DIR / "decision_tree_model.pkl"
)


DROP_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Value",
    "Churn Score",
    "Churn Reason",
    "Churn Label"
]


def predict_customer(customer_data: dict):
    df = pd.DataFrame([customer_data])

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"].astype(str).str.strip(),
        errors="coerce"
    )

    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    knn_prediction = knn_model.predict(df)[0]
    tree_prediction = decision_tree_model.predict(df)[0]

    return {
        "knn_prediction": int(knn_prediction),
        "decision_tree_prediction": int(tree_prediction),
        "knn_result": (
            "Có khả năng rời bỏ"
            if knn_prediction == 1
            else "Không có khả năng rời bỏ"
        ),
        "decision_tree_result": (
            "Có khả năng rời bỏ"
            if tree_prediction == 1
            else "Không có khả năng rời bỏ"
        )
    }