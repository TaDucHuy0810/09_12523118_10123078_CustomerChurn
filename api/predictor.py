from pathlib import Path
import time

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ai-models"


# Load 4 model
knn_model = joblib.load(
    MODEL_DIR / "knn_model.pkl"
)

decision_tree_model = joblib.load(
    MODEL_DIR / "decision_tree_model.pkl"
)

logistic_regression_model = joblib.load(
    MODEL_DIR / "logistic_regression_model.pkl"
)

naive_bayes_model = joblib.load(
    MODEL_DIR / "naive_bayes_model.pkl"
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

    start_time = time.perf_counter()

    df = pd.DataFrame([customer_data])

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"].astype(str).str.strip(),
        errors="coerce"
    )

    df = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    # =========================
    # PREDICTION
    # =========================

    knn_prediction = knn_model.predict(df)[0]

    tree_prediction = decision_tree_model.predict(df)[0]

    logistic_prediction = logistic_regression_model.predict(df)[0]

    naive_bayes_prediction = naive_bayes_model.predict(df)[0]


    # =========================
    # PROBABILITY
    # =========================

    knn_probability = knn_model.predict_proba(df)[0][1]

    tree_probability = decision_tree_model.predict_proba(df)[0][1]

    logistic_probability = (
        logistic_regression_model.predict_proba(df)[0][1]
    )

    naive_bayes_probability = (
        naive_bayes_model.predict_proba(df)[0][1]
    )


    # =========================
    # PROCESSING TIME
    # =========================

    processing_time = time.perf_counter() - start_time


    # =========================
    # RESULT
    # =========================

    return {

        "knn": {
            "prediction": int(knn_prediction),
            "churn": (
                "Yes"
                if knn_prediction == 1
                else "No"
            ),
            "churn_probability": round(
                float(knn_probability),
                4
            )
        },

        "decision_tree": {
            "prediction": int(tree_prediction),
            "churn": (
                "Yes"
                if tree_prediction == 1
                else "No"
            ),
            "churn_probability": round(
                float(tree_probability),
                4
            )
        },

        "logistic_regression": {
            "prediction": int(logistic_prediction),
            "churn": (
                "Yes"
                if logistic_prediction == 1
                else "No"
            ),
            "churn_probability": round(
                float(logistic_probability),
                4
            )
        },

        "naive_bayes": {
            "prediction": int(naive_bayes_prediction),
            "churn": (
                "Yes"
                if naive_bayes_prediction == 1
                else "No"
            ),
            "churn_probability": round(
                float(naive_bayes_probability),
                4
            )
        },

        "processing_time": round(
            processing_time,
            6
        )
    }