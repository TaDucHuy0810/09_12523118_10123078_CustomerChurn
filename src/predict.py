import pandas as pd

from pathlib import Path
from joblib import load


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
MODEL_DIR = BASE_DIR / "ai-models"


def prepare_data():
    df = pd.read_csv(DATA_PATH, sep=";")

    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"].astype(str).str.strip(),
        errors="coerce"
    )

    df = df.dropna(subset=["Total Charges"])

    drop_columns = [
        "CustomerID", "Count", "Country", "State", "City", "Zip Code",
        "Lat Long", "Latitude", "Longitude", "Churn Value",
        "Churn Score", "Churn Reason", "Churn Label"
    ]

    X = df.drop(columns=drop_columns, errors="ignore")

    return X


def main():
    X = prepare_data()

    knn_model = load(MODEL_DIR / "knn_model.pkl")
    decision_tree_model = load(
        MODEL_DIR / "decision_tree_model.pkl"
    )

    sample = X.iloc[[0]]

    knn_prediction = knn_model.predict(sample)[0]
    tree_prediction = decision_tree_model.predict(sample)[0]

    print("Thông tin khách hàng thử nghiệm:")
    print(sample.to_string(index=False))

    print("\nKết quả dự đoán:")

    print(
        "KNN:",
        "Có khả năng rời bỏ" if knn_prediction == 1
        else "Không rời bỏ"
    )

    print(
        "Decision Tree:",
        "Có khả năng rời bỏ" if tree_prediction == 1
        else "Không rời bỏ"
    )


if __name__ == "__main__":
    main()