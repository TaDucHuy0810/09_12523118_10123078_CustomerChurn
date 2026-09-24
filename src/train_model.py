import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# =========================
# 1. Load dataset
# =========================

DATA_PATH = "data/telco_churn.csv"

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset shape:", df.shape)


# =========================
# 2. Data preprocessing
# =========================

# Chuẩn hóa cột Total Charges
df["Total Charges"] = (
    df["Total Charges"]
    .astype(str)
    .str.strip()
    .str.replace(",", ".", regex=False)
)

# Chuyển Total Charges sang kiểu số
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

# Chỉ xóa các dòng bị thiếu Total Charges
df = df.dropna(subset=["Total Charges"])

# Xóa các cột không dùng để dự đoán
# Các cột Churn Value, Churn Score và Churn Reason
# có thể gây data leakage
columns_to_drop = [
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
    "Churn Reason"
]

df = df.drop(
    columns=[
        col for col in columns_to_drop
        if col in df.columns
    ]
)

# Chuyển nhãn Churn Label thành 0 và 1
df["Churn Label"] = df["Churn Label"].map({
    "No": 0,
    "Yes": 1
})

# Xóa các dòng không chuyển đổi được nhãn
df = df.dropna(subset=["Churn Label"])

# Tách dữ liệu đầu vào và nhãn
X = df.drop(columns=["Churn Label"])
y = df["Churn Label"].astype(int)

print("\nData after preprocessing:", df.shape)

print("\nClass distribution:")
print(y.value_counts())


# =========================
# 3. Split dataset
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# 4. Identify columns
# =========================

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()


# =========================
# 5. Preprocessing pipeline
# =========================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# =========================
# 6. Define 2 models
# =========================

models = {
    "knn": KNeighborsClassifier(
        n_neighbors=5
    ),

    "decision_tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
}


# =========================
# 7. Train and evaluate
# =========================

results = []

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    print(f"\nTraining {model_name}...")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    print(f"\n===== {model_name.upper()} =====")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    # Lưu model
    model_path = (
        f"ai-models/{model_name}_model.pkl"
    )

    joblib.dump(
        pipeline,
        model_path
    )

    print(
        f"Saved model to: {model_path}"
    )


# =========================
# 8. Save comparison results
# =========================

results_df = pd.DataFrame(results)

print("\n===== MODEL COMPARISON =====")
print(results_df)

results_df.to_csv(
    "ai-models/model_comparison.csv",
    index=False
)

print("\nTraining completed successfully.")