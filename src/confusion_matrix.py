import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
OUTPUT_DIR = BASE_DIR / "docs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv(DATA_PATH, sep=";")

# Chuẩn hóa cột Total Charges
df["Total Charges"] = pd.to_numeric(
    df["Total Charges"].astype(str).str.strip(),
    errors="coerce"
)

df = df.dropna(subset=["Total Charges"])

# Loại bỏ các cột không dùng
drop_columns = [
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

df = df.drop(columns=drop_columns, errors="ignore")

# Chuyển nhãn
df["Churn Label"] = df["Churn Label"].map({
    "No": 0,
    "Yes": 1
})

df = df.dropna(subset=["Churn Label"])

X = df.drop(columns=["Churn Label"])
y = df["Churn Label"]

# Chia dữ liệu giống file train_model.py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Xác định kiểu cột
numeric_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns

# Tiền xử lý số
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Tiền xử lý dữ liệu dạng chữ
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
}

for model_name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name}")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    display.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")
    output_path = OUTPUT_DIR / f"confusion_matrix_{filename}.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

print("\nConfusion Matrix created successfully.")