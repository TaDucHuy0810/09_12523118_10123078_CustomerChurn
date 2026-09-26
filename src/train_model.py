"""Train, evaluate and package the customer churn classifiers."""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
ARTIFACT_DIR = BASE_DIR / "ai-models" / "models"
LEGACY_DIR = BASE_DIR / "ai-models"
RANDOM_STATE = 42
DROP_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Churn Value", "Churn Score",
    "Churn Reason",
]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PATH, sep=";")
    for column in ("Monthly Charges", "Total Charges"):
        df[column] = pd.to_numeric(
            df[column].astype(str).str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
    df = df.dropna(subset=["Monthly Charges", "Total Charges"])
    df = df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns])
    df["Churn Label"] = df["Churn Label"].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Churn Label"])
    return df.drop(columns=["Churn Label"]), df["Churn Label"].astype(int)


def make_preprocessor(X: pd.DataFrame, *, dense_output: bool = False) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "str"]).columns.tolist()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense_output)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def make_pipeline(X: pd.DataFrame, estimator, *, dense_output: bool = False) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor(X, dense_output=dense_output)),
        ("model", estimator),
    ])


def build_schema(X: pd.DataFrame, y: pd.Series) -> dict:
    features = []
    for column in X.columns:
        values = X[column].dropna()
        item = {
            "name": column,
            "type": "number" if pd.api.types.is_numeric_dtype(X[column]) else "string",
            "required": True,
        }
        if item["type"] == "number":
            item["min"] = float(values.min())
            item["max"] = float(values.max())
        else:
            item["values"] = sorted(values.astype(str).unique().tolist())
        features.append(item)
    return {
        "target": {"name": "Churn Label", "type": "integer", "values": {"No": 0, "Yes": 1}},
        "features": features,
        "n_samples": int(len(X)),
        "class_distribution": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
    }


def evaluate(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test, cv) -> dict:
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    predictions = pipeline.predict(X_test)
    prediction_seconds = time.perf_counter() - prediction_started
    probabilities = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"},
        n_jobs=-1,
    )
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities) if probabilities is not None else None,
        "CV Accuracy Mean": scores["test_accuracy"].mean(),
        "CV Accuracy Std": scores["test_accuracy"].std(),
        "CV F1 Mean": scores["test_f1"].mean(),
        "CV F1 Std": scores["test_f1"].std(),
        "Train Seconds": train_seconds,
        "Predict Seconds": prediction_seconds,
        "Confusion Matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def main() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidate_models = {
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE, class_weight="balanced"),
        "naive_bayes": GaussianNB(),
    }
    tuned_logistic = GridSearchCV(
        make_pipeline(X, candidate_models["logistic_regression"]),
        {"model__C": [0.1, 1.0, 10.0]}, cv=cv, scoring="f1", n_jobs=-1,
    )
    tuned_logistic.fit(X_train, y_train)
    candidate_models["logistic_regression"] = tuned_logistic.best_estimator_.named_steps["model"]

    results = []
    fitted_models = {}
    for name, estimator in candidate_models.items():
        pipeline = make_pipeline(X, estimator, dense_output=name == "naive_bayes")
        result = evaluate(name, pipeline, X_train, X_test, y_train, y_test, cv)
        fitted_models[name] = pipeline
        results.append(result)
        print(f"{name}: F1={result['F1-score']:.4f}, CV F1={result['CV F1 Mean']:.4f}")

    results_df = pd.DataFrame(results)
    LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(LEGACY_DIR / "model_comparison.csv", index=False)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    best_name = results_df.sort_values(
        ["CV F1 Mean", "CV Accuracy Mean"], ascending=False,
    ).iloc[0]["Model"]
    best_pipeline = fitted_models[best_name]
    joblib.dump(best_pipeline, ARTIFACT_DIR / "model.joblib", compress=3)
    joblib.dump(best_pipeline, LEGACY_DIR / f"{best_name}_model.pkl", compress=3)

    (ARTIFACT_DIR / "schema.json").write_text(json.dumps(build_schema(X, y), indent=2), encoding="utf-8")
    best_result = next(item for item in results if item["Model"] == best_name)
    metadata = {
        "model_name": best_name,
        "model_version": "1.0.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "target": "Churn Label",
        "split": {"test_size": 0.2, "random_state": RANDOM_STATE, "stratified": True},
        "cross_validation": {"folds": 5, "strategy": "StratifiedKFold", "scoring": "f1"},
        "best_params": tuned_logistic.best_params_,
        "metrics": {key: value for key, value in best_result.items() if key != "Confusion Matrix"},
        "library_versions": {"python": sys.version.split()[0], "scikit_learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        "artifact": "ai-models/models/model.joblib",
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    print(f"Best model: {best_name}")
    print(f"Artifacts: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
