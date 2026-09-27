import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_PATH = BASE_DIR / "ai-models" / "model_comparison.csv"
OUTPUT_DIR = BASE_DIR / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(RESULT_PATH)
model_labels = {
    "logistic_regression": "Logistic Regression",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
}

for model_name, label in model_labels.items():
    row = results.loc[results["Model"] == model_name]
    if row.empty:
        raise ValueError(f"Missing confusion matrix for {model_name} in {RESULT_PATH}")

    matrix = np.asarray(ast.literal_eval(row.iloc[0]["Confusion Matrix"]))
    print(f"\n{label}\n{matrix}")
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No Churn", "Churn"],
    )
    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {label}")
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

print("\nConfusion matrices created from the canonical holdout results.")