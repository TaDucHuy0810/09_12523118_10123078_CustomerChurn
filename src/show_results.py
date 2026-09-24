import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_PATH = BASE_DIR / "ai-models" / "model_comparison.csv"


def main():
    df = pd.read_csv(RESULT_PATH)

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ]

    for column in metric_columns:
        df[column] = (df[column] * 100).round(2)

    print("\n===== BẢNG SO SÁNH HIỆU QUẢ CÁC MÔ HÌNH =====\n")
    print(df.to_string(index=False))

    print("\n===== MÔ HÌNH CÓ KẾT QUẢ CAO NHẤT THEO TỪNG METRIC =====\n")

    for column in metric_columns:
        best_index = df[column].idxmax()
        best_model = df.loc[best_index, "Model"]
        best_score = df.loc[best_index, column]

        print(f"{column}: {best_model} ({best_score}%)")


if __name__ == "__main__":
    main()