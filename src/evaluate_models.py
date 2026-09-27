import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent
RESULT_PATH = BASE_DIR / "ai-models" / "model_comparison.csv"
OUTPUT_DIR = BASE_DIR / "docs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Đọc kết quả đánh giá
df = pd.read_csv(RESULT_PATH)

print("Model comparison:")
print(df)

# Các metric cần vẽ
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

# Vẽ từng biểu đồ
for metric in metrics:
    plt.figure(figsize=(8, 5))

    bars = plt.bar(df["Model"], df[metric])

    plt.title(f"Model Comparison - {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Hiển thị giá trị trên đầu cột
    for bar in bars:
        value = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2%}",
            ha="center"
        )

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{metric.lower().replace('-', '_')}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")

print("All charts created successfully.")