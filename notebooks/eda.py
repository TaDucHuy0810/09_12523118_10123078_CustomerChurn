import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# Xác định đường dẫn project
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
OUTPUT_DIR = BASE_DIR / "docs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Đọc dữ liệu
df = pd.read_csv(DATA_PATH, sep=";")

# Cấu hình hiển thị
sns.set_theme(style="whitegrid")


def save_plot(filename):
    output_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()


# Hiển thị thông tin tổng quan
print("===== THÔNG TIN DATASET =====")
print(f"Số dòng: {df.shape[0]}")
print(f"Số cột: {df.shape[1]}")

print("\n===== PHÂN BỐ KHÁCH HÀNG =====")
print(df["Churn Label"].value_counts())

print("\n===== TỶ LỆ KHÁCH HÀNG RỜI MẠNG =====")
churn_rate = df["Churn Label"].value_counts(normalize=True) * 100
print(churn_rate.round(2))


# 1. Phân bố khách hàng rời mạng
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Churn Label")
plt.title("Phân bố khách hàng rời mạng")
plt.xlabel("Tình trạng rời mạng")
plt.ylabel("Số lượng khách hàng")
save_plot("eda_churn_distribution.png")


# 2. Rời mạng theo giới tính
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Gender", hue="Churn Label")
plt.title("Tình trạng rời mạng theo giới tính")
plt.xlabel("Giới tính")
plt.ylabel("Số lượng khách hàng")
plt.legend(title="Churn Label")
save_plot("eda_churn_by_gender.png")


# 3. Rời mạng theo loại hợp đồng
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Contract", hue="Churn Label")
plt.title("Tình trạng rời mạng theo loại hợp đồng")
plt.xlabel("Loại hợp đồng")
plt.ylabel("Số lượng khách hàng")
plt.xticks(rotation=15)
plt.legend(title="Churn Label")
save_plot("eda_churn_by_contract.png")


# 4. Phân bố thời gian sử dụng dịch vụ
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="Tenure Months",
    hue="Churn Label",
    bins=30,
    kde=True,
    element="step"
)
plt.title("Phân bố thời gian sử dụng dịch vụ")
plt.xlabel("Số tháng sử dụng")
plt.ylabel("Số lượng khách hàng")
save_plot("eda_tenure_distribution.png")


# 5. Phân bố phí hàng tháng
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn Label",
    y="Monthly Charges"
)
plt.title("Phí hàng tháng theo tình trạng rời mạng")
plt.xlabel("Tình trạng rời mạng")
plt.ylabel("Phí hàng tháng")
save_plot("eda_monthly_charges.png")


# 6. Rời mạng theo phương thức thanh toán
plt.figure(figsize=(11, 6))
sns.countplot(
    data=df,
    x="Payment Method",
    hue="Churn Label"
)
plt.title("Tình trạng rời mạng theo phương thức thanh toán")
plt.xlabel("Phương thức thanh toán")
plt.ylabel("Số lượng khách hàng")
plt.xticks(rotation=25, ha="right")
plt.legend(title="Churn Label")
save_plot("eda_churn_by_payment_method.png")


# 7. Rời mạng theo dịch vụ Internet
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Internet Service",
    hue="Churn Label"
)
plt.title("Tình trạng rời mạng theo loại dịch vụ Internet")
plt.xlabel("Loại dịch vụ Internet")
plt.ylabel("Số lượng khách hàng")
plt.legend(title="Churn Label")
save_plot("eda_churn_by_internet_service.png")


print("\nEDA completed successfully.")