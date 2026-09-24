import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
df = pd.read_csv("data/telco_churn.csv", sep=";")

# Cấu hình hiển thị
sns.set_theme(style="whitegrid")

# 1. Phân bố khách hàng rời mạng
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn Label")
plt.title("Phân bố khách hàng rời mạng")
plt.xlabel("Churn Label")
plt.ylabel("Số lượng khách hàng")
plt.tight_layout()
plt.show()

# 2. Rời mạng theo giới tính
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Gender", hue="Churn Label")
plt.title("Rời mạng theo giới tính")
plt.tight_layout()
plt.show()

# 3. Rời mạng theo loại hợp đồng
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn Label")
plt.title("Rời mạng theo loại hợp đồng")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 4. Phân bố thời gian sử dụng dịch vụ
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Tenure Months", hue="Churn Label",
             bins=30, kde=True, element="step")
plt.title("Phân bố thời gian sử dụng dịch vụ")
plt.tight_layout()
plt.show()

# 5. Phân bố phí hàng tháng
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Churn Label", y="Monthly Charges")
plt.title("Phí hàng tháng theo tình trạng rời mạng")
plt.tight_layout()
plt.show()