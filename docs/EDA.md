# EDA: quan sát và quyết định xử lý

1. `eda_churn_distribution.png`: lớp No chiếm khoảng 73% và Yes khoảng 27%, cho thấy mất cân bằng vừa phải. Vì vậy đánh giá thêm Precision, Recall, F1 và dùng stratify thay vì chỉ Accuracy.
2. `eda_churn_by_gender.png`: phân bố churn giữa hai giới gần nhau, không có lý do bỏ `Gender`; giữ và mã hóa One-Hot.
3. `eda_churn_by_contract.png`: hợp đồng month-to-month có tỷ lệ churn cao hơn hợp đồng dài hạn. Giữ `Contract` vì có tín hiệu nghiệp vụ rõ.
4. `eda_tenure_distribution.png`: khách hàng tenure thấp tập trung nhiều hơn ở nhóm churn. Giữ `Tenure Months` và chuẩn hóa trong pipeline.
5. `eda_monthly_charges.png`: nhóm churn có monthly charge cao hơn và có ngoại lệ. Không cắt ngoại lệ tùy ý; dùng median imputation và model có regularization.
6. `eda_churn_by_payment_method.png`: electronic check có churn cao hơn một số phương thức khác. Giữ `Payment Method` và One-HotEncoder với unknown handling.
7. `eda_churn_by_internet_service.png`: tỷ lệ churn khác nhau theo loại Internet Service. Giữ đặc trưng phân loại này; các cột `Churn Score`, `Churn Value`, `Churn Reason` bị loại vì được sinh từ mục tiêu.

Các quyết định trên được thực hiện lại trong `src/train_model.py`, nơi preprocessing được fit bên trong Pipeline sau khi chia train/test để tránh leakage.
