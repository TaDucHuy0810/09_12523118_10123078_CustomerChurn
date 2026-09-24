document
  .getElementById("predictionForm")
  .addEventListener("submit", async function (event) {

    event.preventDefault();

    const result = document.getElementById("result");

    const data = {
      Tenure_Months: Number(document.getElementById("tenure").value),
      Monthly_Charges: Number(document.getElementById("monthly").value),
      Total_Charges: Number(document.getElementById("total").value),
      Contract: document.getElementById("contract").value
    };

    result.innerHTML = "<p>Đang dự đoán...</p>";

    try {

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(data)
        }
      );

      const output = await response.json();

      console.log("API RESPONSE:", output);

      if (!response.ok) {
        throw new Error(output.detail || "API có lỗi");
      }

      // Kiểm tra response
      if (!output.prediction) {
        throw new Error(
          "API không trả về dữ liệu prediction. Response: " +
          JSON.stringify(output)
        );
      }

      const prediction = output.prediction;

      result.innerHTML = `
        <h2>Kết quả dự đoán</h2>

        <div class="model-result">
          <h3>KNN</h3>
          <p>Kết quả: <strong>${prediction.knn?.churn ?? "Không có dữ liệu"}</strong></p>
          <p>Xác suất rời bỏ:
            ${prediction.knn?.churn_probability != null
              ? (prediction.knn.churn_probability * 100).toFixed(2) + "%"
              : "Không có dữ liệu"}
          </p>
        </div>

        <div class="model-result">
          <h3>Decision Tree</h3>
          <p>Kết quả: <strong>${prediction.decision_tree?.churn ?? "Không có dữ liệu"}</strong></p>
          <p>Xác suất rời bỏ:
            ${prediction.decision_tree?.churn_probability != null
              ? (prediction.decision_tree.churn_probability * 100).toFixed(2) + "%"
              : "Không có dữ liệu"}
          </p>
        </div>

        <div class="model-result">
          <h3>Logistic Regression</h3>
          <p>Kết quả: <strong>${prediction.logistic_regression?.churn ?? "Không có dữ liệu"}</strong></p>
          <p>Xác suất rời bỏ:
            ${prediction.logistic_regression?.churn_probability != null
              ? (prediction.logistic_regression.churn_probability * 100).toFixed(2) + "%"
              : "Không có dữ liệu"}
          </p>
        </div>

        <div class="model-result">
          <h3>Naive Bayes</h3>
          <p>Kết quả: <strong>${prediction.naive_bayes?.churn ?? "Không có dữ liệu"}</strong></p>
          <p>Xác suất rời bỏ:
            ${prediction.naive_bayes?.churn_probability != null
              ? (prediction.naive_bayes.churn_probability * 100).toFixed(2) + "%"
              : "Không có dữ liệu"}
          </p>
        </div>

        <p>
          Thời gian xử lý:
          <strong>${prediction.processing_time ?? "Không có dữ liệu"}</strong>
          giây
        </p>
      `;

    } catch (error) {

      console.error("LỖI:", error);

      result.innerHTML = `
        <p style="color:red">
          <strong>Lỗi:</strong> ${error.message}
        </p>
      `;
    }

  });