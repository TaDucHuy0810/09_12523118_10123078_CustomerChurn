document
  .getElementById("predictionForm")
  .addEventListener("submit", async function (event) {
    event.preventDefault();

    const result = document.getElementById("result");

    const data = {
      Tenure_Months: Number(document.getElementById("tenure").value),
      Monthly_Charges: Number(document.getElementById("monthly").value),
      Total_Charges: Number(document.getElementById("total").value),
      Contract: document.getElementById("contract").value,
    };

    result.textContent = "Đang dự đoán...";

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      const output = await response.json();

      if (!response.ok) {
        throw new Error(output.detail || "Có lỗi xảy ra");
      }

      result.innerHTML = `
                <p>KNN: ${output.prediction.knn_result}</p>
                <p>Decision Tree: ${output.prediction.decision_tree_result}</p>
            `;
    } catch (error) {
      result.textContent = "Lỗi: " + error.message;
    }
  });
