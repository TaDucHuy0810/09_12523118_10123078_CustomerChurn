document
  .getElementById("predictionForm")
  .addEventListener("submit", async function (event) {
    event.preventDefault();

    const result = document.getElementById("result");

      const data = {
        features: {
          "Tenure Months": Number(document.getElementById("tenure").value),
          "Monthly Charges": Number(document.getElementById("monthly").value),
          "Total Charges": Number(document.getElementById("total").value),
          Contract: document.getElementById("contract").value,
        },
      };

    result.textContent = "Đang dự đoán...";

    try {
      const response = await fetch("/api/predict", {
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

        result.innerHTML = `<p>${output.label}</p><p>Khả năng rời mạng: ${(output.probability * 100).toFixed(1)}%</p>`;
    } catch (error) {
      result.textContent = "Lỗi: " + error.message;
    }
  });
