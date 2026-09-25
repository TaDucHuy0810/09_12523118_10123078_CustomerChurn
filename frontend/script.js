const form = document.getElementById("predictionForm");
const statusPill = document.getElementById("statusPill");
const resultTitle = document.getElementById("resultTitle");
const resultDetail = document.getElementById("resultDetail");
const riskPercent = document.getElementById("riskPercent");
const riskFill = document.getElementById("riskFill");
const scoreRing = document.getElementById("scoreRing");

function updateRiskUI(probability) {
  const percent = Math.min(100, Math.max(0, Number(probability || 0) * 100));
  const isRisk = percent >= 50;

  riskPercent.textContent = `${percent.toFixed(1)}%`;
  riskFill.style.width = `${percent}%`;
  scoreRing.style.background = `conic-gradient(${isRisk ? "#ef4444" : "#10b981"} ${percent * 3.6}deg, rgba(148, 163, 184, 0.14) 0deg)`;

  statusPill.textContent = isRisk ? "Rủi ro cao" : "Ổn định";
  statusPill.className = `status-pill ${isRisk ? "alert" : "safe"}`;

  resultTitle.textContent = isRisk ? "Khách hàng có nguy cơ rời mạng" : "Khách hàng có xu hướng duy trì";
  resultDetail.textContent = isRisk
    ? `Xác suất rời mạng đạt ${percent.toFixed(1)}% - cần chú ý chăm sóc khách hàng hơn.`
    : `Xác suất rời mạng chỉ ${percent.toFixed(1)}% - khách hàng đang ở trạng thái ổn định.`;
}

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  const payload = {
    features: {
      "Tenure Months": Number(document.getElementById("tenure").value),
      "Monthly Charges": Number(document.getElementById("monthly").value),
      "Total Charges": Number(document.getElementById("total").value),
      Contract: document.getElementById("contract").value,
      "Internet Service": document.getElementById("internetService").value,
      "Payment Method": document.getElementById("paymentMethod").value,
      "Paperless Billing": document.getElementById("paperlessBilling").value,
      "Tech Support": document.getElementById("techSupport").value,
    },
  };

  statusPill.textContent = "Đang phân tích";
  statusPill.className = "status-pill neutral";
  resultTitle.textContent = "Đang dự đoán...";
  resultDetail.textContent = "Hệ thống đang đánh giá khả năng rời mạng của khách hàng.";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const output = await response.json();
    if (!response.ok) {
      throw new Error(output.detail || "Có lỗi xảy ra khi gọi API");
    }

    updateRiskUI(Number(output.probability ?? 0));
  } catch (error) {
    statusPill.textContent = "Lỗi API";
    statusPill.className = "status-pill alert";
    resultTitle.textContent = "Không thể dự đoán";
    resultDetail.textContent = error.message || "Vui lòng kiểm tra lại dữ liệu hoặc backend.";
    riskPercent.textContent = "0%";
    riskFill.style.width = "0%";
    scoreRing.style.background = "conic-gradient(#ef4444 0deg, rgba(148, 163, 184, 0.14) 0deg)";
  }
});
