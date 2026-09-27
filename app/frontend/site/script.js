const form = document.getElementById("predictionForm");
const statusPill = document.getElementById("statusPill");
const resultTitle = document.getElementById("resultTitle");
const resultDetail = document.getElementById("resultDetail");
const riskPercent = document.getElementById("riskPercent");
const riskFill = document.getElementById("riskFill");
const scoreRing = document.getElementById("scoreRing");
const modelSelect = document.getElementById("modelSelect");
const comparisonBody = document.getElementById("comparisonBody");
const activeModelName = document.getElementById("activeModelName");
const activeModelScore = document.getElementById("activeModelScore");
const activeAccuracy = document.getElementById("activeAccuracy");
const activeRecall = document.getElementById("activeRecall");
const activeAuc = document.getElementById("activeAuc");

const modelLabels = {
  logistic_regression: "Logistic Regression",
  knn: "KNN",
  decision_tree: "Decision Tree",
  naive_bayes: "Naive Bayes",
};

function formatMetric(value) {
  return value == null ? "-" : `${(Number(value) * 100).toFixed(1)}%`;
}

function updateModelSummary(model) {
  const metrics = model.metrics;
  activeModelName.textContent = modelLabels[model.name] || model.name;
  activeModelScore.textContent = `F1-score ${Number(metrics["F1-score"]).toFixed(2)}`;
  activeAccuracy.textContent = formatMetric(metrics.Accuracy);
  activeRecall.textContent = formatMetric(metrics.Recall);
  activeAuc.textContent = Number(metrics["ROC-AUC"]).toFixed(2);
}

async function loadModels() {
  const response = await fetch("/api/models");
  if (!response.ok) throw new Error("Không thể tải danh sách mô hình");
  const data = await response.json();
  modelSelect.innerHTML = "";
  comparisonBody.innerHTML = "";
  data.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = modelLabels[model.name] || model.name;
    modelSelect.append(option);

    const row = document.createElement("tr");
    row.dataset.model = model.name;
    row.innerHTML = `
      <td>${modelLabels[model.name] || model.name}</td>
      <td>${formatMetric(model.metrics.Accuracy)}</td>
      <td>${formatMetric(model.metrics.Precision)}</td>
      <td>${formatMetric(model.metrics.Recall)}</td>
      <td>${formatMetric(model.metrics["F1-score"])}</td>
      <td>${Number(model.metrics["ROC-AUC"]).toFixed(2)}</td>`;
    comparisonBody.append(row);
  });
  modelSelect.value = data.default_model;
  updateModelSummary(
    data.models.find((model) => model.name === data.default_model),
  );
}

modelSelect.addEventListener("change", () => {
  const selectedRow = comparisonBody.querySelector("tr.selected");
  selectedRow?.classList.remove("selected");
  comparisonBody
    .querySelector(`tr[data-model="${modelSelect.value}"]`)
    ?.classList.add("selected");
});

function updateRiskUI(probability) {
  const percent = Math.min(100, Math.max(0, Number(probability || 0) * 100));
  const isRisk = percent >= 50;

  riskPercent.textContent = `${percent.toFixed(1)}%`;
  riskFill.style.width = `${percent}%`;
  scoreRing.style.background = `conic-gradient(${isRisk ? "#ef4444" : "#10b981"} ${percent * 3.6}deg, rgba(148, 163, 184, 0.14) 0deg)`;

  statusPill.textContent = isRisk ? "Rủi ro cao" : "Ổn định";
  statusPill.className = `status-pill ${isRisk ? "alert" : "safe"}`;

  resultTitle.textContent = isRisk
    ? "Khách hàng có nguy cơ rời mạng"
    : "Khách hàng có xu hướng duy trì";
  resultDetail.textContent = isRisk
    ? `Xác suất rời mạng đạt ${percent.toFixed(1)}% - cần chú ý chăm sóc khách hàng hơn.`
    : `Xác suất rời mạng chỉ ${percent.toFixed(1)}% - khách hàng đang ở trạng thái ổn định.`;
}

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  const requestId =
    globalThis.crypto?.randomUUID?.() ??
    `req-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const payload = {
    features: {
      Gender: document.getElementById("gender").value,
      "Senior Citizen": document.getElementById("seniorCitizen").value,
      Partner: document.getElementById("partner").value,
      Dependents: document.getElementById("dependents").value,
      "Tenure Months": Number(document.getElementById("tenure").value),
      "Phone Service": document.getElementById("phoneService").value,
      "Multiple Lines": document.getElementById("multipleLines").value,
      "Internet Service": document.getElementById("internetService").value,
      "Online Security": document.getElementById("onlineSecurity").value,
      "Online Backup": document.getElementById("onlineBackup").value,
      "Device Protection": document.getElementById("deviceProtection").value,
      "Tech Support": document.getElementById("techSupport").value,
      "Streaming TV": document.getElementById("streamingTV").value,
      "Streaming Movies": document.getElementById("streamingMovies").value,
      Contract: document.getElementById("contract").value,
      "Paperless Billing": document.getElementById("paperlessBilling").value,
      "Payment Method": document.getElementById("paymentMethod").value,
      "Monthly Charges": Number(document.getElementById("monthly").value),
      "Total Charges": Number(document.getElementById("total").value),
      CLTV: Number(document.getElementById("cltv").value),
    },
    model: modelSelect.value,
  };

  statusPill.textContent = "Đang phân tích";
  statusPill.className = "status-pill neutral";
  resultTitle.textContent = "Đang dự đoán...";
  resultDetail.textContent =
    "Hệ thống đang đánh giá khả năng rời mạng của khách hàng.";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Request-ID": requestId,
      },
      body: JSON.stringify(payload),
    });

    const output = await response.json();
    if (!response.ok) {
      const detail = output.detail;
      const message =
        typeof detail === "string"
          ? detail
          : detail?.errors?.join("; ") || "Dữ liệu không khớp schema model";
      throw new Error(message || "Có lỗi xảy ra khi gọi API");
    }

    updateRiskUI(Number(output.probability ?? 0));
  } catch (error) {
    statusPill.textContent = "Lỗi API";
    statusPill.className = "status-pill alert";
    resultTitle.textContent = "Không thể dự đoán";
    resultDetail.textContent =
      error.message || "Vui lòng kiểm tra lại dữ liệu hoặc backend.";
    riskPercent.textContent = "0%";
    riskFill.style.width = "0%";
    scoreRing.style.background =
      "conic-gradient(#ef4444 0deg, rgba(148, 163, 184, 0.14) 0deg)";
  }
});

loadModels().catch((error) => {
  comparisonBody.innerHTML = `<tr><td colspan="6">${error.message}</td></tr>`;
});
