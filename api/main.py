from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from api.predictor import predict_customer


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API dự đoán khách hàng có khả năng rời bỏ nhà mạng",
    version="1.0.0"
)

# Cho phép frontend gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    CustomerID: str = "TEST001"
    Count: int = 1
    Country: str = "United States"
    State: str = "California"
    City: str = "Los Angeles"
    Zip_Code: int = 90001
    Lat_Long: str = "34.05,-118.24"
    Latitude: float = 34.05
    Longitude: float = -118.24
    Gender: str = "Female"
    Senior_Citizen: str = "No"
    Partner: str = "No"
    Dependents: str = "Yes"
    Tenure_Months: int = 4
    Phone_Service: str = "Yes"
    Multiple_Lines: str = "No"
    Internet_Service: str = "Fiber optic"
    Online_Security: str = "No"
    Online_Backup: str = "No"
    Device_Protection: str = "No"
    Tech_Support: str = "No"
    Streaming_TV: str = "No"
    Streaming_Movies: str = "No"
    Contract: str = "Month-to-month"
    Paperless_Billing: str = "Yes"
    Payment_Method: str = "Bank transfer (automatic)"
    Monthly_Charges: float = 70.9
    Total_Charges: float = 273.0
    CLTV: int = 4287


def convert_to_dataset_columns(data: dict[str, Any]) -> dict[str, Any]:
    column_mapping = {
        "Zip_Code": "Zip Code",
        "Lat_Long": "Lat Long",
        "Senior_Citizen": "Senior Citizen",
        "Tenure_Months": "Tenure Months",
        "Phone_Service": "Phone Service",
        "Multiple_Lines": "Multiple Lines",
        "Internet_Service": "Internet Service",
        "Online_Security": "Online Security",
        "Online_Backup": "Online Backup",
        "Device_Protection": "Device Protection",
        "Tech_Support": "Tech Support",
        "Streaming_TV": "Streaming TV",
        "Streaming_Movies": "Streaming Movies",
        "Paperless_Billing": "Paperless Billing",
        "Payment_Method": "Payment Method",
        "Monthly_Charges": "Monthly Charges",
        "Total_Charges": "Total Charges",
    }

    return {
        column_mapping.get(key, key): value
        for key, value in data.items()
    }


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API đang hoạt động"
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        data = customer.model_dump()
        dataset_data = convert_to_dataset_columns(data)

        return {
            "success": True,
            "prediction": predict_customer(dataset_data)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi dự đoán: {str(error)}"
        )