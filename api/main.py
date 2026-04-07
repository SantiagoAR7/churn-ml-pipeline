from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Churn Predictor API", version="1.0")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "data", "modelo_churn.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "data", "features.pkl"))

class ClienteData(BaseModel):
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    gender_Female: int
    gender_Male: int
    InternetService_DSL: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    Contract_Month_to_month: int
    Contract_One_year: int
    Contract_Two_year: int
    PaymentMethod_Bank_transfer: int
    PaymentMethod_Credit_card: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

# Mapeo de nombres Pydantic → nombres reales del modelo
FIELD_MAP = {
    "InternetService_Fiber_optic": "InternetService_Fiber optic",
    "Contract_Month_to_month": "Contract_Month-to-month",
    "Contract_One_year": "Contract_One year",
    "Contract_Two_year": "Contract_Two year",
    "PaymentMethod_Bank_transfer": "PaymentMethod_Bank transfer (automatic)",
    "PaymentMethod_Credit_card": "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic_check": "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed_check": "PaymentMethod_Mailed check",
}

@app.get("/")
def root():
    return {"mensaje": "API de predicción de churn activa ✅"}

@app.post("/predecir")
def predecir(cliente: ClienteData):
    datos = cliente.dict()
    datos_renombrados = {FIELD_MAP.get(k, k): v for k, v in datos.items()}
    df = pd.DataFrame([datos_renombrados])
    df = df.reindex(columns=features, fill_value=0)
    proba = model.predict_proba(df)[0][1]
    prediccion = int(proba >= 0.5)
    return {
        "probabilidad_churn": round(float(proba), 4),
        "churn": bool(prediccion),
        "riesgo": "Alto" if proba >= 0.7 else "Medio" if proba >= 0.4 else "Bajo"
    }