import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Configuración de la página
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="🔄",
    layout="centered"
)

# Cargar modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "data", "modelo_churn.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "data", "features.pkl"))

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

# Header
st.title("🔄 Churn Predictor")
st.markdown("Predice la probabilidad de que un cliente cancele el servicio.")
st.divider()

# Inputs del usuario
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Antigüedad (meses)", 0, 72, 12)
    monthly_charges = st.slider("Cargo mensual ($)", 18, 120, 65)
    contract = st.selectbox("Tipo de contrato", 
                           ["Mes a mes", "Un año", "Dos años"])
    internet = st.selectbox("Servicio de internet",
                           ["Fibra óptica", "DSL", "Sin internet"])

with col2:
    senior = st.selectbox("¿Es adulto mayor?", ["No", "Sí"])
    partner = st.selectbox("¿Tiene pareja?", ["No", "Sí"])
    payment = st.selectbox("Método de pago",
                          ["Cheque electrónico", "Cheque por correo", 
                           "Transferencia bancaria", "Tarjeta de crédito"])
    tech_support = st.selectbox("¿Tiene soporte técnico?", ["No", "Sí"])

st.divider()

# Construir vector de features
def build_features(tenure, monthly_charges, contract, internet, 
                   senior, partner, payment, tech_support):
    total_charges = tenure * monthly_charges
    
    data = {
        "SeniorCitizen": 1 if senior == "Sí" else 0,
        "Partner": 1 if partner == "Sí" else 0,
        "Dependents": 0,
        "tenure": tenure,
        "PhoneService": 1,
        "MultipleLines": 0,
        "OnlineSecurity": 0,
        "OnlineBackup": 0,
        "DeviceProtection": 0,
        "TechSupport": 1 if tech_support == "Sí" else 0,
        "StreamingTV": 0,
        "StreamingMovies": 0,
        "PaperlessBilling": 1,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender_Female": 0,
        "gender_Male": 1,
        "InternetService_Fiber_optic": 1 if internet == "Fibra óptica" else 0,
        "InternetService_DSL": 1 if internet == "DSL" else 0,
        "InternetService_No": 1 if internet == "Sin internet" else 0,
        "Contract_Month_to_month": 1 if contract == "Mes a mes" else 0,
        "Contract_One_year": 1 if contract == "Un año" else 0,
        "Contract_Two_year": 1 if contract == "Dos años" else 0,
        "PaymentMethod_Electronic_check": 1 if payment == "Cheque electrónico" else 0,
        "PaymentMethod_Mailed_check": 1 if payment == "Cheque por correo" else 0,
        "PaymentMethod_Bank_transfer": 1 if payment == "Transferencia bancaria" else 0,
        "PaymentMethod_Credit_card": 1 if payment == "Tarjeta de crédito" else 0,
    }
    
    renamed = {FIELD_MAP.get(k, k): v for k, v in data.items()}
    df = pd.DataFrame([renamed])
    df = df.reindex(columns=features, fill_value=0)
    return df

# Predicción
if st.button("🔍 Predecir riesgo de churn", type="primary", use_container_width=True):
    df = build_features(tenure, monthly_charges, contract, internet,
                       senior, partner, payment, tech_support)
    proba = model.predict_proba(df)[0][1]
    
    st.divider()
    
    if proba >= 0.7:
        st.error(f"🔴 **Riesgo ALTO de churn**: {proba*100:.1f}%")
        st.markdown("Este cliente tiene alta probabilidad de cancelar. Se recomienda intervención inmediata.")
    elif proba >= 0.4:
        st.warning(f"🟡 **Riesgo MEDIO de churn**: {proba*100:.1f}%")
        st.markdown("Monitorear este cliente y considerar una oferta de retención.")
    else:
        st.success(f"🟢 **Riesgo BAJO de churn**: {proba*100:.1f}%")
        st.markdown("Este cliente tiene baja probabilidad de cancelar el servicio.")
    
    st.progress(float(proba))
    
    st.caption(f"Modelo: XGBoost | AUC-ROC: 0.84 | Total charges estimado: ${tenure * monthly_charges:,.0f}")