# 🔄 Churn ML Pipeline

> **Predictor de abandono de clientes end-to-end** — desde el análisis exploratorio hasta una API REST en producción, con interpretabilidad de modelo y monitoreo de datos.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-orange?style=flat-square)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.84-success?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)]()

---

## 📌 El problema de negocio

**Perder un cliente cuesta entre 5 y 25 veces más que retenerlo.**

Este proyecto construye un sistema que predice qué clientes tienen alta probabilidad de cancelar un servicio de telecomunicaciones en los próximos 30 días, con suficiente anticipación para que el equipo de retención pueda actuar. No solo predice — **explica por qué**, variable por variable.

---

## 🏗️ Arquitectura del sistema

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Feature Engine  │────▶│  XGBoost Model  │
│  (Telco CSV)    │     │  + SMOTE Balance  │     │   AUC-ROC 0.84  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit App  │◀────│   FastAPI REST   │◀────│  joblib Model   │
│  (Demo visual)  │     │  /predecir POST  │     │   Serializado   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

---

## 📊 Resultados del modelo

| Métrica | Valor |
|--------|-------|
| **AUC-ROC** | **0.8411** |
| Accuracy | 78% |
| F1-Score (Churn) | 0.59 |
| Precision (No Churn) | 0.86 |
| Recall (No Churn) | 0.83 |

> El AUC-ROC de **0.84** indica que el modelo distingue correctamente entre clientes que se van y los que se quedan en el 84% de los casos — un resultado sólido para datos reales de telecomunicaciones.

### 🔍 Variables más importantes (SHAP)

El modelo no es una caja negra. Usando valores SHAP, podemos explicar cada predicción:

1. 🥇 **Contract_Month-to-month** — El mayor predictor de churn. Clientes sin compromiso de permanencia tienen libertad de irse en cualquier momento.
2. 🥈 **PaymentMethod_Electronic check** — Método de pago manual asociado a mayor fricción y menor compromiso automático.
3. 🥉 **MonthlyCharges** — A mayor cargo mensual sin percepción de valor, mayor probabilidad de abandono.
4. **tenure** — A más antigüedad, menor churn. Los clientes leales tienden a quedarse.
5. **InternetService_Fiber optic** — Usuarios de fibra óptica con expectativas altas y menor tolerancia a problemas.

---

## 🗂️ Estructura del proyecto

```
churn-ml-pipeline/
│
├── 📁 data/
│   ├── telco_churn.csv              # Dataset original (Kaggle - Telco Customer Churn)
│   ├── modelo_churn.pkl             # Modelo XGBoost serializado
│   ├── features.pkl                 # Lista de features del modelo
│   ├── confusion_matrix.png         # Matriz de confusión
│   ├── shap_importancia.png         # Gráfico de importancia SHAP
│   └── churn_distribucion.png       # Distribución de la variable objetivo
│
├── 📁 notebooks/
│   ├── 01_eda.ipynb                 # Análisis exploratorio completo
│   └── 02_model.ipynb              # Entrenamiento, evaluación e interpretabilidad
│
├── 📁 api/
│   └── main.py                      # API REST con FastAPI
│
├── 📁 src/                          # Scripts de entrenamiento (próximamente)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/SantiagoAR7/churn-ml-pipeline.git
cd churn-ml-pipeline
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python3 -m venv venv_churn
source venv_churn/bin/activate  # En Windows: venv_churn\Scripts\activate

pip install -r requirements.txt
```

> En Mac, si XGBoost falla: `brew install libomp`

### 3. Ejecutar los notebooks

Abre Jupyter Lab y ejecuta los notebooks en orden:

```bash
jupyter lab
```

- `01_eda.ipynb` → Análisis exploratorio
- `02_model.ipynb` → Entrenamiento del modelo

### 4. Levantar la API

```bash
cd api
uvicorn main:app --reload
```

La documentación interactiva estará disponible en: `http://127.0.0.1:8000/docs`

---

## 🔌 Uso de la API

### Ejemplo de request

```bash
curl -X POST "http://127.0.0.1:8000/predecir" \
  -H "Content-Type: application/json" \
  -d '{
    "SeniorCitizen": 0,
    "tenure": 2,
    "MonthlyCharges": 85.0,
    "TotalCharges": 170.0,
    "Contract_Month_to_month": 1,
    "PaymentMethod_Electronic_check": 1,
    "InternetService_Fiber_optic": 1,
    "Partner": 0, "Dependents": 0,
    "PhoneService": 1, "MultipleLines": 0,
    "OnlineSecurity": 0, "OnlineBackup": 0,
    "DeviceProtection": 0, "TechSupport": 0,
    "StreamingTV": 0, "StreamingMovies": 0,
    "PaperlessBilling": 1,
    "gender_Female": 0, "gender_Male": 1,
    "InternetService_DSL": 0, "InternetService_No": 0,
    "Contract_One_year": 0, "Contract_Two_year": 0,
    "PaymentMethod_Bank_transfer": 0,
    "PaymentMethod_Credit_card": 0,
    "PaymentMethod_Mailed_check": 0
  }'
```

### Respuesta

```json
{
  "probabilidad_churn": 0.9171,
  "churn": true,
  "riesgo": "Alto"
}
```

El campo `riesgo` clasifica automáticamente al cliente en:
- 🔴 **Alto** → probabilidad ≥ 70%
- 🟡 **Medio** → probabilidad entre 40% y 70%
- 🟢 **Bajo** → probabilidad < 40%

---

## 🛠️ Stack tecnológico

| Capa | Tecnología | Propósito |
|------|-----------|-----------|
| Datos | `pandas`, `numpy` | Limpieza y transformación |
| Visualización | `matplotlib`, `seaborn` | EDA y gráficos |
| Balanceo | `imbalanced-learn` (SMOTE) | Corrección de desbalance de clases |
| Modelo | `XGBoost` | Clasificación binaria |
| Interpretabilidad | `SHAP` | Explicabilidad de predicciones |
| API | `FastAPI`, `uvicorn` | Exposición del modelo como servicio REST |
| Serialización | `joblib` | Persistencia del modelo |

---

## 📐 Decisiones técnicas

**¿Por qué XGBoost y no Random Forest o Regresión Logística?**
Para datos tabulares estructurados, XGBoost supera consistentemente a otros algoritmos en precisión, entrena más rápido, y es directamente compatible con SHAP para interpretabilidad. Es el estándar de la industria para este tipo de problema.

**¿Por qué SMOTE?**
El dataset tiene un desbalance natural: 73.5% no churn vs 26.5% churn. Sin corrección, el modelo aprendería a predecir siempre "no churn" y tendría 73% de accuracy sin aprender nada útil. SMOTE genera ejemplos sintéticos de la clase minoritaria para balancear el entrenamiento.

**¿Por qué FastAPI y no Flask?**
FastAPI genera documentación interactiva automáticamente (Swagger UI), tiene validación de datos con Pydantic, y es significativamente más rápido en benchmarks. Para un modelo en producción, esas ventajas importan.

---

## 🗺️ Roadmap

- [x] Análisis exploratorio de datos (EDA)
- [x] Feature engineering + manejo de desbalance (SMOTE)
- [x] Modelo XGBoost con interpretabilidad SHAP
- [x] API REST con FastAPI
- [x] Containerización con Docker
- [x] Versionado de experimentos con MLflow
- [x] Re-entrenamiento automático con GitHub Actions
- [x] Monitoreo de data drift con Evidently
- [x] Demo visual con Streamlit

---

## 📁 Dataset

**Telco Customer Churn** — IBM Sample Data  
Fuente: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
7,043 clientes · 21 variables · Variable objetivo: `Churn` (Yes/No)

---

## 👤 Autor

**Santiago Atehortúa Restrepo**  
Analista de Datos & Automatizaciones → ML Engineer  
[GitHub](https://github.com/SantiagoAR7)

---

**Proyecto en desarrollo activo — GitHub Actions y Evidently próximamente.**
