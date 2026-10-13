import os
import joblib
import pandas as pd
from pydantic import BaseModel, Field

# Ruta donde se guarda el modelo entrenado
MODEL_PATH = os.path.join("models", "fraud_model.joblib")

class TransactionData(BaseModel):
    # Definimos los campos que esperamos recibir basados en el dataset de transacciones
    # 'Time' y las 28 características 'V1' a 'V28', más el monto 'Amount'
    Time: float = Field(..., description="Tiempo transcurrido desde la primera transacción")
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float = Field(..., description="Monto de la transacción")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}. Ejecuta train.py primero.")
    return joblib.load(MODEL_PATH)

def make_prediction(data: TransactionData):
    model = load_model()
    
    # Convertir los datos de entrada a un DataFrame de Pandas en el orden correcto
    input_data = pd.DataFrame([data.model_dump()])
    
    # Realizar la predicción
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    fraud_probability = float(probability[0][1])
    is_fraud = bool(prediction[0])
    
    return {
        "is_fraud": is_fraud,
        "fraud_probability": fraud_probability,
        "status": "success"
    }