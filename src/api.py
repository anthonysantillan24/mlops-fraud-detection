import time
from fastapi import FastAPI, HTTPException
from src.predict import TransactionData, make_prediction

app = FastAPI(
    title="MLOps Fraud Detection API",
    description="API de baja latencia para detección de fraudes en tiempo real",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "¡Bienvenido a la API de Detección de Fraudes con MLOps!", "status": "online"}

@app.post("/predict")
def predict_fraud(transaction: TransactionData):
    start_time = time.time()
    try:
        result = make_prediction(transaction)
        # Calculamos la latencia de respuesta en milisegundos
        latency_ms = (time.time() - start_time) * 1000
        result["latency_ms"] = round(latency_ms, 2)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))