import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score
import mlflow
import mlflow.sklearn
import joblib

def train_model():
    # 1. Cargar los datos crudos
    data_path = os.path.join("data", "raw", "creditcard.csv")
    if not os.path.exists(data_path):
        print(f"Error: No se encontró el archivo en {data_path}. Asegúrate de colocar creditcard.csv en data/raw/")
        return

    print("Cargando el dataset...")
    df = pd.read_csv(data_path)

    # Para propósitos de agilidad en el entrenamiento inicial, tomamos una muestra representativa (puedes quitar esto después)
    df = df.sample(n=50000, random_state=42)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # 2. Dividir en datos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Iniciar experimento con MLflow
    mlflow.start_run()
    
    n_estimators = 50
    max_depth = 10

    print("Entrenando el modelo RandomForest...")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 4. Evaluar el modelo
    preds = model.predict(X_test)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)

    print(f"Métricas - Precisión: {precision:.4f} | Recall: {recall:.4f}")

    # Registrar en MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # 5. Guardar el modelo entrenado en la carpeta models/
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "fraud_model.joblib")
    joblib.dump(model, model_path)
    print(f"Modelo guardado exitosamente en: {model_path}")

    # Registrar artefacto en MLflow
    mlflow.sklearn.log_model(model, "random_forest_fraud_model")
    
    mlflow.end_run()
    print("¡Entrenamiento y registro con MLOps finalizados con éxito!")

if __name__ == "__main__":
    train_model()