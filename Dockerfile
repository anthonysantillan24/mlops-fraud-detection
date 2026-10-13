# Usamos una imagen oficial ligera de Python
FROM python:3.10-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiamos e instalamos las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código del proyecto y los modelos
COPY . .

# Exponemos el puerto donde corre FastAPI
EXPOSE 8000

# Comando para encender la API al iniciar el contenedor
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]