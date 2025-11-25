# Dockerfile (Ejemplo Básico para un entorno ML)

# Usar una imagen base de Python (3.11+ recomendado)
FROM python:3.11-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requisitos e instalar dependencias
# Se recomienda instalar primero para aprovechar el cacheo de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código (incluida la carpeta src)
COPY . /app

# Exponer el puerto que escuchará la API (Hugging Face Spaces espera el puerto 7860 por defecto)
# Aunque FastAPI escucha en 8000 en su __main__.py,
# Uvicorn debe escuchar en el puerto que pide Spaces.
EXPOSE 7860

# Comando para iniciar la aplicación (el módulo principal: src.__main__)
# Se usa el host 0.0.0.0 y el puerto 7860 para la compatibilidad con Spaces
CMD ["python", "-m", "src"]