from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from huggingface_hub import login
from dotenv import load_dotenv
from .utils import get_device
import torch
import gc
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas import DiagnosisRequest, DiagnosisResponse
from .model_manager import manager

# --- CONFIG ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") 
CHECKPOINTS_DIR = "checkpoints"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Login en Hugging Face 
    if HF_TOKEN:
        print("🔑 Logueando en Hugging Face...")
        login(token=HF_TOKEN)
    
    # 2. PRECARGA DE MODELOS
    print("⏳ Precargando modelos en memoria... Esto puede tardar unos minutos.")

    # 1. Validar Clasificación
    cls_path = os.path.join(CHECKPOINTS_DIR, "classification")
    if not os.path.exists(cls_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint de clasificación no encontrado: {cls_path}")

    # 2. Validar Summarization
    sum_path = os.path.join(CHECKPOINTS_DIR, "summarization")
    if not os.path.exists(sum_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint de resumen no encontrado: {sum_path}")

    # 3. Validar Generación
    gen_path = os.path.join(CHECKPOINTS_DIR, "generation")
    if not os.path.exists(gen_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint de generación no encontrado: {gen_path}")

    try:
        # Cargamos usando el manager global
        manager.load_classifier()
        manager.load_summarizer()
        manager.load_generator()
        print("🚀 ¡Modelos precargados y listos!")
    except Exception as e:
        print(f"⚠️ Alerta: No se pudieron precargar los modelos: {e}")
        print("   La API funcionará, pero la primera petición será lenta.")

    yield
    
    # 3. Apagado - Limpieza de Caché adaptada
    print("🛑 Apagando API...")
    device = get_device()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache() 
    gc.collect()

app = FastAPI(title="Mental Health AI Pipeline", lifespan=lifespan)

origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_status")
def health_check():
    return {"status": "ok", "device": get_device()}

@app.post("/analyze", response_model=DiagnosisResponse)
async def analyze_case(request: DiagnosisRequest):
    try:
        result = manager.process_request(
            text=request.patient_text
        )
        
        return result

    except Exception as e:
        print(f"❌ Error interno: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

# Monta la carpeta 'frontend' para que sirva index.html en la raíz (/)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # NOTA: Usamos el puerto 8001
    uvicorn.run(app, host='0.0.0.0', port=8001)

# Comando para ejecutar:
# uvicorn api.__main__:app --host 0.0.0.0 --port 8001