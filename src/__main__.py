from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from huggingface_hub import login
from dotenv import load_dotenv
import torch
import os

from .schemas import DiagnosisRequest, DiagnosisResponse
from .model_manager import manager
import torch._dynamo

# --- CONFIG ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") 
CHECKPOINTS_DIR = "src/checkpoints"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Login en Hugging Face 
    if HF_TOKEN:
        print("🔑 Logueando en Hugging Face...")
        login(token=HF_TOKEN)
    
    # 2. PRECARGA DE MODELOS (Nueva parte)
    print("⏳ Precargando modelos en memoria... Esto puede tardar unos minutos.")
    
    DEFAULT_CLS = "final_model"  
    DEFAULT_SUM = "checkpoint-799"
    DEFAULT_GEN = "checkpoint-51"

    # 1. Validar Clasificación
    cls_path = os.path.join(CHECKPOINTS_DIR, "classification", DEFAULT_CLS)
    if not os.path.exists(cls_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint de clasificación no encontrado: {cls_path}")

    # 2. Validar Summarization
    sum_path = os.path.join(CHECKPOINTS_DIR, "summarization", DEFAULT_SUM)
    if not os.path.exists(sum_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint de resumen no encontrado: {sum_path}")

    # 3. Validar Generación
    gen_path = os.path.join(CHECKPOINTS_DIR, "generation", DEFAULT_GEN)
    if not os.path.exists(gen_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint de generación no encontrado: {gen_path}")

    try:
        # Cargamos usando el manager global que ya importas
        manager.load_classifier(DEFAULT_CLS)
        manager.load_summarizer(DEFAULT_SUM)
        manager.load_generator(DEFAULT_GEN)
        print("🚀 ¡Modelos precargados y listos!")
    except Exception as e:
        print(f"⚠️ Alerta: No se pudieron precargar los modelos: {e}")
        print("   La API funcionará, pero la primera petición será lenta.")

    yield
    
    # 3. Apagado
    print("🛑 Apagando API...")
    # Opcional: Limpiar memoria al cerrar
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

app = FastAPI(title="Mental Health AI Pipeline", lifespan=lifespan)

@app.get("/get_status")
def health_check():
    return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}

@app.post("/analyze", response_model=DiagnosisResponse)
async def analyze_case(request: DiagnosisRequest):
    try:
        # Ejecutar pipeline (esto sigue igual)
        result = manager.process_request(
            text=request.patient_text
        )
        
        return result

    except Exception as e:
        print(f"❌ Error interno: {str(e)}")
        # Si ya es una HTTPException, la relanzamos tal cual
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)