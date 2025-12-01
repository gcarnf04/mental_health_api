import re
import torch

# Mapa de etiquetas igual al del notebook
LABEL_MAP = {
    0: "BPD",
    1: "Bipolar Disorder",
    2: "Depression",
    3: "Anxiety",
    4: "Schizophrenia"
}
def get_device():
    """
    Determina el dispositivo óptimo: MPS (Apple Silicon), CUDA (NVIDIA), o CPU.
    """
    if torch.backends.mps.is_available():
        # Utiliza el backend de Metal Performance Shaders (MPS) de Apple
        return "mps"
    elif torch.cuda.is_available():
        # Alternativa para GPUs NVIDIA
        return "cuda"
    else:
        # Modo por defecto
        return "cpu"

def clean_text(text: str) -> str:
    """Limpieza de texto básica"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()