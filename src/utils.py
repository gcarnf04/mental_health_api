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
    return "cuda" if torch.cuda.is_available() else "cpu"

def clean_text(text: str) -> str:
    """Limpieza de texto básica"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()