# 🧠 Mental Health AI Pipeline (mental-health-api)

Una API de alto rendimiento basada en FastAPI que implementa un *pipeline* de tres etapas de Modelos de Lenguaje Grandes (LLMs) y Modelos de Aprendizaje Automático (ML) para procesar notas clínicas (texto de pacientes) y generar una clasificación diagnóstica, un resumen del caso y recomendaciones de tratamiento basadas en evidencia.

El proyecto está diseñado para ser desplegado fácilmente en entornos como Hugging Face Spaces o Docker.

## ✨ Características Principales

Esta API implementa una tubería (pipeline) de procesamiento de lenguaje natural (PLN) que consta de tres etapas consecutivas:

1. **Clasificación Diagnóstica (Clasificador Fine-Tuned):** Clasifica el texto clínico de entrada en una de las categorías patológicas definidas.
    * **Modelos utilizados:** Modelo de Hugging Face de Clasificación de Secuencias.
2. **Resumen Clínico (Modelo T5):** Genera un resumen conciso y relevante del caso a partir del texto completo del paciente.
    * **Modelos utilizados:** Modelo T5 (encoder-decoder) *fine-tuned*.
3. **Generación de Recomendaciones (Llama 3 + LoRA):** Utiliza la clasificación y el resumen para generar una recomendación de tratamiento completa, incluyendo psicoterapia, consideraciones de medicación e intervenciones de estilo de vida.
    * **Modelos utilizados:** **Llama-3-2-1B-Instruct**, optimizado con un adaptador LoRA y cargado en 4-bit (si hay GPU disponible).

### Patologías Soportadas

El modelo de clasificación actualmente soporta las siguientes categorías diagnósticas:

* BPD (Trastorno Límite de la Personalidad)
* Bipolar Disorder (Trastorno Bipolar)
* Depression (Depresión)
* Anxiety (Ansiedad)
* Schizophrenia (Esquizofrenia)

## ⚙️ Configuración y Ejecución

### Requisitos

* Python **>=3.11**
* **GPU (Opcional pero Recomendado):** Para el módulo de generación, se recomienda una GPU con soporte CUDA para habilitar la cuantización de 4 bits y optimizar el rendimiento.

### Instalación

1. **Clonar el repositorio:**

    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd mental-health-api
    ```

2. **Crear y activar un entorno virtual:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Linux/macOS
    # .venv\Scripts\activate   # En Windows
    ```

3. **Instalar dependencias:**
    Las dependencias se encuentran en `requirements.txt`.

    ```bash
    pip install -r requirements.txt

    ```

    *Nota: Si estás usando una GPU, es posible que necesites instalar `torch` con el comando específico de PyTorch para tu versión de CUDA, como se sugiere en `requirements.txt`.*

4. **Modelos y Checkpoints:**
    Asegúrate de tener los modelos pre-entrenados y *fine-tuned* en la estructura de carpetas esperada por `src/model_manager.py`:

    ```text
    src/checkpoints/
    ├── classification/
    │   └── final_model/
    ├── summarization/
    │   └── checkpoint-799/
    └── generation/
        └── checkpoint-51/
    ```

5. **Archivo `.env`:**
    Crea un archivo llamado `.env` en la raíz del proyecto para gestionar el token de Hugging Face, que se requiere para la autenticación durante la precarga del modelo.

    .env
    HF_TOKEN="tu_token_de_hugging_face"

## Ejecución Local

Para ejecutar la API localmente usando Uvicorn:

```bash
python -m src
