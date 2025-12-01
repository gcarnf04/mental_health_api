
-----

## 🧠 Mental Health AI Pipeline (`mental-health-api`) - README.md

Una API de alto rendimiento basada en **FastAPI** que implementa un *pipeline* de tres etapas de Modelos de Lenguaje Grandes (LLMs) y Modelos de Aprendizaje Automático (ML) para procesar notas clínicas (texto de pacientes) y generar una clasificación diagnóstica, un resumen del caso y recomendaciones de tratamiento basadas en evidencia.

El proyecto está diseñado con soporte optimizado para **Apple Silicon (MPS)** y **NVIDIA (CUDA)** en entornos de desarrollo y es fácil de desplegar en Docker.

## ✨ Características Principales

Esta API implementa una tubería (pipeline) de procesamiento de lenguaje natural (PLN) que consta de tres etapas consecutivas:

1.  **Clasificación Diagnóstica (Clasificador Fine-Tuned):** Clasifica el texto clínico de entrada en una de las categorías patológicas definidas.
      * **Modelos utilizados:** Modelo de Hugging Face de Clasificación de Secuencias.
2.  **Resumen Clínico (Modelo T5):** Genera un resumen conciso y relevante del caso a partir del texto completo del paciente.
      * **Modelos utilizados:** Modelo T5 (*encoder-decoder fine-tuned*).
3.  **Generación de Recomendaciones (Llama 3 + LoRA):** Utiliza la clasificación y el resumen para generar una recomendación de tratamiento completa, incluyendo psicoterapia, consideraciones de medicación e intervenciones de estilo de vida.
      * **Optimización:** El modelo **Llama-3-2-1B-Instruct** se carga optimizado con un adaptador **LoRA** y usa cuantización de **4-bit** (si hay GPU disponible) o el backend **MPS** (si se detecta Apple Silicon).

### Patologías Soportadas

  * BPD (Trastorno Límite de la Personalidad)
  * Bipolar Disorder (Trastorno Bipolar)
  * Depression (Depresión)
  * Anxiety (Ansiedad)
  * Schizophrenia (Esquizofrenia)

-----

## ⚙️ Configuración y Ejecución

### Nueva Estructura del Proyecto

La estructura actual del proyecto se ha simplificado. Los directorios principales (`api` y `frontend`) están en la raíz:

```text
/mental-health-api
├── api/             <-- Código Python (FastAPI)
├── frontend/        <-- Interfaz Gráfica (HTML, CSS, JS)
├── checkpoints/     <-- Modelos pre-entrenados
├── requirements.txt
└── Dockerfile
```

### Requisitos

  * Python **\>=3.11**
  * **GPU (Opcional):** Se recomienda **NVIDIA CUDA** o **Apple Silicon MPS** para acelerar la etapa de Generación con Llama 3.

### Instalación

1.  **Clonar el repositorio y entrar al directorio.**
2.  **Crear y activar un entorno virtual:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # En Linux/macOS
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Token de Hugging Face:**
    Crea un archivo llamado `.env` en la raíz del proyecto para la autenticación, si es necesaria para descargar modelos privados o realizar la precarga:
    ```
    # .env
    HF_TOKEN="tu_token_de_hugging_face"
    ```

-----

## 3\. 🏃 Ejecución y Despliegue

### 3.1 Ejecución en Entorno Local (con GPU MPS)

Para ejecutar la aplicación localmente y aprovechar la **GPU de Apple Silicon (MPS)**, debes ejecutarla directamente en el *host* (fuera de Docker).

**Comando de Ejecución Local:**

```bash
uvicorn api.__main__:app --host 0.0.0.0 --port 8001 --reload
```

  * **Verificación:** Accede a `http://localhost:8001/` y verifica que el **Dispositivo de Ejecución** muestre **🍎 GPU (Apple Silicon)**.

### 3.2 Despliegue Contenerizado (Docker)

Para desplegar la aplicación en un contenedor de Docker, debes usar el nombre del módulo `api.__main__`.

#### 1\. Construir la Imagen

```bash
docker build -t mental-health-api .
```

#### 2\. Ejecutar con Aceleración (NVIDIA CUDA)

Si tienes una GPU NVIDIA, usa el *flag* `--gpus all` para exponer el hardware al contenedor, lo cual permite que PyTorch use CUDA.

**Comando de Ejecución con GPU:**

```bash
docker run -d -p 8001:8001 --gpus all -e HF_TOKEN="<TU_TOKEN_HF>" --name mental_app mental-health-api
```

#### 3\. Probar la Aplicación

La aplicación es accesible en: **`http://localhost:8001/`**