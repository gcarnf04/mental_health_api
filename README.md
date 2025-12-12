
-----

# 🧠 Mental Health AI Pipeline (`mental-health-api`) - README.md

A high-performance **FastAPI** based API that implements a three-stage pipeline of Large Language Models (LLMs) and Machine Learning (ML) models to process clinical notes (patient text) and generate a diagnostic classification, a case summary, and evidence-based treatment recommendations.

The project is designed with optimized support for **Apple Silicon (MPS)** and **NVIDIA (CUDA)** in development environments and is easy to deploy in Docker.

-----

## ✨ Key Features

This API implements a Natural Language Processing (NLP) pipeline consisting of three consecutive stages:

1.  **Diagnostic Classification (Fine-Tuned Classifier):** Classifies the input clinical text into one of the defined pathological categories.
      * **Models Used:** Hugging Face Sequence Classification Model.
2.  **Clinical Summary (T5 Model):** Generates a concise and relevant case summary from the patient's full text.
      * **Models Used:** Fine-tuned T5 (*encoder-decoder*) Model.
3.  **Recommendation Generation (Llama 3 + LoRA):** Uses the classification and summary to generate a comprehensive treatment recommendation, including psychotherapy, medication considerations, and lifestyle interventions.
      * **Optimization:** The **Llama-3-2-1B-Instruct** model is loaded optimized with a **LoRA** adapter and uses **4-bit quantization** (if GPU is available) or the **MPS** backend (if Apple Silicon is detected).

### Supported Pathologies

  * BPD (Borderline Personality Disorder)
  * Bipolar Disorder
  * Depression
  * Anxiety
  * Schizophrenia

-----

## ⚙️ Configuration and Execution

### Project Structure

The current project structure is flat and simplified. The main directories (`api` and `frontend`) are in the repository root:

```text
/mental-health-api
├── api/             <-- Python Code (FastAPI)
├── frontend/        <-- Graphical Interface (HTML, CSS, JS)
├── checkpoints/     <-- Pre-trained Models
├── requirements.txt
└── Dockerfile
```

### Prerequisites

  * Python **\>=3.11**
  * **GPU (Optional):** **NVIDIA CUDA** or **Apple Silicon MPS** is recommended to accelerate the Generation stage with Llama 3.
  * **Model Checkpoints:** You must download the model files and place them in the correct structure.

#### 📦 Checkpoints Download

Create the necessary folder structure in the repository root and download the models from the following source:

**Download URL:** [INSERTAR URL DE DESCARGA AQUÍ]

Expected structure inside the repository root:

```text
/mental-health-api
└── checkpoints/
    ├── classification/
    ├── summarization/
    └── generation/
```

### Installation

1.  **Clone the repository and enter the directory.**
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    # .venv\Scripts\activate   # On Windows
    ```
3.  **Install dependencies (using pip or uv):**
    ```bash
    # Using pip (Standard)
    pip install -r requirements.txt

    # OR using uv (Faster, if installed)
    uv pip install -r requirements.txt
    ```
4.  **Hugging Face Token:**
    Create a file named `.env` in the repository root to manage the Hugging Face token, which is required for authentication during model preloading:
    ```
    # .env
    HF_TOKEN="your_hugging_face_token"
    ```

-----

## 3\. 🏃 Execution and Deployment

### 3.1 Local Execution (Development Environment)

The application should be run as a module for correct import resolution.

#### A. Execution with Uvicorn (Recommended)

This command is the standard way to run FastAPI during development and supports live reloading.

```bash
uvicorn api.__main__:app --host 0.0.0.0 --port 8001 --reload
```

#### B. Execution as a Script (Alternative)

This uses the Python executable to run the module directly.

```bash
python -m api.__main__
```

  * **Verification:** Access `http://localhost:8001/`. The **Execution Device** should display **🍎 GPU (Apple Silicon)** if you are on an M-series Mac with the correct PyTorch installation.

### 3.2 Containerized Deployment (Docker)

To deploy the application in a Docker container, you must use the module name `api.__main__`.

#### 1\. Build the Image

```bash
docker build -t mental-health-api .
```

#### 2\. Run with Acceleration (NVIDIA CUDA)

If you have an NVIDIA GPU, use the `--gpus all` flag to expose the hardware to the container, allowing PyTorch to use CUDA.

**Execution Command with GPU:**

```bash
docker run -d -p 8001:8001 --gpus all -e HF_TOKEN="<YOUR_HF_TOKEN>" --name mental_app mental-health-api
```

#### 3\. Test the Application

The application, including the GUI and API, is accessible at: **`http://localhost:8001/`**