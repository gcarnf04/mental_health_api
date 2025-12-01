import os
import re
import gc
import time
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM, 
    pipeline, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import PeftModel
from .utils import clean_text, LABEL_MAP, get_device

CHECKPOINTS_DIR = "checkpoints"

class ModelManager:
    def __init__(self):
        self.current_cls_name = None
        self.current_sum_name = None
        self.current_gen_name = None
        
        # Objetos de modelos en memoria
        self.cls_model = None
        self.cls_tokenizer = None
        self.sum_pipeline = None
        self.gen_model = None
        self.gen_tokenizer = None
        self.base_llama_model = None

    def load_classifier(self):
        if self.cls_model is not None:
            return 
        
        print("🔄 Cargando Clasificador...")
        path = os.path.join(CHECKPOINTS_DIR, "classification")
        
        self.cls_tokenizer = AutoTokenizer.from_pretrained(path)
        self.cls_model = AutoModelForSequenceClassification.from_pretrained(path).to(get_device())
        self.cls_model.eval()
        print("✅ Clasificador cargado.")

    def load_summarizer(self):
        if self.sum_pipeline is not None:
            return 
        
        print("🔄 Cargando Summarizer (T5)...")
        path = os.path.join(CHECKPOINTS_DIR, "summarization")
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        
        device_id = 0 if get_device() == "cuda" else -1 
        self.sum_pipeline = pipeline(
            "summarization", 
            model=model, 
            tokenizer=tokenizer, 
            device=device_id
        )
        print("✅ Summarizer cargado.")

    def load_generator(self):
        # 1. Verificar si ya tenemos este adaptador cargado
        if self.gen_model is not None:
            return

        print("🔄 Cargando Generador (Llama 3 + Adapter)...")
        path = os.path.join(CHECKPOINTS_DIR, "generation")
        base_model_id = "meta-llama/Llama-3.2-1B-Instruct"

        # 2. Configuración Dinámica según Hardware (CPU vs GPU)
        device = get_device() 
        
        if device == "cuda":
            print("   ⚡ Modo GPU detectado: Usando 4-bit quantization")
            # CUDA: Usa cuantización de 4 bits para ahorrar VRAM.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_dtype = torch.bfloat16 
        else:
            # Para MPS (Mac) y CPU
            print(f"   🐢 Modo {device.upper()} detectado: Desactivando 4-bit (Standard Loading)")
            bnb_config = None 
            model_dtype = torch.bfloat16 if device == "mps" else torch.float32

        # 3. Cargar Tokenizer
        if self.gen_tokenizer is None:
            self.gen_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            if self.gen_tokenizer.pad_token is None:
                self.gen_tokenizer.pad_token = self.gen_tokenizer.eos_token

        # 4. Cargar Modelo Base
        if self.base_llama_model is None:
            print(f"   ↳ Cargando Llama Base en {device.upper()}...")
            # Aquí se aplican las configuraciones dinámicas de device_map y dtype
            self.base_llama_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map=device,            
                torch_dtype=model_dtype    
            )

        # 5. Cargar/Cambiar Adaptador (Peft)
        if self.gen_model is not None:
            del self.gen_model
            gc.collect()
            
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            
        print(f"   ↳ Aplicando adaptador LoRA desde {path}...")
        self.gen_model = PeftModel.from_pretrained(self.base_llama_model, path)
        self.gen_model.eval()
        print("✅ Generador cargado.")

    def process_request(self, text):
        inicio = time.time()
        # --- ETAPA 1: CLASIFICACIÓN ---
        cleaned_text = clean_text(text)
        inputs = self.cls_tokenizer(
            cleaned_text, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(self.cls_model.device)

        with torch.no_grad():
            outputs = self.cls_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_label = LABEL_MAP[pred_id]
        confidence = float(probs[pred_id])
        all_probs = {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)}

        # --- ETAPA 2: SUMMARIZATION ---
        summary_result = self.sum_pipeline(
            cleaned_text,
            min_length=256,
            max_length=512,
            clean_up_tokenization_spaces=True
        )
        diagnosis_summary = summary_result[0]["summary_text"]

        # --- ETAPA 3: GENERATION ---
        system_prompt = (
            "You are an expert clinical psychologist providing evidence-based treatment "
            "recommendations. Your recommendations should be specific, actionable, and "
            "tailored to the diagnosed condition."
        )
        
        user_prompt = (
            f"Diagnosed Pathology: {pred_label}\n"
            f"Clinical Summary: {diagnosis_summary}\n\n"
            "Generate a comprehensive, evidence-based treatment recommendation including:\n"
            "1. Recommended psychotherapy approaches\n"
            "2. Medication considerations (if applicable)\n"
            "3. Lifestyle interventions\n"
            "4. Follow-up and monitoring plan"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids = self.gen_tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).to(self.gen_model.device)

        with torch.no_grad():
            output_tokens = self.gen_model.generate(
                **input_ids,
                max_new_tokens=256,
                min_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=self.gen_tokenizer.eos_token_id,
            )

        # Decodificar solo la respuesta nueva
        response = self.gen_tokenizer.decode(
            output_tokens[0][input_ids["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Limpiar prefijo "Recommendation:" si existe
        match = re.search(r"Recommendation:\s*", response, re.IGNORECASE)
        final_recommendation = response[match.end():].strip() if match else response
        fin = time.time()
        print(f"⏱️ Tiempo de procesamiento: {fin - inicio:.2f} segundos")

        return {
            "classification": {
                "pathology": pred_label,
                "confidence": confidence,
                "all_probabilities": all_probs
            },
            "summary": diagnosis_summary,
            "recommendation": final_recommendation,
            "metadata": {
                "original_text_length": len(text),
                "summary_length": len(diagnosis_summary),
                "recommendation_length": len(final_recommendation)
            }
        }

# Instancia global única
manager = ModelManager()