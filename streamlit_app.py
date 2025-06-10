import os
os.environ["OMP_NUM_THREADS"] = "1" # Limita los hilos de OpenMP para evitar conflictos
import streamlit as st
import torch
import logging
import random
import numpy as np
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
from typing import Dict, Any, Optional
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
import gc

# Fija las semillas para reproducibilidad.
def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Crea la configuración BitsAndBytes optimizada.
def create_bnb_config(compute_dtype=torch.float16) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=torch.uint8
    )

class ModelComparator:
    def __init__(self, base_model_id: str, tuned_model_path: str, config: Dict[str, Any], device: Optional[str] = None):
        self.base_model_id = base_model_id
        self.tuned_model_path = tuned_model_path
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ModelComparator")

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.logger.info(f"ModelComparator usando device: {self.device}")

        self.tokenizer = None
        self.base_model = None
        self.tuned_model = None
        self.rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def load_models(self):
        self.logger.info("Cargando tokenizer...")
        tokenizer_path = self.config.get("tokenizer_path_for_comparison", self.tuned_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        bnb_config_val_comp = None
        model_dtype_comp = torch.float32
        if self.device == 'cuda':
            model_dtype_comp = torch.float16
            bnb_config_val_comp = create_bnb_config(compute_dtype=model_dtype_comp)
            self.logger.info("ModelComparator: Usando float16 y BNB compute_dtype=float16 para CUDA.")
        else:
            self.logger.warning("ModelComparator: BitsAndBytesConfig no se usará en CPU para ModelComparator.")
            
        self.logger.info(f"Cargando modelo base {self.base_model_id} en {self.device}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config_val_comp,
            device_map=self.device if self.device != "cpu" else {"": "cpu"},
            torch_dtype=model_dtype_comp,
            trust_remote_code=True
        )
        
        self.logger.info(f"Cargando modelo fine-tuned desde {self.tuned_model_path} en {self.device}...")
        
        base_for_peft_config = {
            "quantization_config": bnb_config_val_comp,
            "torch_dtype": model_dtype_comp,
            "trust_remote_code": True
        }
        if self.device == "cpu":
            base_for_peft_config["device_map"] = {"": "cpu"}

        base_for_peft = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            **base_for_peft_config
        )
        if self.device != "cpu":
             base_for_peft.to(self.device)

        self.tuned_model = PeftModel.from_pretrained(
            base_for_peft,
            self.tuned_model_path,
            is_trainable=False
        )
        if self.device != "cpu":
            self.tuned_model.to(self.device)
        
        self.base_model.eval()
        self.tuned_model.eval()
        self.logger.info("Modelos cargados y en modo evaluación.")
        
    def generate_response(self, model, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.config.get("max_length", 1024) - max_new_tokens)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        use_amp = self.device == 'cuda'
        autocast_dtype_comp = torch.float16 if use_amp else torch.float32
        current_device_type_str = self.device
        with torch.no_grad():
            with torch.amp.autocast(device_type=current_device_type_str, dtype=autocast_dtype_comp, enabled=use_amp):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()
    
    def calculate_metrics(self, predictions: list[str], references: list[str]) -> Dict[str, float]:
        metrics = {}
        rouge_scores_agg = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for pred, ref in zip(predictions, references):
            if not pred.strip() or not ref.strip():
                self.logger.warning(f"Predicción o referencia vacía encontrada. Pred: '{pred}', Ref: '{ref}'. Saltando para ROUGE.")
                for key_rouge in rouge_scores_agg:
                    rouge_scores_agg[key_rouge].append(0.0)
                continue

            scores = self.rouge_scorer_obj.score(ref, pred)
            for key_rouge in rouge_scores_agg:
                rouge_scores_agg[key_rouge].append(scores[key_rouge].fmeasure)
        
        for key_rouge in rouge_scores_agg:
            metrics[key_rouge] = np.mean(rouge_scores_agg[key_rouge]) if rouge_scores_agg[key_rouge] else 0.0
        
        try:
            valid_preds = [p for p, r in zip(predictions, references) if p.strip() and r.strip()]
            valid_refs = [r for p, r in zip(predictions, references) if p.strip() and r.strip()]

            if valid_preds and valid_refs:
                P, R, F1 = bert_score_fn(valid_preds, valid_refs, lang="en", verbose=False, device=self.device if self.device != "cpu" else None)
                metrics['bert_score_f1'] = F1.mean().item()
                metrics['bert_score_precision'] = P.mean().item()
                metrics['bert_score_recall'] = R.mean().item()
            else:
                metrics.update({'bert_score_f1': 0.0, 'bert_score_precision': 0.0, 'bert_score_recall': 0.0})
        except Exception as e:
            self.logger.warning(f"No se pudo calcular BERT Score: {e}")
            metrics['bert_score_f1'] = 0.0
        
        pred_lengths = [len(pred.split()) for pred in predictions]
        ref_lengths = [len(ref.split()) for ref in references]
        metrics['avg_pred_length'] = np.mean(pred_lengths) if pred_lengths else 0.0
        metrics['avg_ref_length'] = np.mean(ref_lengths) if ref_lengths else 0.0
        if metrics['avg_ref_length'] > 0:
            metrics['length_ratio'] = metrics['avg_pred_length'] / metrics['avg_ref_length']
        else:
            metrics['length_ratio'] = 0.0
        return metrics
    
    def evaluate_on_test_set(self, test_dataset, num_samples: int = 50) -> Dict[str, Any]:
        self.logger.info(f"Evaluando modelos en {num_samples} muestras del conjunto de prueba...")
        
        if len(test_dataset) > num_samples:
            if len(test_dataset) < num_samples: num_samples = len(test_dataset)
            indices = np.random.choice(len(test_dataset), num_samples, replace=False)
            test_samples = test_dataset.select(indices)
        else:
            test_samples = test_dataset
            
        base_predictions, tuned_predictions, references, prompts_used = [], [], [], []

        code_col = self.config.get("code_column_name", "sentences")
        explanation_col = self.config.get("explanation_column_name", "Explanation")
        
        prompt_template_inference = self.config.get("prompt_template_inference",
            "<|system|>\\nYou are a programming expert who explains code clearly and concisely.\\n<|user|>\\nExplain this code:\\n{code}\\n<|assistant|>\\n"
        )

        for i, example in enumerate(test_samples): # Removed tqdm as it's not imported
            code = example[code_col]
            reference = example[explanation_col]
            
            prompt = prompt_template_inference.format(code=code)
            
            try:
                base_pred = self.generate_response(self.base_model, prompt)
                tuned_pred = self.generate_response(self.tuned_model, prompt)
                
                base_predictions.append(base_pred)
                tuned_predictions.append(tuned_pred)
                references.append(reference)
                prompts_used.append(code)
                
            except Exception as e:
                self.logger.warning(f"Error en muestra {i} (código: {code[:50]}...): {e}")
                base_predictions.append("")
                tuned_predictions.append("")
                references.append(reference if reference else "")
                prompts_used.append(code if code else "")
                continue
                
            if i % 10 == 0 and self.device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        
        base_metrics = self.calculate_metrics(base_predictions, references)
        tuned_metrics = self.calculate_metrics(tuned_predictions, references)
        
        results = {
            'base_model_metrics': base_metrics,
            'tuned_model_metrics': tuned_metrics,
            'samples': {
                'prompts': prompts_used[:10],
                'references': references[:10],
                'base_predictions': base_predictions[:10],
                'tuned_predictions': tuned_predictions[:10]
            },
            'improvement': {}
        }
        
        for metric in base_metrics:
            if isinstance(base_metrics[metric], (int, float)) and isinstance(tuned_metrics[metric], (int, float)):
                if abs(base_metrics[metric]) > 1e-9:
                    improvement = ((tuned_metrics[metric] - base_metrics[metric]) / base_metrics[metric]) * 100
                    results['improvement'][metric] = improvement
        return results
    
    def generate_comparison_report(self, results: Dict[str, Any], save_path: Optional[str] = None):
        # This method is not used in streambit_app.py, so it's not strictly necessary to move.
        # However, if the user wants to completely remove the dependency, it should be moved or removed.
        # For now, I will omit it as it's not called by the Streamlit app.
        pass

    def interactive_comparison(self):
        # This method is not used in streambit_app.py, so it's not strictly necessary to move.
        pass

# Configuración básica del logger para la aplicación Streamlit
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración del modelo y rutas ---
# Asegúrate de que esta ruta coincida con la salida de tu entrenamiento
TUNED_MODEL_PATH = "fineTunedModel"
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Configuración para la inferencia
# Puedes ajustar estos parámetros según sea necesario para el despliegue
INFERENCE_CONFIG = {
    "dataset_name": "salony/code_explanations", # Necesario para ModelComparator, aunque no se use para cargar datos aquí
    "code_column_name": "sentences",
    "explanation_column_name": "Explanation",
    "prompt_template_inference": "You are a programming expert who explains code clearly and concisely.Explain this code:\\n{code}\\n<|assistant|>\\n",
    "tokenizer_path_for_comparison": TUNED_MODEL_PATH, # Usar el tokenizer del modelo fine-tuned
    "max_length": 768, # Longitud máxima para tokenización
    "output_dir": "./", # No se usa para inferencia, pero es requerido por ModelComparator
    "num_comparison_samples": 0, # No se necesitan muestras para comparación en la app
    "interactive_comparison_enabled": False, # Deshabilitar interactividad de consola
    "comparison_device": "cuda" if torch.cuda.is_available() else "cpu"
}

@st.cache_resource # Cachear el modelo para evitar recargarlo en cada interacción
def load_models_for_inference():
    """Carga el modelo fine-tuned y el modelo base para inferencia."""
    if not os.path.exists(TUNED_MODEL_PATH):
        st.error(f"Error: El modelo fine-tuned no se encontró en la ruta: {TUNED_MODEL_PATH}")
        st.stop() # Detener la ejecución de Streamlit si el modelo no está
        
    set_seed(INFERENCE_CONFIG.get("seed", 42)) # Asegurar reproducibilidad si es relevante
    
    comparator = ModelComparator(
        base_model_id=BASE_MODEL_ID,
        tuned_model_path=TUNED_MODEL_PATH,
        config=INFERENCE_CONFIG,
        device=INFERENCE_CONFIG["comparison_device"]
    )
    
    try:
        comparator.load_models()
        logger.info("Modelos cargados exitosamente para inferencia en Streamlit.")
        return comparator
    except Exception as e:
        logger.error(f"Error al cargar los modelos para inferencia: {e}", exc_info=True)
        st.error(f"No se pudieron cargar los modelos. Por favor, asegúrate de que el entrenamiento se haya completado y el modelo esté en '{TUNED_MODEL_PATH}'. Error: {e}")
        st.stop()
        return None

# --- Interfaz de usuario de Streamlit ---
st.set_page_config(page_title="Code Explainer AI", layout="wide")

st.title("🤖 Code Explainer AI")
st.markdown("""
Esta aplicación utiliza un modelo de lenguaje fine-tuned para generar explicaciones claras y concisas de fragmentos de código.
""")

# Cargar los modelos una vez
comparator_instance = load_models_for_inference()

if comparator_instance:
    code_input = st.text_area("Pega tu código aquí:", height=300, 
                              placeholder="Ejemplo:\n\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)")

    if st.button("Explicar Código"):
        if code_input.strip():
            with st.spinner("Generando explicación..."):
                try:
                    # Clean the code input from unwanted prefixes
                    cleaned_code_input = code_input.replace("Código escrito por el usuario:", "").strip()
                    # Generar prompt para el modelo
                    prompt = INFERENCE_CONFIG["prompt_template_inference"].format(code=cleaned_code_input)
                    
                    # Generar respuesta con el modelo fine-tuned
                    tuned_explanation = comparator_instance.generate_response(
                        comparator_instance.tuned_model,
                        prompt,
                        max_new_tokens=256, # Ajusta según la longitud esperada de la explicación
                        temperature=0.7
                    ).replace("<|assistant|>", "").replace("<|end|>", "")
                    
                    st.subheader("Generated Explanation (Fine-tuned Model):")
                    st.text_area("Generated Explanation (Fine-tuned Model):", value=tuned_explanation, height=250, key="tuned_explanation_output")
                    
                    # Mostrar también la explicación del modelo base para comparación
                    base_explanation = comparator_instance.generate_response(
                        comparator_instance.base_model,
                        prompt,
                        max_new_tokens=256,
                        temperature=0.7
                    ).replace("<|assistant|>", "").replace("<|end|>", "")
                    st.subheader("Generated Explanation (Base Model):")
                    st.text_area("Generated Explanation (Base Model):", value=base_explanation, height=250, key="base_explanation_output")
                        
                except Exception as e:
                    st.error(f"Ocurrió un error al generar la explicación: {e}")
                    logger.error(f"Error durante la generación de explicación: {e}", exc_info=True)
        else:
            st.warning("Por favor, ingresa algún código para explicar.")

st.markdown("---")
st.markdown("Desarrollado para la plataforma Strembit.")
