import os
os.environ["OMP_NUM_THREADS"] = "1" # Limita los hilos de OpenMP para evitar conflictos
import streamlit as st
import torch
import logging
from training_utils import ModelComparator, set_seed

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
