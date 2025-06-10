import torch
import torch.multiprocessing as mp

# ----- AÑADIR ESTO AL PRINCIPIO DE LA EJECUCIÓN DEL SCRIPT/NOTEBOOK -----
# Es crucial que esto se ejecute en el contexto de __main__ ANTES de que se creen
# procesos hijos (incluyendo los de mp.spawn y los de datasets.map)
if __name__ == '__main__': # Esta condición es clave para que solo se ejecute una vez
    try:
        mp.set_start_method('spawn', force=True)
        print("INFO: Multiprocessing start method successfully set to 'spawn'.")
    except RuntimeError as e:
        # Esto puede pasar si se llama varias veces o en un entorno donde ya está fijado.
        print(f"WARNING: Could not set start method to 'spawn': {e}. It might have been set already.")
# --------------------------------------------------------------------------

import logging
import os

# CONFIGURACIÓN DEL LOGGER PRINCIPAL
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger para el notebook

# Importar tu módulo DESPUÉS de establecer el método de inicio
# y DESPUÉS de configurar el logger principal
import training_utils 

# (Aquí iría tu función main() como la tenías)
def main():
    # ... (tu función main sin cambios en su lógica interna por ahora) ...
    config = {
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "dataset_name": "salony/code_explanations",
        "code_column_name": "sentences",
        "explanation_column_name": "Explanation",
        "output_dir": "./code_explainer_model_titan_xp",
        
        # AJUSTES DE MEMORIA PARA TITAN Xp (12GB VRAM)
        "max_length": 768,  # Reducido de 512 para aprovechar más memoria
        "batch_size": 4,    # Incrementado de 2 (TITAN Xp tiene más VRAM que T4)
        "eval_batch_size": 8,  # Incrementado de 4
        "gradient_accumulation_steps": 2,  # Reducido de 4 ya que aumentamos batch_size
        
        # PARÁMETROS DE ENTRENAMIENTO
        "num_epochs": 2,  # Puedes aumentar si quieres
        "learning_rate": 3e-5,  # Ligeramente reducido para más estabilidad
        "weight_decay": 0.01,
        "warmup_ratio": 0.15,  # Incrementado para mejor convergencia
        
        # CONFIGURACIÓN LORA OPTIMIZADA
        "lora_r": 32,      # Incrementado de 16 para aprovechar más capacidad
        "lora_alpha": 64,  # Incrementado proporcionalmente
        "lora_dropout": 0.1,  # Ligeramente incrementado
        
        # ATENCIÓN: SDPA en lugar de Flash Attention 2
        "use_flash_attention": True,  # Se usará SDPA automáticamente
        
        "debug_mode": False,
        "run_comparison": True,
        "seed": 42,
        
        # CONFIGURACIÓN DDP
        "master_addr": "127.0.0.1",
        "master_port": "12355",
        "num_dataloader_workers": 2,  # Incrementado ya que tienes buena GPU
        "ddp_find_unused_parameters": True,
        
        # TEMPLATES
        "prompt_template_train": "<|system|>\\nYou are a programming expert who explains code clearly and concisely.\\n<|user|>\\nExplain this code:\\n{code}\\n<|assistant|>\\n{explanation}<|end|>",
        "prompt_template_inference": "<|system|>\\nYou are a programming expert who explains code clearly and concisely.\\n<|user|>\\nExplain this code:\\n{code}\\n<|assistant|>\\n",
        
        # COMPARACIÓN
        "num_comparison_samples": 50,  # Incrementado para mejor evaluación
        "interactive_comparison_enabled": True,
        "log_level": "INFO"
    }
    
    training_utils.set_seed(config["seed"]) 

    world_size = torch.cuda.device_count()
    # world_size = min(2, torch.cuda.device_count()) 

    if world_size == 0:
        logger.warning("No GPUs detectadas. El entrenamiento será en CPU.")
        config['local_rank'] = -1 
        logger.info("Iniciando entrenamiento en un solo proceso (CPU).")
        # Simplificado para CPU:
        trainer_cpu = training_utils.CodeExplainerTrainer(config)
        trainer_cpu.setup_model_and_tokenizer() # Tokenizer primero
        train_dataset_cpu, eval_dataset_cpu = trainer_cpu.load_and_preprocess_data()
        trainer_cpu.model.to(trainer_cpu.device)
        trainer_cpu.create_dataloaders(train_dataset_cpu, eval_dataset_cpu)
        optimizer_cpu, lr_scheduler_cpu = trainer_cpu.setup_training_components()
        for epoch_cpu in range(config["num_epochs"]):
            train_loss_cpu = trainer_cpu.train_epoch(trainer_cpu.model, trainer_cpu.train_dataloader, optimizer_cpu, lr_scheduler_cpu, epoch_cpu)
            logger.info(f"Época {epoch_cpu+1} CPU. Loss: {train_loss_cpu:.4f}")
            eval_loss_cpu = trainer_cpu.evaluate(trainer_cpu.model, trainer_cpu.eval_dataloader)
            logger.info(f"Evaluación CPU Época {epoch_cpu+1}. Loss: {eval_loss_cpu:.4f}")
            trainer_cpu.save_model(epoch=epoch_cpu)
        final_model_path_cpu = trainer_cpu.save_model(final=True)
        logger.info(f"Entrenamiento CPU completado! Modelo en {final_model_path_cpu}")
        
        if config.get("run_comparison", True) and final_model_path_cpu:
            logger.info("Iniciando comparación en CPU...")
            comparison_cfg_cpu = {
                "dataset_name": config.get("dataset_name"),
                "code_column_name": config.get("code_column_name"),
                "explanation_column_name": config.get("explanation_column_name"),
                "prompt_template_inference": config.get("prompt_template_inference"),
                "tokenizer_path_for_comparison": final_model_path_cpu,
                "num_comparison_samples": config.get("num_comparison_samples"),
                "max_length": config.get("max_length"),
                "output_dir": config.get("output_dir"),
                "interactive_comparison_enabled": True, # Podría no funcionar bien en algunos entornos no TTY
                "comparison_device": "cpu"
            }
            training_utils.run_model_comparison(
                base_model_id=config["model_id"],
                tuned_model_path=final_model_path_cpu,
                comparison_config=comparison_cfg_cpu
            )

    elif world_size >= 1: # Cubre single GPU y multi-GPU
        if world_size == 1:
            logger.info("Detectada 1 GPU. Iniciando entrenamiento en una sola GPU (usando lógica de DDP con world_size=1).")
            config['local_rank'] = 0 
        else:
            logger.info(f"Detectadas {world_size} GPUs. Iniciando entrenamiento distribuido.")
        
        # mp.spawn necesita que la función target (train_distributed) sea importable.
        # El método 'spawn' ya está configurado globalmente.
        mp.spawn(training_utils.train_distributed, args=(world_size, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()