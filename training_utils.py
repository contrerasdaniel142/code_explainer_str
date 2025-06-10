import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# No importes torch.multiprocessing as mp aquí a menos que train_distributed lo use internamente para spawn más.
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from transformers.utils import is_flash_attn_2_available
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from tqdm.auto import tqdm
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import gc
import numpy as np
from rouge_score import rouge_scorer
# Renombrar bert_score para evitar posible conflicto si la función se llama 'score'
from bert_score import score as bert_score_fn
from datetime import datetime
import random
import html

logger = logging.getLogger(__name__)

def set_seed(seed_value: int):
    """Fija las semillas para reproducibilidad."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # logger.info(f"Set seed to {seed_value}") # El logger puede no estar full configurado aquí

def create_bnb_config(compute_dtype=torch.float16) -> BitsAndBytesConfig:
    """Crea la configuración BitsAndBytes optimizada."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=torch.uint8
    )

class CodeExplainerTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Asegurar que el logger se obtenga correctamente
        self.logger = logging.getLogger(f"{__name__}.CodeExplainerTrainer")

        # El device se maneja de forma diferente en DDP vs single
        if 'local_rank' in config and config['local_rank'] != -1: # DDP or specific single GPU
             self.device = torch.device(f"cuda:{config['local_rank']}")
        elif torch.cuda.is_available(): # Single GPU default
            self.device = torch.device("cuda")
        else: # CPU
            self.device = torch.device("cpu")
        
        self.model = None
        self.tokenizer = None
        self.train_dataloader = None
        self.eval_dataloader = None
        
    def setup_distributed(self, rank: int, world_size: int):
        os.environ['MASTER_ADDR'] = self.config.get("master_addr", "localhost")
        os.environ['MASTER_PORT'] = self.config.get("master_port", "12355")
        dist.init_process_group("gloo", rank=rank, world_size=world_size) # nccl para linux u otros
        torch.cuda.set_device(rank)
        self.device = torch.device(f"cuda:{rank}")
        self.config['local_rank'] = rank
        self.logger.info(f"Rank {rank}: Distributed setup complete on device {self.device}.")
        
    def cleanup_distributed(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        
    def load_and_preprocess_data(self):
        self.logger.info("Cargando dataset...")
        dataset = load_dataset(self.config.get("dataset_name", "salony/code_explanations"))
        self.logger.info(f"Dataset cargado: {len(dataset['train'])} train, {len(dataset['test'])} test")
        
        if self.config.get("debug_mode", False):
            dataset["train"] = dataset["train"].select(range(min(1000, len(dataset['train'])))) # Evitar error si es pequeño
            dataset["test"] = dataset["test"].select(range(min(200, len(dataset['test']))))
            self.logger.info("Modo debug: usando subconjunto de datos")

        code_col = self.config.get("code_column_name", "sentences")
        explanation_col = self.config.get("explanation_column_name", "Explanation")
        
        prompt_template_train = self.config.get("prompt_template_train",
            "<|system|>\\nYou are a programming expert who explains code clearly and concisely.\\n<|user|>\\nExplain this code:\\n{code}\\n<|assistant|>\\n{explanation}<|end|>"
        )

        def preprocess_function(examples):
            inputs = []
            for code, explanation in zip(examples[code_col], examples[explanation_col]):
                # Usar el prompt template para entrenamiento
                prompt = prompt_template_train.format(code=code, explanation=explanation)
                inputs.append(prompt)
            
            tokenized = self.tokenizer(
                inputs,
                padding=False, 
                truncation=True,
                max_length=self.config["max_length"],
                return_tensors=None,
            )

            return tokenized

        num_proc_map = 1 # O None, o 0, para forzar la ejecución secuencial
        self.logger.info(f"Usando num_proc={num_proc_map} para datasets.map() para evitar problemas de CUDA con fork.")
        
        train_dataset = dataset["train"].map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=num_proc_map,
            desc="Preprocessing train data"
        )
        eval_dataset = dataset["test"].map(
            preprocess_function, # La evaluación usa el mismo formato para calcular la loss de validación
            batched=True,
            remove_columns=dataset["test"].column_names,
            num_proc=num_proc_map,
            desc="Preprocessing eval data"
        )
        return train_dataset, eval_dataset
    
    def setup_model_and_tokenizer(self):
        model_id = self.config["model_id"]
        self.logger.info(f"Cargando tokenizer {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Solo usar bnb_config si no es CPU
        bnb_config_val = None
        model_dtype = torch.float32  # Valor por defecto para CPU
        if self.device.type == 'cuda':
            bnb_config_val = create_bnb_config()
            model_dtype = torch.float16
        else:
            self.logger.warning("BitsAndBytesConfig no se usará en CPU.")

        self.logger.info(f"Cargando modelo {model_id}...")
        attn_implementation_config = None 
        use_flash_attention_from_config = self.config.get("use_flash_attention", True)

        if self.device.type == 'cuda':
            if use_flash_attention_from_config:
                if is_flash_attn_2_available():
                    gpu_capability = torch.cuda.get_device_capability()
                    if gpu_capability[0] >= 8: # Ampere o más nuevo
                        attn_implementation_config = "flash_attention_2"
                        self.logger.info("Flash Attention 2 habilitado (GPU Cap: {gpu_capability[0]}.{gpu_capability[1]}).")
                    else: # GPU < 8.0 (ej. T4)
                        attn_implementation_config = "sdpa"
                        self.logger.info("Flash Attention 2 solicitado pero GPU (Cap: {gpu_capability[0]}.{gpu_capability[1]}) no es >= Ampere. Usando 'sdpa'.")
                else: # Flash Attention 2 no está disponible
                    attn_implementation_config = "sdpa"
                    self.logger.info("Flash Attention 2 solicitado pero no disponible. Usando 'sdpa'.")
            else: # use_flash_attention es False en config
                attn_implementation_config = "sdpa" 
                self.logger.info("Flash Attention no solicitado en config. Usando 'sdpa'.")
        else: # CPU
            self.logger.info("Entrenamiento en CPU, no se especifica attn_implementation para CUDA.")
        # --- Fin Lógica de Attn_implementation ---
                
        # Device map: "auto" para single GPU/CPU, {"": self.device} para DDP
        device_map_config = "auto"
        if dist.is_initialized(): # DDP
            device_map_config = {"": self.device}
        elif self.device.type == 'cpu': # CPU
            device_map_config = {"": "cpu"}
        # else: "auto" para single GPU
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config_val, # Será None si es CPU
            device_map=device_map_config,
            torch_dtype=model_dtype, # Usará float16 en CUDA, float32 en CPU
            trust_remote_code=True,
            attn_implementation=attn_implementation_config
        )
        
        if self.device.type == 'cuda': # PEFT y kbit training solo para GPU
            peft_config = LoraConfig(
                r=self.config["lora_r"],
                lora_alpha=self.config["lora_alpha"],
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=self.config["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
                modules_to_save=["lm_head"] 
            )
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
            self.model = get_peft_model(self.model, peft_config)
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.logger.info(f"Parámetros entrenables (PEFT):")
            self.model.print_trainable_parameters()
        else:
            self.logger.info("PEFT/LoRA no aplicado para entrenamiento en CPU.")

    def create_dataloaders(self, train_dataset, eval_dataset):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # DistributedSampler para DDP
        train_sampler = None
        eval_sampler = None
        shuffle_train = True # True para single GPU/CPU, False para DDP (sampler se encarga)

        if dist.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, 
                num_replicas=dist.get_world_size(), 
                rank=dist.get_rank(), 
                shuffle=True # Shuffle dentro del sampler
            )
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset, 
                num_replicas=dist.get_world_size(), 
                rank=dist.get_rank(), 
                shuffle=False
            )
            shuffle_train = False # El sampler ya lo hace

        self.train_dataloader = DataLoader(
            train_dataset, 
            shuffle=shuffle_train, 
            batch_size=self.config["batch_size"],
            collate_fn=data_collator,
            num_workers=self.config.get("num_dataloader_workers", 2),
            pin_memory=True,
            persistent_workers=True if self.config.get("num_dataloader_workers", 2) > 0 else False,
            sampler=train_sampler
        )
        self.eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=self.config["eval_batch_size"],
            collate_fn=data_collator,
            num_workers=self.config.get("num_dataloader_workers", 2),
            pin_memory=True,
            persistent_workers=True if self.config.get("num_dataloader_workers", 2) > 0 else False,
            sampler=eval_sampler
        )
        
    def setup_training_components(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
            eps=1e-8,
            betas=(0.9, 0.95)
        )
        total_steps = len(self.train_dataloader) * self.config["num_epochs"]
        # Ajustar total_steps para gradient accumulation si no lo hace DataLoader
        total_steps //= self.config.get("gradient_accumulation_steps", 1)

        warmup_steps = int(total_steps * self.config["warmup_ratio"])
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return optimizer, lr_scheduler
    
    def train_epoch(self, model, dataloader, optimizer, lr_scheduler, epoch):
        model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        if hasattr(dataloader.sampler, 'set_epoch') and dist.is_initialized():
            dataloader.sampler.set_epoch(epoch)

        current_rank = dist.get_rank() if dist.is_initialized() else 0
        
        progress_bar_enum = enumerate(dataloader)
        if current_rank == 0:
            progress_bar = tqdm(progress_bar_enum, total=num_batches, desc=f"Epoch {epoch+1}")
        else:
            progress_bar = progress_bar_enum
        
        for step, batch in progress_bar:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Mixed precision solo si es CUDA
            use_amp = self.device.type == 'cuda'
            autocast_dtype = torch.float16 if use_amp else torch.float32 # dtype for autocast
            current_device_type = self.device.type 
            
            # Ensure 'mps' uses 'cpu' for autocast if problematic, otherwise use device type
            if current_device_type == 'mps': # Specific check for MPS
                 # Autocast for MPS might still be under development or have limitations.
                 # Forcing CPU or disabling AMP might be safer if issues arise.
                 # However, PyTorch aims to support it. Let's try to use it if not 'cuda'.
                 # If AMP is not desired/problematic for MPS, `use_amp` should be false.
                 # For now, let's assume if not 'cuda', it's 'cpu' for AMP logic if 'use_amp' is based on 'cuda'.
                 # The `use_amp` flag already controls if AMP is used.
                 pass


            # CORREGIDO: Usar torch.amp.autocast y la firma correcta
            # Primer argumento es device_type (string)
            # Siguientes son keyword arguments: dtype, enabled, cache_enabled
            with torch.amp.autocast(device_type=current_device_type, dtype=autocast_dtype, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
                if self.config.get("gradient_accumulation_steps", 1) > 1:
                    loss = loss / self.config["gradient_accumulation_steps"]
            
            loss.backward()
            
            if (step + 1) % self.config.get("gradient_accumulation_steps", 1) == 0:
                if self.device.type == 'cuda': # Gradient clipping solo para CUDA usualmente
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.detach().float()
            
            if current_rank == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item() / (step + 1):.4f}",
                    'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })
            
            if step % 50 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
        
        # Promediar la loss entre todos los procesos DDP
        if dist.is_initialized():
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item() / dist.get_world_size() # loss total global
        
        return total_loss / num_batches # loss promedio por batch

    def evaluate(self, model, dataloader): # Rank no es necesario, DDP sampler se encarga
        model.eval()
        total_loss = 0
        num_batches = len(dataloader)
        current_rank = dist.get_rank() if dist.is_initialized() else 0

        with torch.no_grad():
            eval_iterator_enum = enumerate(dataloader)
            if current_rank == 0:
                eval_iterator = tqdm(eval_iterator_enum, total=num_batches, desc="Evaluando")
            else:
                eval_iterator = eval_iterator_enum
                
            for _, batch in eval_iterator: # Renombrar step a _ para evitar confusión
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                
                use_amp = self.device.type == 'cuda'
                autocast_dtype = torch.float16 if use_amp else torch.float32
                current_device_type = self.device.type

                # CORREGIDO: Usar torch.amp.autocast y la firma correcta
                with torch.amp.autocast(device_type=current_device_type, dtype=autocast_dtype, enabled=use_amp):
                    outputs = model(**batch)
                    total_loss += outputs.loss.detach().float() # Acumular loss del rank
        
        # Promediar la loss de evaluación entre todos los procesos DDP
        if dist.is_initialized():
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM) # Suma de losses de todos los ranks
            # Ensure correct averaging for global evaluation loss
            # num_batches is per rank. Total batches processed approx num_batches * world_size
            # total_loss_tensor is sum of losses from all ranks
            # So, avg_loss_per_batch_global = total_loss_tensor / (num_batches * dist.get_world_size())
            # However, if DDP sampler ensures each sample is seen once across all ranks,
            # and num_batches = ceil(len(dataset_subset_for_rank) / batch_size),
            # then the sum of losses (total_loss_tensor) divided by total number of samples, then multiplied by batch_size,
            # or simply, if each rank computes its average loss, then average these averages.
            # The current calculation returns average loss per batch based on rank 0's number of batches.
            # A more robust way might be total_loss_tensor / (sum of batches processed across all ranks)
            # For now, let's assume num_batches is similar across ranks due to DistributedSampler.
            # The prior code had: avg_loss_per_rank = total_loss_tensor.item() / (num_batches * dist.get_world_size())
            # This seems correct if total_loss was sum of batch losses.
            # Since total_loss here is sum of batch losses for *one rank*,
            # total_loss_tensor will be sum of (sum of batch_losses for each rank).
            # To get overall average batch loss: total_loss_tensor / (total_num_batches_across_all_ranks)
            # If len(dataloader) is already adjusted for DDP (i.e., number of batches for this rank),
            # then sum of all batches processed is len(dataloader.dataset) / batch_size_per_rank.
            # total_loss_tensor is the sum of all losses.
            # len(dataloader.dataset) is the total number of evaluation samples.
            # Loss per sample = total_loss_tensor / len(dataloader.dataset)
            # Average loss per batch (comparable to training) = (total_loss_tensor / len(dataloader.dataset)) * effective_batch_size
            # Let's stick to average loss of rank 0 view after all_reduce, as it's a common practice for logging.
            # The calculation `total_loss_tensor.item() / (num_batches * dist.get_world_size())` was previously:
            # `avg_loss_per_rank = total_loss_tensor.item() / (num_batches * dist.get_world_size())`
            # This is average loss per batch, where num_batches is per-rank. This is correct.
            return total_loss_tensor.item() / (num_batches * dist.get_world_size())

        return total_loss / num_batches # Promedio de loss por batch para single GPU/CPU

    def save_model(self, epoch=None, final=False):
        # Solo el rank 0 guarda el modelo
        if dist.is_initialized() and dist.get_rank() != 0:
            return None

        if final:
            output_path = self.config["output_dir"]
        else:
            output_path = os.path.join(self.config["output_dir"], f"checkpoint-epoch-{epoch+1}")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Si es DDP, guardar el model.module
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        
        model_to_save.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        config_path = os.path.join(output_path, "training_config.json")
        with open(config_path, "w") as f:
            # Filtrar tipos no serializables si es necesario, o convertir self.device
            serializable_config = self.config.copy()
            if 'local_rank' in serializable_config and isinstance(serializable_config['local_rank'], torch.device):
                 serializable_config['local_rank'] = str(serializable_config['local_rank'])
            json.dump(serializable_config, f, indent=2)
        
        self.logger.info(f"Modelo guardado en {output_path}")
        return output_path

class ModelComparator:
    def __init__(self, base_model_id: str, tuned_model_path: str, config: Dict[str, Any], device: Optional[str] = None):
        self.base_model_id = base_model_id
        self.tuned_model_path = tuned_model_path
        self.config = config # Guardar la config de comparación
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
        self.rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True) # Renombrado

    def load_models(self):
        self.logger.info("Cargando tokenizer...")
        tokenizer_path = self.config.get("tokenizer_path_for_comparison", self.tuned_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Solo usar bnb_config si es CUDA
        bnb_config_val_comp = None
        model_dtype_comp = torch.float32
        if self.device == 'cuda':
            # Usar float16 para inferencia en T4s también
            model_dtype_comp = torch.float16
            # Pasar el dtype correcto a bnb_config para la comparación
            bnb_config_val_comp = create_bnb_config(compute_dtype=model_dtype_comp) 
            self.logger.info("ModelComparator: Usando float16 y BNB compute_dtype=float16 para CUDA.")
        else:
            self.logger.warning("ModelComparator: BitsAndBytesConfig no se usará en CPU para ModelComparator.")
            
        self.logger.info(f"Cargando modelo base {self.base_model_id} en {self.device}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config_val_comp,
            device_map=self.device if self.device != "cpu" else {"": "cpu"}, # More robust device_map
            torch_dtype=model_dtype_comp,
            trust_remote_code=True
        )
        
        self.logger.info(f"Cargando modelo fine-tuned desde {self.tuned_model_path} en {self.device}...")
        
        # For PeftModel, the base model should be loaded fresh for applying adapters,
        # especially if quantization or dtype might differ or if base_model above was modified.
        base_for_peft_config = {
            "quantization_config": bnb_config_val_comp,
            "torch_dtype": model_dtype_comp,
            "trust_remote_code": True
        }
        # If CPU, device_map must be explicit for from_pretrained for base of PEFT
        if self.device == "cpu":
            base_for_peft_config["device_map"] = {"": "cpu"}

        base_for_peft = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, 
            **base_for_peft_config
        )
        # If CUDA, move to device before PeftModel, device_map="auto" handles multi-GPU for Peft but here it's single device
        if self.device != "cpu":
             base_for_peft.to(self.device)

        self.tuned_model = PeftModel.from_pretrained(
            base_for_peft, 
            self.tuned_model_path,
            is_trainable=False
        )
        # Ensure tuned PEFT model is on the correct device after loading adapters
        if self.device != "cpu": # PeftModel.from_pretrained might not move all parts if base is already on device
            self.tuned_model.to(self.device) 
        
        self.base_model.eval()
        self.tuned_model.eval()
        self.logger.info("Modelos cargados y en modo evaluación.")
        
    def generate_response(self, model, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.config.get("max_length", 1024) - max_new_tokens) # Dejar espacio
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        use_amp = self.device == 'cuda' # self.device es una cadena aquí "cuda" o "cpu"
        autocast_dtype_comp = torch.float16 if use_amp else torch.float32
        current_device_type_str = self.device # es "cuda" o "cpu"
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
                    eos_token_id=self.tokenizer.eos_token_id # Para que pare en EOS
                )
        
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()
    
    def calculate_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        metrics = {}
        rouge_scores_agg = {'rouge1': [], 'rouge2': [], 'rougeL': []} # Renombrado
        for pred, ref in zip(predictions, references):
            # Asegurar que pred y ref no sean vacíos, ROUGE puede fallar
            if not pred.strip() or not ref.strip():
                self.logger.warning(f"Predicción o referencia vacía encontrada. Pred: '{pred}', Ref: '{ref}'. Saltando para ROUGE.")
                # Asignar score 0 o manejar como prefieras
                for key_rouge in rouge_scores_agg:
                    rouge_scores_agg[key_rouge].append(0.0)
                continue

            scores = self.rouge_scorer_obj.score(ref, pred)
            for key_rouge in rouge_scores_agg:
                rouge_scores_agg[key_rouge].append(scores[key_rouge].fmeasure)
        
        for key_rouge in rouge_scores_agg:
            metrics[key_rouge] = np.mean(rouge_scores_agg[key_rouge]) if rouge_scores_agg[key_rouge] else 0.0
        
        try:
            # Filtrar pares vacíos para BERTScore también
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
            # Ensure indices are unique and within bounds
            if len(test_dataset) < num_samples: num_samples = len(test_dataset)
            indices = np.random.choice(len(test_dataset), num_samples, replace=False)
            test_samples = test_dataset.select(indices)
        else:
            test_samples = test_dataset
            
        base_predictions, tuned_predictions, references, prompts_used = [], [], [], []

        code_col = self.config.get("code_column_name", "sentences")
        explanation_col = self.config.get("explanation_column_name", "Explanation")
        
        # Prompt para inferencia (solo pide la explicación)
        prompt_template_inference = self.config.get("prompt_template_inference",
            "<|system|>\\nYou are a programming expert who explains code clearly and concisely.\\n<|user|>\\nExplain this code:\\n{code}\\n<|assistant|>\\n"
        )

        for i, example in enumerate(tqdm(test_samples, desc="Generando predicciones")):
            code = example[code_col]
            reference = example[explanation_col]
            
            # Usar el prompt_template_inference
            prompt = prompt_template_inference.format(code=code)
            
            try:
                base_pred = self.generate_response(self.base_model, prompt)
                tuned_pred = self.generate_response(self.tuned_model, prompt)
                
                base_predictions.append(base_pred)
                tuned_predictions.append(tuned_pred)
                references.append(reference)
                prompts_used.append(code) # Guardar el código original como "prompt" para el reporte
                
            except Exception as e:
                self.logger.warning(f"Error en muestra {i} (código: {code[:50]}...): {e}")
                # Añadir placeholders para mantener la alineación o saltar
                base_predictions.append("")
                tuned_predictions.append("")
                references.append(reference if reference else "") # Evitar None
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
                if abs(base_metrics[metric]) > 1e-9: # Evitar división por cero
                    improvement = ((tuned_metrics[metric] - base_metrics[metric]) / base_metrics[metric]) * 100
                    results['improvement'][metric] = improvement
        return results
    
    def generate_comparison_report(self, results: Dict[str, Any], save_path: Optional[str] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_path is None:
            report_dir = self.config.get("output_dir", ".") # Guardar reporte en output_dir
            os.makedirs(report_dir, exist_ok=True)
            save_path = os.path.join(report_dir, f"comparison_report_{timestamp}.html")

        # ... (resto del código de generate_comparison_report sin cambios, asegurando que use html.escape) ...
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Comparación de Modelos</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .metric-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .improvement-positive {{ color: green; font-weight: bold; }}
                .improvement-negative {{ color: red; font-weight: bold; }}
                .sample-box {{ background-color: #f9f9f9; padding: 15px; margin: 20px 0; border-radius: 5px; border: 1px solid #eee; }}
                .code-block {{ background-color: #272822; color: #f8f8f2; padding: 10px; border-radius: 3px; font-family: monospace; white-space: pre-wrap; word-wrap: break-word; }}
                h1, h2, h3, h4 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>Reporte de Comparación: Modelo Base vs Modelo Fine-tuned</h1>
            <p><strong>Fecha:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Modelo Base:</strong> {html.escape(self.base_model_id)}</p>
            <p><strong>Modelo Fine-tuned:</strong> {html.escape(self.tuned_model_path)}</p>
            
            <h2>Resumen de Métricas</h2>
            <table class="metric-table">
                <tr>
                    <th>Métrica</th>
                    <th>Modelo Base</th>
                    <th>Modelo Fine-tuned</th>
                    <th>Mejora (%)</th>
                </tr>
        """
        
        base_metrics = results['base_model_metrics'] 
        tuned_metrics = results['tuned_model_metrics']
        improvements = results['improvement']
        
        metric_display_order = ['rouge1', 'rouge2', 'rougeL', 'bert_score_f1', 'bert_score_precision', 'bert_score_recall', 'avg_pred_length', 'avg_ref_length', 'length_ratio']
        
        for metric in metric_display_order:
            if metric not in base_metrics: continue # Saltar si la métrica no existe

            if isinstance(base_metrics[metric], (int, float)):
                base_val = f"{base_metrics[metric]:.4f}"
                tuned_val = f"{tuned_metrics[metric]:.4f}"
                
                improvement_str = "N/A"
                if metric in improvements:
                    improvement = improvements[metric]
                    improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
                    improvement_str = f'<span class="{improvement_class}">{improvement:+.2f}%</span>'
                
                html_content += f"""
                <tr>
                    <td>{html.escape(metric.replace('_', ' ').title())}</td>
                    <td>{base_val}</td>
                    <td>{tuned_val}</td>
                    <td>{improvement_str}</td>
                </tr>
                """
        
        html_content += """
            </table>
            
            <h2>Ejemplos de Comparación (hasta 10)</h2>
        """
        
        samples = results['samples']
        num_display_samples = min(10, len(samples.get('prompts', [])))

        for i in range(num_display_samples):
            safe_prompt = html.escape(samples['prompts'][i] if samples['prompts'][i] else "N/A")
            safe_reference = html.escape(samples['references'][i] if samples['references'][i] else "N/A")
            safe_base_pred = html.escape(samples['base_predictions'][i] if samples['base_predictions'][i] else "N/A")
            safe_tuned_pred = html.escape(samples['tuned_predictions'][i] if samples['tuned_predictions'][i] else "N/A")
            
            html_content += f"""
            <div class="sample-box">
                <h3>Ejemplo {i+1}</h3>
                <h4>Código (Prompt para el modelo):</h4>
                <div class="code-block">{safe_prompt}</div>
                
                <h4>Explicación Esperada (Referencia):</h4>
                <p>{safe_reference}</p>
                
                <h4>Respuesta del Modelo Base:</h4>
                <p>{safe_base_pred}</p>
                
                <h4>Respuesta del Modelo Fine-tuned:</h4>
                <p>{safe_tuned_pred}</p>
            </div>
            """
        
        if num_display_samples == 0:
            html_content += "<p>No hay ejemplos para mostrar (posiblemente debido a errores durante la generación o un conjunto de prueba vacío).</p>"

        html_content += """
            </body>
            </html>
        """
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Reporte de comparación guardado en: {save_path}")
        return save_path

    def interactive_comparison(self):
        self.logger.info("Iniciando comparación interactiva...")
        self.logger.info("Ingresa código para comparar las explicaciones (escribe 'quit' para salir)")
        
        prompt_template_inference = self.config.get("prompt_template_inference",
            "<|system|>\\nYou are a programming expert who explains code clearly and concisely.\\n<|user|>\\nExplain this code:\\n{code}\\n<|assistant|>\\n"
        )

        while True:
            try:
                code_input = input("\n>>> Ingresa el código a explicar: ") # Renombrado para evitar conflicto
                if code_input.lower() in ['quit', 'exit', 'salir']:
                    break
                if not code_input.strip():
                    self.logger.info("Por favor, ingresa algún código.")
                    continue

                prompt = prompt_template_inference.format(code=code_input)
                
                print("\n" + "="*80)
                print("🔹 MODELO BASE:")
                print("-"*40)
                base_response = self.generate_response(self.base_model, prompt)
                print(base_response)
                
                print("\n🔸 MODELO FINE-TUNED:")
                print("-"*40)
                tuned_response = self.generate_response(self.tuned_model, prompt)
                print(tuned_response)
                print("="*80)
                
            except KeyboardInterrupt:
                self.logger.info("Interrupción por teclado detectada.")
                break
            except Exception as e:
                self.logger.error(f"Error durante comparación: {e}", exc_info=True)
                continue
        self.logger.info("Comparación interactiva finalizada.")

def run_model_comparison(base_model_id: str, tuned_model_path: str, comparison_config: Optional[Dict[str, Any]] = None):
    # Obtener logger del módulo actual
    current_logger = logging.getLogger(__name__)
    current_logger.info("Iniciando comparación de modelos...") # Cambiado a current_logger
    
    if comparison_config is None:
        comparison_config = {}

    dataset_name = comparison_config.get("dataset_name", "salony/code_explanations")
    try:
        dataset = load_dataset(dataset_name, trust_remote_code=True) # Añadir trust_remote_code
        test_dataset = dataset["test"]
    except Exception as e:
        current_logger.error(f"No se pudo cargar el dataset {dataset_name}: {e}")
        return None, None # Retornar None si falla la carga

    # Pasar la config completa al comparador
    comparator = ModelComparator(
        base_model_id, 
        tuned_model_path,
        config=comparison_config, # Pasar la config
        device=comparison_config.get("comparison_device", None)
    )
    # comparator.config ya se establece en el __init__ de ModelComparator
    comparator.load_models()
    
    results = comparator.evaluate_on_test_set(test_dataset, num_samples=comparison_config.get("num_comparison_samples", 100))
    
    report_path = comparator.generate_comparison_report(results)
    
    current_logger.info("\n" + "="*60)
    current_logger.info("RESUMEN DE COMPARACIÓN")
    current_logger.info("="*60)
    
    base_metrics = results['base_model_metrics']
    tuned_metrics = results['tuned_model_metrics']
    improvements = results['improvement']
    
    for metric in ['rouge1', 'rouge2', 'rougeL', 'bert_score_f1']:
        if metric in base_metrics and metric in tuned_metrics and metric in improvements: # Chequear existencia
            base_val = base_metrics[metric]
            tuned_val = tuned_metrics[metric]
            improvement = improvements[metric]
            current_logger.info(f"{metric.upper()}: {base_val:.4f} → {tuned_val:.4f} ({improvement:+.2f}%)")
        elif metric in base_metrics and metric in tuned_metrics: # Si no hay mejora calculada
             current_logger.info(f"{metric.upper()}: {base_metrics[metric]:.4f} → {tuned_metrics[metric]:.4f} (Mejora N/A)")


    current_logger.info(f"\n📊 Reporte completo guardado en: {report_path}")
    
    if comparison_config.get("interactive_comparison_enabled", True): # Permitir deshabilitar
        try:
            # Solo pedir input si es interactivo, no en un script DDP por ejemplo
            if os.isatty(0) : # Chequea si stdin es un tty
                interactive_input = input("\n¿Deseas hacer comparación interactiva? (y/n): ").lower().strip()
                if interactive_input in ['y', 'yes', 'sí', 's']:
                    comparator.interactive_comparison()
            else:
                current_logger.info("Modo no interactivo, saltando prompt de comparación interactiva.")
        except Exception as e:
            current_logger.warning(f"No se pudo iniciar la comparación interactiva: {e}")
            pass
    return results, report_path

# Esta es la función que mp.spawn llamará
def train_distributed(rank, world_size, config):
    # Configurar logging para este proceso hijo
    # Puedes pasar el nivel de logging via config si quieres
    log_level = config.get("log_level", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                        format=f'%(asctime)s - RANK {rank} - %(name)s - %(levelname)s - %(message)s')
    
    process_logger = logging.getLogger(__name__) # Logger para este proceso
    process_logger.info(f"Proceso rank {rank}/{world_size} iniciado.")

    set_seed(config.get("seed", 42) + rank)
    trainer = CodeExplainerTrainer(config) # config ya tiene local_rank = -1 o GPU_ID
    trainer.setup_distributed(rank, world_size) # Esto sobreescribirá device y local_rank
    
    try:
        process_logger.info(f"Rank {rank}: Configurando modelo y tokenizer...")
        trainer.setup_model_and_tokenizer() # Primero inicializar el tokenizer y el modelols
        
        process_logger.info(f"Rank {rank}: Cargando y preprocesando datos...")
        train_dataset, eval_dataset = trainer.load_and_preprocess_data()
        
        process_logger.info(f"Rank {rank}: Creando DataLoaders...")
        trainer.create_dataloaders(train_dataset, eval_dataset)
        
        # DDP se aplica al modelo aquí. El modelo ya está en trainer.device por setup_model_and_tokenizer
        # y el device_map {"": trainer.device}
        process_logger.info(f"Rank {rank}: Configurando DDP para el modelo...")
        trainer.model = DDP(trainer.model, device_ids=[rank], find_unused_parameters=config.get("ddp_find_unused_parameters", False))

        process_logger.info(f"Rank {rank}: Configurando componentes de entrenamiento...")
        optimizer, lr_scheduler = trainer.setup_training_components()
        
        if rank == 0:
            process_logger.info("Comenzando entrenamiento distribuido...")
            
        for epoch in range(config["num_epochs"]):
            process_logger.info(f"Rank {rank}: Iniciando Época {epoch+1}...")
            train_loss = trainer.train_epoch(trainer.model, trainer.train_dataloader, optimizer, lr_scheduler, epoch)
            
            # La loss ya está promediada globalmente si DDP está activo
            if rank == 0: 
                process_logger.info(f"Época {epoch+1} completada. Loss de entrenamiento (global avg): {train_loss:.4f}")

            # Sincronizar todos los procesos antes de la evaluación si es necesario,
            # aunque con DDP y DistributedSampler, cada proceso trabaja en su subconjunto.
            if dist.is_initialized():
                dist.barrier()

            process_logger.info(f"Rank {rank}: Iniciando evaluación para Época {epoch+1}...")
            eval_loss = trainer.evaluate(trainer.model.module, trainer.eval_dataloader) 
            
            if rank == 0:
                process_logger.info(f"Evaluación Época {epoch+1} (global avg): Loss de validación: {eval_loss:.4f}")
                trainer.save_model(epoch=epoch)
            
            if dist.is_initialized():
                dist.barrier() # Sincronizar antes de la siguiente época o guardado final
        
        if rank == 0:
            final_model_path = trainer.save_model(final=True)
            process_logger.info(f"Entrenamiento completado! Modelo final en {final_model_path}")
            
            if config.get("run_comparison", True) and final_model_path:
                try:
                    process_logger.info("\n" + "="*60 + "\nINICIANDO COMPARACIÓN DE MODELOS (desde rank 0)\n" + "="*60)
                    comparison_cfg = {
                        "dataset_name": config.get("dataset_name", "salony/code_explanations"),
                        "code_column_name": config.get("code_column_name", "sentences"),
                        "explanation_column_name": config.get("explanation_column_name", "Explanation"),
                        "prompt_template_inference": config.get("prompt_template_inference"),
                        "tokenizer_path_for_comparison": final_model_path, 
                        "num_comparison_samples": config.get("num_comparison_samples", 100),
                        "max_length": config.get("max_length", 1024),
                        "output_dir": config.get("output_dir"), 
                        "interactive_comparison_enabled": config.get("interactive_comparison_enabled", True) if rank == 0 and os.isatty(0) else False,
                        "comparison_device": "cuda:0" if torch.cuda.is_available() else "cpu" # Especificar device para comparación
                    }
                    # Llamar a la función directamente ya que está en el mismo módulo
                    run_model_comparison(
                        base_model_id=config["model_id"],
                        tuned_model_path=final_model_path,
                        comparison_config=comparison_cfg
                    )
                    process_logger.info("Comparación completada exitosamente!")
                except Exception as e:
                    process_logger.error(f"Error durante la comparación (rank 0): {e}", exc_info=True)
    except Exception as e:
        process_logger.error(f"Error catastrófico en rank {rank}: {e}", exc_info=True)
    finally:
        process_logger.info(f"Rank {rank}: Limpiando procesos distribuidos...")
        trainer.cleanup_distributed()
        process_logger.info(f"Proceso rank {rank} finalizado.")