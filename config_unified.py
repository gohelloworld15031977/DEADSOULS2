# Единая конфигурация проекта DeadSouls

# ==================== МОДЕЛИ ====================
# Единственная поддерживаемая модель - русскоязычный GPT-2
MODELS = {
    "gpt2": {
        "name": "ai-forever/rugpt3small_based_on_gpt2",
        "type": "gpt2",
        "vocab_size": 32000,
        "max_length": 256,
        "target_modules": ["c_attn", "c_proj", "c_fc"],
        "small_model": True
    }
}

# Текущая модель для обучения (единственная)
DEFAULT_MODEL = "gpt2"

# ==================== ПУТИ ====================
DATA_DIR = "data"
OUTPUT_DIR = "./gogol_finetuned_final"
CHECKPOINT_DIR = "./checkpoints"
LOGS_DIR = "./logs"
VISUALIZATION_DIR = "./visualization"

# Пути к данным
DATASETS = {
    "gogol_only": "data/tokenized_gpt2",
    "extended": "data/extended_tokenized",
    "processed_text": "data/gogol_processed.txt",
    "combined_text": "data/combined_dataset.txt"
}

# ==================== ГИПЕРПАРАМЕТРЫ ОБУЧЕНИЯ ====================
TRAINING = {
    # Базовые параметры
    "num_train_epochs": 7,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 4,
    
    # Оптимизатор
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 0.5,
    "warmup_steps": 100,
    "lr_scheduler_type": "linear",
    
    # Precision
    "fp16": False,  # True для CUDA
    "bf16": False,  # True для новых GPU
    
    # LoRA параметры для GPT-2
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    
    # Квантование не требуется для GPT-2
    "load_in_4bit": False,
    "load_in_8bit": False,
    
    # Стратегии
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_steps": 20,
    "save_total_limit": 2,
    
    # Early stopping
    "enable_early_stopping": True,
    "early_stopping_patience": 2,
    "early_stopping_threshold": 0.01,
    
    # Метрики
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False
}

# Параметры для GPU-обучения (переопределяют базовые)
GPU_TRAINING = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "fp16": True,
    "learning_rate": 2e-4,
    "num_train_epochs": 3
}

# ==================== ГЕНЕРАЦИЯ ТЕКСТА ====================
GENERATION = {
    "max_length": 100,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.2,
    "num_return_sequences": 1,
    "do_sample": True,
    "pad_token_id": None,  # Будет установлен из токенизатора
    "eos_token_id": None   # Будет установлено из токенизатора
}

# Промпты для тестовой генерации
TEST_PROMPTS = [
    "Чичиков приехал в город",
    "В губернском городе N",
    "Однажды вечером сидел я",
    "Россия, куда ж несешься ты",
    "Мертвые души - это"
]

# ==================== ВАЛИДАЦИЯ ДАТАСЕТА ====================
VALIDATION = {
    "min_paragraph_length": 50,
    "max_paragraph_length": 1000,
    "max_sequence_length": 1024,
    "min_sequence_length": 10,
    "max_allowed_problematic_examples": 100
}

# ==================== МОНИТОРИНГ ====================
MONITORING = {
    # TensorBoard
    "enable_tensorboard": True,
    "tensorboard_log_dir": "logs/tensorboard",
    
    # Weights & Biases (опционально)
    "enable_wandb": False,
    "wandb_project": "deadsouls-gogol",
    "wandb_entity": None,
    
    # Логирование в файл
    "enable_file_logging": True,
    "log_file": "logs/training.log"
}

# ==================== РЕСУРСЫ ====================
RESOURCES = {
    "use_gpu": True,  # Автоопределение
    "device": None,   # Будет установлено автоматически
    "max_memory": None,  # Для multi-GPU
    "cpu_offload": False
}

# ==================== ЭКСПЕРИМЕНТЫ ====================
EXPERIMENTS = {
    "experiment_name": "gogol_style_transfer",
    "run_id": None,  # Будет сгенерировано автоматически
    "git_commit": None,  # Будет установлено автоматически
    "notes": ""
}

# ==================== ДОПОЛНИТЕЛЬНО ====================
def get_model_config(model_key=None):
    """Получить конфигурацию модели"""
    key = model_key or DEFAULT_MODEL
    return MODELS.get(key, MODELS[DEFAULT_MODEL])

def get_training_config(use_gpu=False):
    """Получить конфигурацию обучения"""
    config = TRAINING.copy()
    if use_gpu:
        config.update(GPU_TRAINING)
    return config

# Автоматическое определение устройства
import torch
if RESOURCES["use_gpu"] and torch.cuda.is_available():
    RESOURCES["device"] = "cuda"
    print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
else:
    RESOURCES["device"] = "cpu"
    print("Используется CPU")
