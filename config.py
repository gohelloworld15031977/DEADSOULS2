

# config.py
MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"   # русскоязычный GPT-2
OUTPUT_DIR = "./gogol_finetuned_final"                # Совместимость с config_unified.py
DATA_PATH = "./data/gogol_processed.txt"              # Обработанный файл с разделенными абзацами

# Параметры дообучения (LoRA для GPT-2)
LORA_R = 8                     # rank для GPT-2
LORA_ALPHA = 16                # alpha для GPT-2
LORA_DROPOUT = 0.05
LEARNING_RATE = 1e-4
EPOCHS = 7                     # увеличено для лучшего усвоения стиля
BATCH_SIZE = 1                 # на GPU 8GB batch size 1
GRADIENT_ACCUMULATION = 4      # эффективный batch size = 4 (совместимость с config_unified.py)

# Устаревшие параметры (оставлены для обратной совместимости)
LORA_R_SMALL = 8               # дублирует LORA_R
LORA_ALPHA_SMALL = 16          # дублирует LORA_ALPHA
LEARNING_RATE_SMALL = 1e-4     # дублирует LEARNING_RATE
WARMUP_STEPS = 100             # шаги разогрева
WEIGHT_DECAY = 0.01            # весовая дека
MAX_GRAD_NORM = 0.5            # клиппинг градиента

# Квантование не требуется для GPT-2 (отключено)
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False
USE_BFLOAT16 = False

# Максимальная длина текста (токенов)
MAX_LENGTH = 256               # соответствует max_length в config_unified.py

# Использование GPU (True для GPU)
USE_GPU = True