# finetune.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk, DatasetDict
from config import *

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
print(f"Используется устройство: {device}")

# Конфигурация квантования для QLoRA
bnb_config = None
if LOAD_IN_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    print("Используется 4-битное квантование (QLoRA)")

# Загрузка модели и токенизатора
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Подготовка модели для k-bit обучения
if LOAD_IN_4BIT:
    model = prepare_model_for_kbit_training(model)

# LoRA конфигурация для GPT-2
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["c_attn", "c_proj", "c_fc"],  # для GPT-2
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Загрузка данных
loaded = load_from_disk("./data/tokenized_dataset")
# Если загружен DatasetDict, берём split "train"
if isinstance(loaded, DatasetDict) and "train" in loaded:
    dataset = loaded["train"]
else:
    dataset = loaded

# Убеждаемся, что dataset — это Dataset, а не DatasetDict
from datasets import Dataset
assert isinstance(dataset, Dataset), "dataset должен быть экземпляром Dataset"

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Аргументы тренировки
training_args = TrainingArguments(  # type: ignore[call-arg]
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=(device == "cuda"),  # использовать mixed precision на GPU
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
    gradient_checkpointing=True,  # экономия памяти
    optim="paged_adamw_8bit",     # оптимизатор для QLoRA
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

print("Начало обучения...")
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Модель сохранена в {OUTPUT_DIR}")