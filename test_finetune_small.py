#!/usr/bin/env python3
# Тестовый скрипт для дообучения с меньшей моделью
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
import os

# Используем меньшую модель для теста
MODEL_NAME = "gpt2"  # Маленькая модель для теста
OUTPUT_DIR = "./test_finetuned"
DATA_PATH = "./data/tokenized_dataset"

# Параметры
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
EPOCHS = 1  # Только 1 эпоха для теста
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 2
MAX_LENGTH = 128

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используется устройство: {device}")

# Загрузка датасета
print("Загрузка датасета...")
dataset = load_from_disk(DATA_PATH)
print(f"Датасет загружен: {len(dataset)} примеров")

# Разделение на train/validation
if isinstance(dataset, DatasetDict):
    if "train" in dataset:
        train_data = dataset["train"]
    else:
        train_data = dataset[list(dataset.keys())[0]]
    split_dataset = train_data.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
else:
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Загрузка модели и токенизатора
print(f"Загрузка модели {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Используем CPU для теста
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=None if device == "cpu" else "auto",
    torch_dtype=torch.float32,
)

# LoRA конфигурация
lora_config = LoraConfig(  # type: ignore[call-arg]
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["c_attn", "c_proj", "c_fc"],  # для GPT2
    bias="none",
    task_type="CAUSAL_LM",
)

# Применяем LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Аргументы обучения
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    report_to="none",
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Запуск обучения
print("Начало обучения...")
trainer.train()

# Сохранение модели
print("Сохранение модели...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Модель сохранена в {OUTPUT_DIR}")