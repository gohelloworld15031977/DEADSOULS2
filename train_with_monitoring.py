#!/usr/bin/env python3
"""
Обучение модели с мониторингом (TensorBoard + логирование в файл).
Интеграция с TensorBoard для визуализации метрик в реальном времени.
"""

import os
import sys
import logging
import torch
from datetime import datetime
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

# Импорт конфигурации
from config_unified import (
    get_model_config,
    get_training_config,
    MONITORING,
    RESOURCES,
    DATASETS,
    OUTPUT_DIR
)

# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================
def setup_logging():
    """Настройка логирования в файл и консоль"""
    log_dir = MONITORING.get("log_file", "logs/training.log")
    Path(log_dir).parent.mkdir(parents=True, exist_ok=True)
    
    # Формат логов
    log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    
    # Логирование в файл
    file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Логирование в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Корневой логгер
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
def main():
    logger = setup_logging()
    logger.info("=== Запуск обучения с мониторингом ===")
    
    # Параметры
    model_key = "gpt2_small"  # или "llama8b"
    model_config = get_model_config(model_key)
    training_config = get_training_config(RESOURCES["device"] == "cuda")
    
    model_name = model_config["name"]
    max_length = model_config["max_length"]
    target_modules = model_config["target_modules"]
    
    dataset_path = DATASETS["gogol_only"]
    output_dir = OUTPUT_DIR
    
    logger.info(f"Модель: {model_name}")
    logger.info(f"Датасет: {dataset_path}")
    logger.info(f"Выходная директория: {output_dir}")
    
    # Проверка наличия датасета
    if not os.path.exists(dataset_path):
        logger.error(f"Датасет не найден: {dataset_path}")
        logger.info("Сначала запустите: python validate_dataset.py")
        return
    
    # ==================== ЗАГРУЗКА ДАННЫХ ====================
    logger.info("Загрузка датасета...")
    try:
        dataset = load_from_disk(dataset_path)
        
        # Разделение на train/test если нужно
        # DatasetDict уже содержит разделённые датасеты, Dataset нужно делить вручную
        from datasets import DatasetDict
        
        if isinstance(dataset, DatasetDict):
            # Датасет уже в формате DatasetDict
            logger.info("Датасет в формате DatasetDict")
            if "train" in dataset:
                train_dataset = dataset["train"]
            else:
                train_dataset = dataset[list(dataset.keys())[0]]
            
            if "test" in dataset:
                eval_dataset = dataset["test"]
            elif "validation" in dataset:
                eval_dataset = dataset["validation"]
            else:
                # Если нет test/validation, создаём через train_test_split
                logger.info("Создание eval датасета через train_test_split...")
                split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
        else:
            # Датасет — обычный Dataset, нужно разделить
            logger.info("Разделение датасета на train/test...")
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        
        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    except Exception as e:
        logger.error(f"Ошибка загрузки датасета: {e}")
        return
    
    # ==================== ЗАГРУЗКА МОДЕЛИ ====================
    logger.info(f"Загрузка модели {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Квантование для больших моделей
        if training_config.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if RESOURCES["device"] == "cuda" else torch.float32
            )
        
        logger.info(f"Модель загружена, параметров: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return
    
    # ==================== НАСТРОЙКА LoRA ====================
    logger.info("Настройка LoRA...")
    lora_config = LoraConfig(
        r=training_config["lora_r"],
        lora_alpha=training_config["lora_alpha"],
        lora_dropout=training_config["lora_dropout"],
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    logger.info(f"LoRA параметры:")
    model.print_trainable_parameters()
    
    # ==================== DATA COLLATOR ====================
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # ==================== МОНІТОРИНГ ====================
    # Настройка TensorBoard
    tensorboard_log_dir = None
    if MONITORING.get("enable_tensorboard"):
        tensorboard_log_dir = os.path.join(
            MONITORING["tensorboard_log_dir"],
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"TensorBoard логирование: {tensorboard_log_dir}")
    
    # ==================== АРГУМЕНТЫ ОБУЧЕНИЯ ====================
    logger.info("Настройка аргументов обучения...")
    
    # Определение device
    device = RESOURCES["device"]
    fp16 = training_config["fp16"] and device == "cuda"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        eval_strategy=training_config["eval_strategy"],
        save_strategy=training_config["save_strategy"],
        logging_steps=training_config["logging_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        fp16=fp16,
        max_grad_norm=training_config["max_grad_norm"],
        warmup_steps=training_config["warmup_steps"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        report_to="tensorboard" if MONITORING.get("enable_tensorboard") else "none",
        logging_dir=tensorboard_log_dir,
        save_total_limit=training_config["save_total_limit"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        push_to_hub=False,
    )
    
    # ==================== TRAINER ====================
    callbacks = []
    if training_config.get("enable_early_stopping"):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=training_config["early_stopping_patience"],
                early_stopping_threshold=training_config["early_stopping_threshold"]
            )
        )
        logger.info(f"Early stopping: patience={training_config['early_stopping_patience']}, "
                   f"threshold={training_config['early_stopping_threshold']}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # ==================== ЗАПУСК ОБУЧЕНИЯ ====================
    logger.info("\n" + "="*50)
    logger.info("НАЧАЛО ОБУЧЕНИЯ")
    logger.info("="*50 + "\n")
    
    try:
        trainer.train()
        
        logger.info("\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===\n")
        
        # Сохранение модели
        logger.info("Сохранение модели...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Модель сохранена в: {output_dir}")
        
        # Финальные метрики
        if trainer.state.log_history:
            final_log = trainer.state.log_history[-1]
            logger.info("\n=== ФИНАЛЬНЫЕ МЕТРИКИ ===")
            if "eval_loss" in final_log:
                logger.info(f"Validation loss: {final_log['eval_loss']:.4f}")
            if "loss" in final_log:
                logger.info(f"Train loss: {final_log['loss']:.4f}")
        
        # Тест генерации
        logger.info("\n=== ТЕСТ ГЕНЕРАЦИИ ===")
        test_prompts = [
            "Чичиков приехал в город",
            "В губернском городе N"
        ]
        
        model.eval()
        for prompt in test_prompts[:2]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"\nПромпт: {prompt}")
            logger.info(f"Генерация: {generated}")
        
    except Exception as e:
        logger.error(f"\nОШИБКА ВО ВРЕМЯ ОБУЧЕНИЯ: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
