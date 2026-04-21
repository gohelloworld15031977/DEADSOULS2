#!/usr/bin/env python3
# Скрипт обучения с мониторингом validation loss и early stopping
import torch
import os
import json
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

class EarlyStoppingCallback:
    """Callback для ранней остановки при переобучении"""
    def __init__(self, patience=1, min_delta=0.03):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        
    def on_epoch_end(self, trainer, logs):
        current_loss = logs.get("eval_loss", None)
        if current_loss is None:
            return
            
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss > self.best_loss * (1 - self.min_delta):
            # Loss не улучшился достаточно
            self.counter += 1
            print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered!")
        else:
            # Loss улучшился
            self.best_loss = current_loss
            self.counter = 0

def main():
    # Проверка доступности GPU
    device = "cuda" if torch.cuda.is_available() and USE_GPU else "cpu"
    print(f"Используется устройство: {device}")
    
    # Загрузка датасета
    print("Загрузка датасета...")
    dataset_path = "./data/tokenized_dataset"
    if not os.path.exists(dataset_path):
        print("Ошибка: датасет не найден. Сначала запустите prepare_dataset.py")
        return
        
    dataset = load_from_disk(dataset_path)
    print(f"Датасет загружен: {len(dataset)} примеров")
    
    # Разделение на train/validation (80/20)
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            train_data = dataset["train"]
        else:
            train_data = dataset[list(dataset.keys())[0]]
        split_dataset = train_data.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    
    print(f"Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
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
    print(f"Загрузка модели {MODEL_NAME}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Попытка загрузки без квантования...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Подготовка модели для k-bit обучения
    if LOAD_IN_4BIT:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA конфигурация
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["c_attn", "c_proj", "c_fc"],  # для GPT-2
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
    
    # Создаем callback для ранней остановки
    early_stopping = EarlyStoppingCallback(patience=1, min_delta=0.03)
    
    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        eval_strategy="epoch",  # Оценка после каждой эпохи
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        fp16=device == "cuda",
        bf16=False,
        max_grad_norm=0.3,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none",
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,  # Сохранять только 2 лучшие модели
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Добавляем callback для мониторинга
    class ValidationMonitor(TrainerCallback):  # type: ignore[name-defined]
        def __init__(self):
            self.epoch_losses = []
            
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.epoch is not None:
                print(f"\n=== Эпоха {state.epoch:.1f} завершена ===")
                if hasattr(state, "eval_loss"):
                    loss = state.eval_loss
                    self.epoch_losses.append(loss)
                    print(f"Validation loss: {loss:.4f}")
                    
                    # Проверка улучшения
                    if len(self.epoch_losses) > 1:
                        prev_loss = self.epoch_losses[-2]
                        improvement = (prev_loss - loss) / prev_loss * 100
                        print(f"Улучшение: {improvement:.2f}%")
                        
                        if improvement < 3:  # Меньше 3% улучшения
                            print("Предупреждение: незначительное улучшение")
                        else:
                            print("Хорошее улучшение, продолжаем обучение")
    
    monitor = ValidationMonitor()
    trainer.add_callback(monitor)
    
    # Запуск обучения
    print(f"\n{'='*50}")
    print(f"Начало обучения на {EPOCHS} эпох")
    print(f"Размер train датасета: {len(train_dataset)}")
    print(f"Размер validation датасета: {len(eval_dataset)}")
    print(f"Эффективный batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"LoRA rank: {LORA_R}")
    print(f"{'='*50}\n")
    
    try:
        trainer.train()
        
        # Сохранение финальной модели
        print("\nСохранение модели...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Сохранение метрик обучения
        metrics = {
            "epochs_trained": EPOCHS,
            "final_train_loss": trainer.state.log_history[-1]["loss"] if trainer.state.log_history else None,
            "final_eval_loss": trainer.state.log_history[-1]["eval_loss"] if trainer.state.log_history else None,
            "dataset_size": len(dataset),
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
        }
        
        with open(os.path.join(OUTPUT_DIR, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Модель сохранена в {OUTPUT_DIR}")
        print(f"Метрики обучения сохранены в {OUTPUT_DIR}/training_metrics.json")
        
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем")
    except Exception as e:
        print(f"\nОшибка во время обучения: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()