#!/usr/bin/env python3
# Тестовый пайплайн обучения с маленькой моделью для демонстрации
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import numpy as np

def create_test_dataset():
    """Создание тестового датасета"""
    texts = [
        "Чичиков приехал в город N.",
        "Мертвые души - это поэма Гоголя.",
        "Россия - страна с богатой литературой.",
        "Гоголь писал о русской душе.",
        "Помещики продавали души крестьян.",
        "Чичиков скупал мертвые души.",
        "Гоголь создал образы русских людей.",
        "Поэма осталась незавершенной.",
        "Гоголь умер в Москве.",
        "Творчество Гоголя изучают в школах."
    ]
    
    # Создаем датасет из 10 примеров для быстрого теста
    dataset = Dataset.from_dict({"text": texts})
    return dataset.train_test_split(test_size=0.3, seed=42)

def main():
    print("=== Тестовый пайплайн обучения на 3 эпохи ===")
    
    # Используем очень маленькую модель для теста
    MODEL_NAME = "distilgpt2"  # Всего 82M параметров
    OUTPUT_DIR = "./test_trained_model"
    
    # Создаем тестовый датасет
    print("Создание тестового датасета...")
    dataset = create_test_dataset()
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"Train examples: {len(train_dataset)}, Eval examples: {len(eval_dataset)}")
    
    # Загружаем модель и токенизатор
    print(f"Загрузка модели {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Токенизация датасета
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)
    
    print("Токенизация датасета...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
    
    # Добавляем labels для language modeling
    tokenized_train = tokenized_train.map(lambda x: {"labels": x["input_ids"]})
    tokenized_eval = tokenized_eval.map(lambda x: {"labels": x["input_ids"]})
    
    # Настраиваем LoRA (упрощенная версия)
    lora_config = LoraConfig(
        r=4,  # Очень маленький rank для теста
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Callback для мониторинга
    class EpochMonitor(TrainerCallback):
        def __init__(self):
            self.epoch_losses = []
            
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.epoch is not None:
                epoch_num = int(state.epoch)
                print(f"\n{'='*40}")
                print(f"Эпоха {epoch_num} завершена")
                
                # Получаем метрики из логов
                if hasattr(state, "log_history"):
                    logs = state.log_history
                    if logs:
                        # Ищем последнюю запись с eval_loss
                        for log in reversed(logs):
                            if "eval_loss" in log:
                                eval_loss = log["eval_loss"]
                                self.epoch_losses.append(eval_loss)
                                print(f"Validation loss: {eval_loss:.4f}")
                                
                                # Анализ улучшения
                                if len(self.epoch_losses) > 1:
                                    prev_loss = self.epoch_losses[-2]
                                    improvement = (prev_loss - eval_loss) / prev_loss * 100
                                    print(f"Улучшение по сравнению с предыдущей эпохой: {improvement:.2f}%")
                                    
                                    if improvement < 3:
                                        print("[WARNING] Незначительное улучшение (<3%)")
                                    elif improvement < 0:
                                        print("[ERROR] Переобучение! Loss увеличился")
                                    else:
                                        print("[SUCCESS] Хорошее улучшение, продолжаем обучение")
                                break
    
    # Аргументы обучения (3 эпохи)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # Ровно 3 эпохи как запрошено
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        eval_strategy="epoch",  # Оценка после каждой эпохи
        save_strategy="epoch",
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=False,
        max_grad_norm=0.3,
        warmup_steps=2,
        lr_scheduler_type="linear",
        report_to="none",
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Создаем trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Добавляем монитор
    monitor = EpochMonitor()
    trainer.add_callback(monitor)
    
    print(f"\n{'='*50}")
    print("НАЧАЛО ОБУЧЕНИЯ НА 3 ЭПОХИ")
    print(f"Модель: {MODEL_NAME}")
    print(f"Размер train датасета: {len(train_dataset)}")
    print(f"Размер validation датасета: {len(eval_dataset)}")
    print(f"Количество эпох: 3")
    print(f"{'='*50}\n")
    
    # Запускаем обучение
    try:
        trainer.train()
        
        print(f"\n{'='*50}")
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
        print(f"{'='*50}")
        
        # Сохраняем модель
        print("\nСохранение модели...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Анализ результатов
        print("\n=== АНАЛИЗ РЕЗУЛЬТАТОВ ===")
        if len(monitor.epoch_losses) >= 2:
            final_improvement = (monitor.epoch_losses[0] - monitor.epoch_losses[-1]) / monitor.epoch_losses[0] * 100
            print(f"Общее улучшение за 3 эпохи: {final_improvement:.2f}%")
            
            if final_improvement > 10:
                print("✅ Отличный результат! Модель значительно улучшилась.")
            elif final_improvement > 3:
                print("⚠️  Умеренное улучшение. Можно рассмотреть дополнительную эпоху.")
            else:
                print("🔴 Минимальное улучшение. Рекомендуется остановить обучение.")
        
        print(f"\nМодель сохранена в: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\nОшибка во время обучения: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()