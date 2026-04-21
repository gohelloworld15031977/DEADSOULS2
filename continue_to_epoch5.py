#!/usr/bin/env python3
"""
Продолжение обучения с чекпоинта эпохи 4 до 5 эпох
"""

import torch
import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import PeftModel
from datasets import load_from_disk, DatasetDict
from datetime import datetime

def main():
    print("=== Продолжение обучения до 5 эпох (с чекпоинта 4) ===")
    
    # Настройки
    MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
    DATA_PATH = "data/tokenized_gpt2"
    CHECKPOINT_PATH = "./gogol_finetuned_final/checkpoint-1500"  # Эпоха 4.0
    OUTPUT_DIR = "./gogol_finetuned_final"
    TARGET_EPOCHS = 5
    
    # Проверка данных
    if not os.path.exists(DATA_PATH):
        print(f"Ошибка: датасет не найден в {DATA_PATH}")
        return
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Ошибка: чекпоинт не найден в {CHECKPOINT_PATH}")
        return
    
    # Загрузка датасета
    print("Загрузка датасета...")
    dataset = load_from_disk(DATA_PATH)
    
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
    
    print(f"Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
    # Загрузка токенизатора
    print(f"Загрузка токенизатора из {CHECKPOINT_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загрузка модели с LoRA
    print("Загрузка модели с LoRA адаптером (эпоха 4)...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.print_trainable_parameters()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TARGET_EPOCHS,  # Всего 5 эпох
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=False,
        max_grad_norm=0.5,
        warmup_steps=50,
        lr_scheduler_type="linear",
        report_to="none",
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)],
    )
    
    # Вывод параметров
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ ОБУЧЕНИЯ:")
    print(f"Модель: {MODEL_NAME}")
    print(f"Чекпоинт для продолжения: {CHECKPOINT_PATH}")
    print(f"Текущая эпоха: 4.0")
    print(f"Целевая эпоха: {TARGET_EPOCHS}")
    print(f"Дополнительные эпохи: {TARGET_EPOCHS - 4}")
    print(f"Learning rate: 5e-5")
    print(f"{'='*60}\n")
    
    # Запуск обучения
    print("Начало продолжения обучения...")
    start_time = datetime.now()
    print(f"Время начала: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print(f"{'='*60}")
        print(f"Время обучения: {duration}")
        
        # Сохранение модели
        print("\nСохранение модели...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Сохранение метрик
        metrics = {
            "model_name": MODEL_NAME,
            "checkpoint_resumed": CHECKPOINT_PATH,
            "epochs_trained": TARGET_EPOCHS,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "dataset_size": len(dataset),
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "training_args": {
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation": training_args.gradient_accumulation_steps,
            }
        }
        
        # Добавляем историю обучения
        if trainer.state.log_history:
            metrics["log_history"] = trainer.state.log_history
            final_log = trainer.state.log_history[-1]
            if "eval_loss" in final_log:
                metrics["final_eval_loss"] = final_log["eval_loss"]
            if "loss" in final_log:
                metrics["final_train_loss"] = final_log["loss"]
        
        metrics_path = os.path.join(OUTPUT_DIR, "training_results.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\nМодель сохранена в: {OUTPUT_DIR}")
        print(f"Метрики сохранены в: {metrics_path}")
        
        # Вывод финальных метрик
        if trainer.state.log_history:
            print("\n=== ФИНАЛЬНЫЕ МЕТРИКИ ===")
            for log in trainer.state.log_history:
                if "eval_loss" in log:
                    print(f"Эпоха {log.get('epoch', 'N/A')}: eval_loss = {log['eval_loss']:.4f}")
        
        # Тест генерации
        print("\n" + "="*60)
        print("ТЕСТ ГЕНЕРАЦИИ")
        print("="*60)
        
        model.eval()
        prompts = [
            "Чичиков приехал в город",
            "В губернском городе N",
            "Мёртвые души - это"
        ]
        
        generation_results = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nПромпт: {prompt}")
            print(f"Генерация: {generated}")
            
            generation_results.append({"prompt": prompt, "generated": generated})
        
        # Сохранение примеров генерации
        gen_path = os.path.join(OUTPUT_DIR, "generation_samples_epoch5.json")
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(generation_results, f, indent=2, ensure_ascii=False)
        print(f"\nПримеры генерации сохранены в: {gen_path}")
        
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ ДО 5 ЭПОХ ЗАВЕРШЕНО УСПЕШНО!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nОбучение прервано пользователем")
        print("Сохраняем текущее состояние...")
        model.save_pretrained(OUTPUT_DIR + "_interrupted")
        tokenizer.save_pretrained(OUTPUT_DIR + "_interrupted")
    except Exception as e:
        print(f"\nОшибка во время обучения: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
