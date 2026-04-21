#!/usr/bin/env python3
# Скрипт для продолжения обучения с checkpoint (исправленная версия)
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_from_disk, DatasetDict

def main():
    print("=== Продолжение обучения модели на текстах Гоголя ===")
    
    # Модель и данные
    MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
    DATA_PATH = "data/tokenized_gpt2"
    CHECKPOINT_PATH = "./gogol_finetuned_final/checkpoint-375"
    OUTPUT_DIR = "./gogol_finetuned_resumed"
    
    # Проверка данных
    if not os.path.exists(DATA_PATH):
        print(f"Ошибка: датасет не найден в {DATA_PATH}")
        print("Сначала запустите retokenize_for_gpt2.py")
        return
    
    # Проверка чекпоинта
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Ошибка: чекпоинт не найден в {CHECKPOINT_PATH}")
        print("Проверьте путь к чекпоинту")
        return
    
    # Загрузка датасета
    print("Загрузка датасета...")
    dataset = load_from_disk(DATA_PATH)
    
    # Разделение
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
    
    print(f"Датасет: {len(dataset)} примеров")
    print(f"Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
    # Загрузка токенизатора из чекпоинта
    print(f"Загрузка токенизатора из чекпоинта {CHECKPOINT_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загрузка базовой модели
    print("Загрузка базовой модели...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    
    # Загрузка LoRA адаптера из чекпоинта
    print("Загрузка LoRA адаптера из чекпоинта...")
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
    model.print_trainable_parameters()
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Аргументы обучения с продолжением
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,                    # дополнительные 5 эпох
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        learning_rate=5e-5,                    # уменьшенный learning rate для продолжения
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
    
    # Trainer с early stopping callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.01)],
    )
    
    print(f"\nПараметры обучения:")
    print(f"- Модель: {MODEL_NAME}")
    print(f"- Чекпоинт: {CHECKPOINT_PATH}")
    print(f"- Эпохи: 5 (дополнительно)")
    print(f"- Batch size: 1 (effective: 4)")
    print(f"- Learning rate: 5e-5")
    print(f"- Данные: {len(train_dataset)} train, {len(eval_dataset)} validation")
    print(f"- Выходная директория: {OUTPUT_DIR}")
    print(f"- Early stopping: включен (patience=2, threshold=0.01)")
    print("\nПродолжение обучения...")
    
    # Запуск обучения с продолжением
    try:
        trainer.train(resume_from_checkpoint=CHECKPOINT_PATH)
        
        print("\nОбучение завершено успешно!")
        
        # Сохранение
        print("Сохранение модели...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Анализ результатов
        if trainer.state.log_history:
            final_log = trainer.state.log_history[-1]
            if "eval_loss" in final_log:
                print(f"\nФинальные метрики:")
                print(f"- Validation loss: {final_log['eval_loss']:.4f}")
                if "loss" in final_log:
                    print(f"- Train loss: {final_log['loss']:.4f}")
        
        print(f"\nМодель сохранена в: {OUTPUT_DIR}")
        
        # Тест генерации
        print("\nТест генерации текста...")
        model.eval()
        prompts = [
            "Чичиков приехал в город",
            "В губернском городе N",
            "Однажды вечером сидел я"
        ]
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(  # type: ignore[union-attr]
                    inputs.input_ids,
                    max_length=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nПромпт: {prompt}")
            print(f"Сгенерировано: {generated}")
            
    except Exception as e:
        print(f"\nОшибка во время обучения: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()