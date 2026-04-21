#!/usr/bin/env python3
# Упрощенное обучение с перетокенизированным датасетом и early stopping
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
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

def main():
    print("=== Упрощенное обучение модели на текстах Гоголя ===")
    
    # Модель и данные
    MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
    DATA_PATH = "data/tokenized_gpt2"
    OUTPUT_DIR = "./gogol_finetuned_final"
    
    # Проверка данных
    if not os.path.exists(DATA_PATH):
        print(f"Ошибка: датасет не найден в {DATA_PATH}")
        print("Сначала запустите retokenize_for_gpt2.py")
        return
    
    # Загрузка датасета
    print("Загрузка датасета...")
    dataset = load_from_disk(DATA_PATH)
    
    # Разделение
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Датасет: {len(dataset)} примеров")
    print(f"Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
    # Загрузка модели
    print(f"Загрузка модели {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )
    
    # LoRA конфигурация
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
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
    
    # Аргументы обучения с early stopping
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=7,                    # увеличено до 7 эпох
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        learning_rate=1e-4,
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
        save_total_limit=2,
        early_stopping_patience=2,             # остановка после 2 эпох без улучшений
        early_stopping_threshold=0.01,         # минимальное улучшение
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
    print(f"- Эпохи: 7 (с early stopping patience=2)")
    print(f"- Batch size: 1 (effective: 4)")
    print(f"- Learning rate: 1e-4")
    print(f"- Данные: {len(train_dataset)} train, {len(eval_dataset)} validation")
    print(f"- Выходная директория: {OUTPUT_DIR}")
    print(f"- Early stopping: включен (threshold=0.01)")
    print("\nНачало обучения...")
    
    # Запуск обучения
    try:
        trainer.train()
        
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
        prompt = "Чичиков приехал в город"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Промпт: {prompt}")
        print(f"Сгенерировано: {generated}")
        
    except Exception as e:
        print(f"\nОшибка во время обучения: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()