"""
Продолжение обучения с чекпоинта checkpoint-750 (эпоха 2.0)
Цель: обучить до 5 эпох всего
"""

import json
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from datetime import datetime

def main():
    # === КОНФИГУРАЦИЯ ===
    MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
    DATA_PATH = "data/tokenized_gpt2"
    CHECKPOINT_PATH = "./gogol_finetuned_final/checkpoint-750"  # Эпоха 2.0
    OUTPUT_DIR = "./gogol_finetuned_from_epoch2"
    TARGET_EPOCHS = 5  # Всего 5 эпох
    
    # Гиперпараметры
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LEARNING_RATE = 5e-5  # Уменьшенный LR для продолжения
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 4
    
    print(f"{'='*60}")
    print(f"Продолжение обучения с чекпоинта эпохи 2")
    print(f"{'='*60}")
    print(f"Модель: {MODEL_NAME}")
    print(f"Чекпоинт для продолжения: {CHECKPOINT_PATH}")
    print(f"Текущая эпоха: 2.0")
    print(f"Целевая эпоха: {TARGET_EPOCHS}")
    print(f"Дополнительные эпохи: {TARGET_EPOCHS - 2}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"{'='*60}\n")
    
    # Проверка чекпоинта
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Ошибка: чекпоинт не найден: {CHECKPOINT_PATH}")
        return
    
    # Загрузка токенизатора
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
    
    # Загрузка датасета
    print("Загрузка датасета...")
    full_dataset = load_from_disk(DATA_PATH)
    print(f"Тип датасета: {type(full_dataset)}")
    
    # Разделение на train/eval (90/10)
    from datasets import Dataset, DatasetDict
    
    if isinstance(full_dataset, DatasetDict):
        # Если это DatasetDict (уже есть split'ы)
        if 'train' in full_dataset:
            train_dataset = full_dataset['train']
            eval_dataset = full_dataset.get('test', full_dataset.get('eval'))
            print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 'N/A'}")
        else:
            # Если нет train, берём первый ключ и делим
            first_key = list(full_dataset.keys())[0]
            dataset = full_dataset[first_key].train_test_split(test_size=0.1, seed=42)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]
    elif isinstance(full_dataset, Dataset):
        # Если это Dataset - делим напрямую
        dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        print(f"Ошибка: неизвестный тип датасета: {type(full_dataset)}")
        return
    
    # Загрузка базовой модели
    print("Загрузка базовой модели...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="cpu"  # Trainer сам переместит модель на GPU
    )
    
    # Загрузка чекпоинта с LoRA адаптером
    print(f"Загрузка LoRA адаптера из {CHECKPOINT_PATH}...")
    
    # Загружаем config адаптера
    from peft import PeftConfig
    peft_config = PeftConfig.from_pretrained(CHECKPOINT_PATH)
    print(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, target_modules={peft_config.target_modules}")
    
    # Загружаем модель с адаптером в режиме обучения
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH, is_trainable=True)
    
    # Активируем адаптер
    model.set_adapter(model.active_adapters[0] if hasattr(model, 'active_adapters') and model.active_adapters else 'default')
    
    # Настраиваем градиенты для LoRA параметров
    trainable_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += param.numel()
        if 'lora' in name.lower():
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
    
    model.train()
    
    print(f"Модель загружена с весами из чекпоинта эпохи 2.0")
    print(f"Trainable params: {trainable_count}, Total params: {total_count}")
    
    # Проверка, что есть градиенты
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params}, Total params: {total_params}")
    
    # Аргументы обучения
    print("Настройка аргументов обучения...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=TARGET_EPOCHS,  # Всего 5 эпох
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=f"{OUTPUT_DIR}/runs"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Инициализация Trainer
    print("\nИнициализация Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
    
    # Запуск обучения
    print("\n" + "="*60)
    print("НАЧАЛО ПРОДОЛЖЕНИЯ ОБУЧЕНИЯ")
    print("="*60 + "\n")
    
    start_time = datetime.now()
    print(f"Время начала: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    trainer.train()
    
    end_time = datetime.now()
    print(f"\nВремя окончания: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Общее время: {end_time - start_time}")
    
    # Сохранение финальной модели
    print("\nСохранение финальной модели...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Сохранение результатов
    results = {
        "model_name": MODEL_NAME,
        "checkpoint_resumed": CHECKPOINT_PATH,
        "epochs_trained": TARGET_EPOCHS,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "final_eval_loss": trainer.state.log_history[-1].get("eval_loss") if trainer.state.log_history else None,
        "final_train_loss": trainer.state.log_history[-1].get("loss") if trainer.state.log_history else None
    }
    
    results_path = os.path.join(OUTPUT_DIR, "training_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nРезультаты сохранены в: {results_path}")
    
    # Вывод истории обучения
    print("\n" + "="*60)
    print("ИСТОРИЯ ОБУЧЕНИЯ")
    print("="*60)
    for log in trainer.state.log_history:
        if "eval_loss" in log:
            print(f"Эпоха {log.get('epoch')}: eval_loss = {log['eval_loss']:.4f}")
    
    # Тест генерации
    print("\n" + "="*60)
    print("ТЕСТ ГЕНЕРАЦИИ")
    print("="*60)
    
    test_prompts = [
        "В один из самых теплых дней",
        "На дворе стояла",
        "Николай Гоголь написал"
    ]
    
    generation_results = []
    model.eval()
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_results.append({
            "prompt": prompt,
            "generated": generated
        })
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated[:100]}...")
    
    # Сохранение примеров генерации
    gen_path = os.path.join(OUTPUT_DIR, "generation_samples.json")
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(generation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nПримеры генерации сохранены в: {gen_path}")
    print(f"\nОбучение завершено! Модель сохранена в: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
