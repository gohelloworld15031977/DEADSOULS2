#!/usr/bin/env python3
# Оптимизированное обучение для CPU с русскоязычной моделью
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
from datasets import load_from_disk
import json

class TrainingMonitor(TrainerCallback):
    """Мониторинг процесса обучения"""
    def __init__(self):
        self.epoch_losses = []
        
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch is not None:
            epoch_num = int(state.epoch)
            print(f"\n{'='*50}")
            print(f"Эпоха {epoch_num} завершена")
            
            if hasattr(state, "log_history"):
                logs = state.log_history
                if logs:
                    for log in reversed(logs):
                        if "eval_loss" in log:
                            eval_loss = log["eval_loss"]
                            self.epoch_losses.append(eval_loss)
                            print(f"Validation loss: {eval_loss:.4f}")
                            
                            if len(self.epoch_losses) > 1:
                                prev_loss = self.epoch_losses[-2]
                                improvement = (prev_loss - eval_loss) / prev_loss * 100
                                print(f"Улучшение: {improvement:.2f}%")
                                
                                if improvement < 3:
                                    print("[WARNING] Незначительное улучшение (<3%)")
                                elif improvement < 0:
                                    print("[ERROR] Переобучение! Loss увеличился")
                                else:
                                    print("[SUCCESS] Хорошее улучшение")
                            break

def main():
    print("=== Запуск обучения на CPU с оптимизированными параметрами ===")
    
    # Используем меньшую русскоязычную модель
    MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"  # 125M параметров
    OUTPUT_DIR = "./gogol_finetuned_cpu"
    DATA_PATH = "./data/tokenized_gpt2"  # Правильный датасет для GPT2
    
    # Проверка доступности данных
    if not os.path.exists(DATA_PATH):
        print(f"Ошибка: датасет не найден в {DATA_PATH}")
        print("Сначала запустите prepare_dataset.py")
        return
    
    # Загрузка датасета
    print("Загрузка датасета...")
    dataset = load_from_disk(DATA_PATH)
    
    from datasets import DatasetDict
    
    # Фильтрация примеров с pad токенами (токен 0)
    print("Фильтрация примеров с pad токенами...")
    def filter_pad_tokens(example):
        # Удаляем примеры где много pad токенов (более 50%)
        pad_count = example['input_ids'].count(0)
        return pad_count < len(example['input_ids']) * 0.5
    
    dataset = dataset.filter(filter_pad_tokens)
    print(f"После фильтрации: {len(dataset)} примеров")
    
    # Разделение на train/validation (90/10)
    if isinstance(dataset, DatasetDict):
        # Если DatasetDict, берём train и делим его
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
    
    print(f"Датасет загружен: {len(dataset)} примеров")
    print(f"Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")
    
    # Загрузка модели и токенизатора
    print(f"Загрузка модели {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Настройка токенизатора
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загружаем модель для CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Конфигурация LoRA для CPU
    lora_config = LoraConfig(  # type: ignore[call-arg]
        r=8,  # Меньший rank для CPU
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj", "c_fc"],
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
    
    # Монитор обучения
    monitor = TrainingMonitor()
    
    # Оптимизированные аргументы обучения для CPU
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,  # 3 эпохи как запрошено
        per_device_train_batch_size=1,  # Минимальный batch size для CPU
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Эффективный batch size = 8
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=1e-4,  # Меньший learning rate для стабильности
        weight_decay=0.01,
        fp16=False,  # Не использовать mixed precision на CPU
        bf16=False,
        max_grad_norm=0.5,
        warmup_steps=50,
        lr_scheduler_type="linear",
        report_to="none",
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_pin_memory=False,  # Важно для CPU
        remove_unused_columns=True,
    )
    
    # Создаем trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[monitor],
    )
    
    print(f"\n{'='*60}")
    print("ПАРАМЕТРЫ ОБУЧЕНИЯ:")
    print(f"Модель: {MODEL_NAME}")
    print(f"Размер модели: {model.num_parameters():,} параметров")  # type: ignore[operator]
    print(f"Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Датасет: {len(train_dataset)} train, {len(eval_dataset)} validation")
    print(f"Эпохи: 3")
    print(f"Batch size: 1 (effective: 8 с gradient accumulation)")
    print(f"Learning rate: 1e-4")
    print(f"LoRA rank: 8")
    print(f"{'='*60}\n")
    
    # Запуск обучения
    print("Начало обучения...")
    try:
        trainer.train()
        
        print(f"\n{'='*60}")
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
        print(f"{'='*60}")
        
        # Сохранение модели
        print("\nСохранение модели...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Сохранение метрик
        metrics = {
            "model": MODEL_NAME,
            "epochs_trained": 3,
            "final_train_loss": trainer.state.log_history[-1]["loss"] if trainer.state.log_history else None,
            "final_eval_loss": trainer.state.log_history[-1]["eval_loss"] if trainer.state.log_history else None,
            "dataset_size": len(dataset),
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "improvement_history": monitor.epoch_losses
        }
        
        with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\nМодель сохранена в: {OUTPUT_DIR}")
        print(f"Метрики сохранены в: {OUTPUT_DIR}/training_results.json")
        
        # Анализ результатов
        if len(monitor.epoch_losses) >= 2:
            final_improvement = (monitor.epoch_losses[0] - monitor.epoch_losses[-1]) / monitor.epoch_losses[0] * 100
            print(f"\nОбщее улучшение за 3 эпохи: {final_improvement:.2f}%")
            
            if final_improvement > 10:
                print("Отличный результат! Модель значительно улучшилась.")
            elif final_improvement > 3:
                print("Хороший результат. Модель адаптировалась к стилю Гоголя.")
            else:
                print("Минимальное улучшение. Рекомендуется увеличить количество эпох или изменить параметры.")
        
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем")
    except Exception as e:
        print(f"\nОшибка во время обучения: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()