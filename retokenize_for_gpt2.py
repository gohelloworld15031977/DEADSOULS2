#!/usr/bin/env python3
# Перетокенизация датасета для модели GPT2
from transformers import AutoTokenizer
from datasets import Dataset
import os

def main():
    print("Перетокенизация датасета для модели GPT2...")
    
    # Модель для токенизации
    MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
    
    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загружаем исходные тексты
    with open("data/gogol_processed.txt", "r", encoding="utf-8") as f:
        paragraphs = [line.strip() for line in f if line.strip()]
    
    print(f"Загружено {len(paragraphs)} абзацев")
    
    # Создаем датасет
    dataset = Dataset.from_list([{"text": p} for p in paragraphs])
    
    # Функция токенизации
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,  # Уменьшенная длина для CPU
            return_tensors=None
        )
    
    # Токенизируем
    print("Токенизация...")
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Добавляем labels
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]})
    
    # Сохраняем
    output_dir = "data/tokenized_gpt2"
    tokenized.save_to_disk(output_dir)
    
    print(f"Датасет сохранен в {output_dir}")
    print(f"Количество примеров: {len(tokenized)}")
    
    # Проверяем размер словаря
    print(f"\nИнформация о токенизаторе:")
    print(f"Размер словаря: {tokenizer.vocab_size}")
    print(f"Pad token ID: {tokenizer.pad_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    
    # Проверяем диапазон токенов в датасете
    all_token_ids = []
    for example in tokenized:
        all_token_ids.extend(example["input_ids"])  # type: ignore[index]
    
    max_token = max(all_token_ids)
    min_token = min(all_token_ids)
    
    print(f"\nПроверка токенов в датасете:")
    print(f"Максимальный токен ID: {max_token}")
    print(f"Минимальный токен ID: {min_token}")
    print(f"Размер словаря модели: {tokenizer.vocab_size}")
    
    if max_token >= tokenizer.vocab_size:
        print(f"ОШИБКА: Максимальный токен {max_token} >= размера словаря {tokenizer.vocab_size}")
        # Фильтруем проблемные примеры
        print("Фильтрация проблемных примеров...")
        valid_examples = []
        for example in tokenized:
            if max(example["input_ids"]) < tokenizer.vocab_size:  # type: ignore[index]
                valid_examples.append(example)
        
        print(f"Оставлено {len(valid_examples)} валидных примеров из {len(tokenized)}")
        
        # Создаем новый датасет
        from datasets import Dataset as HFDataset
        valid_dataset = HFDataset.from_list(valid_examples)
        valid_dataset.save_to_disk(output_dir + "_filtered")
        print(f"Отфильтрованный датасет сохранен в {output_dir}_filtered")
    else:
        print("Все токены в пределах словаря ✓")

if __name__ == "__main__":
    main()