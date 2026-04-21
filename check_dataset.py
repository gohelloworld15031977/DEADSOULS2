#!/usr/bin/env python3
"""Анализ датасета Гоголя"""

from datasets import load_from_disk
import os

print("=== Анализ датасета Гоголя ===\n")

# Проверка токенизированного датасета
print("1. Токенизированный датасет (data/tokenized_gpt2):")
dataset = load_from_disk('data/tokenized_gpt2')
print(f"   Тип: {type(dataset).__name__}")
print(f"   Количество примеров: {len(dataset)}")
if len(dataset) > 0:
    sample = dataset[0]  # type: ignore[index]
    print(f"   Пример структуры: {list(sample.keys())}")  # type: ignore[union-attr]
    print(f"   Длина первого примера (токенов): {len(sample.get('input_ids', []))}")  # type: ignore[union-attr]

# Проверка текстовых файлов
print("\n2. Текстовые файлы:")
files = [
    'data/gogol_books.txt',
    'data/gogol_processed.txt',
    'data/gogol_complete.txt',
    'data/combined_dataset.txt'
]

for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f)
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
            chars = len(content)
            lines = len(content.split('\n'))
        print(f"   {f}:")
        print(f"      - Размер: {size:,} байт")
        print(f"      - Символов: {chars:,}")
        print(f"      - Строк: {lines:,}")
    else:
        print(f"   {f}: НЕ НАЙДЕН")

# Проверка дополнительных текстов
print("\n3. Дополнительные тексты (data/gogol_additional/):")
additional_dir = 'data/gogol_additional'
if os.path.exists(additional_dir):
    for filename in os.listdir(additional_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(additional_dir, filename)
            size = os.path.getsize(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"   {filename}: {size:,} байт, {len(content):,} символов")
else:
    print("   Папка не найдена")

# Итого
print("\n4. Итого:")
print(f"   Датасет для обучения: {len(dataset)} примеров")
print(f"   Модель: ai-forever/rugpt3small_based_on_gpt2")
print(f"   Текущая эпоха обучения: ~3.15 из 5")
