#!/usr/bin/env python3
"""
Валидация датасета перед обучением.
Проверяет:
- Валидность токенов (в пределах словаря)
- Распределение длин последовательностей
- Отсутствие проблемных примеров
- Баланс train/test split
"""

import os
import sys
import json
import io

# Установка кодировки для Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer
from datasets import load_from_disk

def validate_tokenizer(model_name, dataset_path):
    """Проверяет совместимость токенизатора и датасета"""
    print(f"=== Валидация токенизатора: {model_name} ===")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить токенизатор {model_name}: {e}")
        return None
    
    print(f"Размер словаря: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    
    return tokenizer

def validate_dataset(dataset_path, tokenizer, max_allowed_token=None):
    """Проверяет датасет на проблемы"""
    print(f"\n=== Валидация датасета: {dataset_path} ===")
    
    if not os.path.exists(dataset_path):
        print(f"ОШИБКА: Датасет не найден: {dataset_path}")
        return None
    
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить датасет: {e}")
        return None
    
    print(f"Тип датасета: {type(dataset)}")
    
    # Если DatasetDict, берем train
    if hasattr(dataset, 'keys'):
        print(f"Сplits: {list(dataset.keys())}")
        if 'train' in dataset:
            dataset = dataset['train']
            print("Используется split 'train'")
    
    print(f"Количество примеров: {len(dataset)}")
    print(f"Фичи: {dataset.features}")
    
    # Анализ токенов
    all_lengths = []
    problematic_examples = []
    max_token_id = 0
    min_token_id = float('inf')
    
    print("\nАнализ примеров...")
    for i, example in enumerate(dataset):
        if i % 1000 == 0 and i > 0:
            print(f"  Обработано {i}/{len(dataset)}")
        
        input_ids = example.get('input_ids', [])
        if not input_ids:
            problematic_examples.append((i, "Пустые input_ids"))
            continue
        
        all_lengths.append(len(input_ids))
        max_token_id = max(max_token_id, max(input_ids))
        min_token_id = min(min_token_id, min(input_ids))
        
        # Проверка на токены вне словаря
        if max_allowed_token and max(input_ids) >= max_allowed_token:
            problematic_examples.append((i, f"Токен {max(input_ids)} >= vocab_size {max_allowed_token}"))
        
        # Проверка на слишком длинные/короткие примеры
        if len(input_ids) > 1024:
            problematic_examples.append((i, f"Слишком длинный: {len(input_ids)} токенов"))
        elif len(input_ids) < 10:
            problematic_examples.append((i, f"Слишком короткий: {len(input_ids)} токенов"))
    
    # Вывод статистики
    print(f"\n=== Статистика длин последовательностей ===")
    print(f"Минимальная длина: {min(all_lengths)}")
    print(f"Максимальная длина: {max(all_lengths)}")
    print(f"Средняя длина: {np.mean(all_lengths):.1f}")
    print(f"Медианная длина: {np.median(all_lengths):.1f}")
    print(f"Стандартное отклонение: {np.std(all_lengths):.1f}")
    
    print(f"\n=== Диапазон токенов ===")
    print(f"Минимальный токен ID: {min_token_id}")
    print(f"Максимальный токен ID: {max_token_id}")
    
    if max_allowed_token:
        print(f"Максимальный допустимый токен: {max_allowed_token}")
        if max_token_id >= max_allowed_token:
            print(f"[WARNING] Некоторые токены超出 словаря!")
    
    print(f"\n=== Проблемные примеры ===")
    if problematic_examples:
        print(f"Найдено {len(problematic_examples)} проблемных примеров:")
        for idx, reason in problematic_examples[:10]:
            print(f"  Пример {idx}: {reason}")
        if len(problematic_examples) > 10:
            print(f"  ... и еще {len(problematic_examples) - 10}")
    else:
        print("Проблемных примеров не найдено [OK]")
    
    return {
        "dataset": dataset,
        "total_examples": len(dataset),
        "problematic_examples": problematic_examples,
        "lengths": all_lengths,
        "max_token_id": max_token_id,
        "min_token_id": min_token_id
    }

def plot_length_distribution(lengths, output_path="data/length_distribution.png"):
    """Строит гистограмму распределения длин"""
    print(f"\nПостроение гистограммы распределения длин...")
    
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Длина последовательности (токены)')
    plt.ylabel('Количество примеров')
    plt.title('Распределение длин последовательностей в датасете')
    plt.axvline(x=np.median(lengths), color='r', linestyle='--', label=f'Медиана: {np.median(lengths):.0f}')
    plt.axvline(x=np.mean(lengths), color='g', linestyle='--', label=f'Среднее: {np.mean(lengths):.0f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Гистограмма сохранена в {output_path}")

def validate_train_test_split(dataset_path):
    """Проверяет баланс train/test split"""
    print(f"\n=== Проверка train/test split ===")
    
    if not os.path.exists(dataset_path):
        return
    
    try:
        dataset = load_from_disk(dataset_path)
        
        if hasattr(dataset, 'keys'):
            splits = list(dataset.keys())
            print(f"Наличие splits: {splits}")
            
            if 'train' in dataset and 'test' in dataset:
                train_size = len(dataset['train'])
                test_size = len(dataset['test'])
                total = train_size + test_size
                
                print(f"Train: {train_size} ({train_size/total*100:.1f}%)")
                print(f"Test: {test_size} ({test_size/total*100:.1f}%)")
                
                if test_size < 50:
                    print(f"[WARNING] Валидационный набор слишком мал (<50 примеров)")
                if test_size / train_size < 0.05:
                    print(f"[WARNING] Валидационный набор < 5% от train")
            else:
                print("Сplits не найдены. Требуется ручное разделение.")
        else:
            print(f"Датасет без splits, количество примеров: {len(dataset)}")
    except Exception as e:
        print(f"Ошибка проверки splits: {e}")

def generate_report(validation_results, output_path="data/validation_report.json"):
    """Генерирует JSON отчет о валидации"""
    report = {
        "status": "passed" if not validation_results.get("problematic_examples") else "warnings",
        "total_examples": validation_results.get("total_examples"),
        "problematic_count": len(validation_results.get("problematic_examples", [])),
        "length_stats": {
            "min": min(validation_results.get("lengths", [0])),
            "max": max(validation_results.get("lengths", [0])),
            "mean": float(np.mean(validation_results.get("lengths", [0]))),
            "median": float(np.median(validation_results.get("lengths", [0])))
        },
        "token_range": {
            "min": validation_results.get("min_token_id"),
            "max": validation_results.get("max_token_id")
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\nОтчет валидации сохранен в {output_path}")
    return report

def main():
    print("=== Валидация датасета для обучения ===\n")
    
    # Конфигурация
    MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
    DATASET_PATH = "data/tokenized_gpt2"
    
    # Валидация токенизатора
    tokenizer = validate_tokenizer(MODEL_NAME, DATASET_PATH)
    if not tokenizer:
        return
    
    # Валидация датасета
    results = validate_dataset(DATASET_PATH, tokenizer, max_allowed_token=tokenizer.vocab_size)
    
    if not results:
        print("\nВалидация завершена с ошибками!")
        return
    
    # Построение гистограммы
    if results["lengths"]:
        plot_length_distribution(results["lengths"])
    
    # Проверка splits
    validate_train_test_split(DATASET_PATH)
    
    # Генерация отчета
    report = generate_report(results)
    
    # Итоговый статус
    print("\n" + "="*50)
    if report["status"] == "passed":
        print("[OK] Валидация пройдена успешно!")
    else:
        print(f"[WARNING] Валидация завершена с предупреждениями ({report['problematic_count']} проблемных примеров)")
    print("="*50)

if __name__ == "__main__":
    main()
