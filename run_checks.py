#!/usr/bin/env python3
"""
Тестовый скрипт для проверки всех улучшений.
"""

import os
import sys
import io

# Установка кодировки для Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_dependencies():
    """Проверка установленных зависимостей"""
    print("=== Проверка зависимостей ===\n")
    
    dependencies = {
        "torch": "PyTorch",
        "transformers": "HuggingFace Transformers",
        "datasets": "HuggingFace Datasets",
        "peft": "Parameter-Efficient Fine-Tuning",
        "accelerate": "HuggingFace Accelerate",
        "matplotlib": "Визуализация",
    }
    
    optional = {
        "nltk": "BLEU метрика",
        "rouge_score": "ROUGE метрика",
        "mlflow": "Трекинг экспериментов",
        "gitpython": "Git информация",
    }
    
    installed = []
    missing = []
    
    for package, name in dependencies.items():
        try:
            __import__(package)
            print(f"[OK] {name} ({package})")
            installed.append(package)
        except ImportError:
            print(f"[MISSING] {name} ({package}) - НЕ УСТАНОВЛЕН")
            missing.append(package)
    
    print("\nОпциональные:")
    for package, name in optional.items():
        try:
            __import__(package)
            print(f"[OK] {name} ({package})")
            installed.append(package)
        except ImportError:
            print(f"[OPTIONAL] {name} ({package}) - НЕ УСТАНОВЛЕН (опционально)")
    
    print(f"\nУстановлено: {len(installed)}/{len(dependencies) + len(optional)}")
    
    return len(missing) == 0

def check_project_structure():
    """Проверка структуры проекта"""
    print("\n=== Проверка структуры проекта ===\n")
    
    required_files = [
        "config_unified.py",
        "finetune.py",
        "train_with_monitoring.py",
        "validate_dataset.py",
        "test_quality_ci.py",
        "experiment_tracker.py",
        "visualize_results.py",
        "improve_data_quality.py",
        "create_extended_dataset.py",
        "test_generation_quality.py",
        "debug_repetitions.py",
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"[OK] {file}")
        else:
            print(f"[MISSING] {file} - ОТСУТСТВУЕТ")
    
    required_dirs = ["data/", "logs/", "visualization/"]
    for dir in required_dirs:
        if os.path.exists(dir):
            print(f"[OK] {dir}")
        else:
            print(f"[NOTE] {dir} - НЕ СУЩЕСТВУЕТ (будет создан)")

def check_gpu():
    """Проверка доступности GPU"""
    print("\n=== Проверка GPU ===\n")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] GPU доступен: {torch.cuda.get_device_name(0)}")
            print(f"  Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("[NOTE] GPU не обнаружен. Обучение будет на CPU.")
    except ImportError:
        print("[MISSING] PyTorch не установлен")

def check_dataset():
    """Проверка датасета"""
    print("\n=== Проверка датасета ===\n")
    
    dataset_paths = [
        "data/gogol_processed.txt",
        "data/tokenized_gpt2",
        "data/combined_dataset.txt"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(path)
                    for filename in filenames
                )
                print(f"[OK] {path} ({size / 1e6:.2f} MB)")
            else:
                size = os.path.getsize(path)
                print(f"[OK] {path} ({size / 1e3:.2f} KB)")
        else:
            print(f"[NOTE] {path} - НЕ НАЙДЕН")

def main():
    print("="*60)
    print("ПРОВЕРКА ПРОЕКТА DEADSOULS")
    print("="*60)
    
    # Проверки
    deps_ok = check_dependencies()
    check_project_structure()
    check_gpu()
    check_dataset()
    
    print("\n" + "="*60)
    print("ИТОГОВАЯ ПРОВЕРКА ЗАВЕРШЕНА")
    print("="*60)
    
    if not deps_ok:
        print("\n[WARNING] Некоторые зависимости отсутствуют.")
        print("Установите их командой:")
        print("  pip install torch transformers datasets peft accelerate matplotlib")
        print("  pip install nltk rouge-score mlflow gitpython")

if __name__ == "__main__":
    main()
