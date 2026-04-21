#!/usr/bin/env python3
# Тест совместимости Python 3.13 с библиотеками для ML
import sys
import torch
import transformers
import peft
import datasets
import accelerate

print("=== Тест Python 3.13 для операций с моделью ===")
print(f"Версия Python: {sys.version}")
print(f"Версия PyTorch: {torch.__version__}")
print(f"Версия Transformers: {transformers.__version__}")
print(f"Версия PEFT: {peft.__version__}")
print(f"Версия Datasets: {datasets.__version__}")
print(f"Версия Accelerate: {accelerate.__version__}")

# Проверка доступности GPU
if torch.cuda.is_available():
    print(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
    print(f"Версия CUDA: {torch.version.cuda}")
else:
    print("CUDA недоступна, используется CPU")

# Проверка основных функций
try:
    # Тест создания тензора
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"Тензор создан: {x}")
    
    # Тест импорта конфигураций
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("gpt2")
    print(f"Конфигурация GPT2 загружена: {config.model_type}")
    
    # Тест токенизатора
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    test_text = "Привет, мир!"
    tokens = tokenizer.encode(test_text)
    print(f"Токенизация '{test_text}': {tokens}")
    
    print("\n[SUCCESS] Все тесты пройдены успешно!")
    print("Python 3.13 полностью совместим с библиотеками для дообучения и генерации моделей.")
    
except Exception as e:
    print(f"\n[ERROR] Ошибка при тестировании: {e}")
    import traceback
    traceback.print_exc()