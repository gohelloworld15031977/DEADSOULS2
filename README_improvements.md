# DeadSouls: Генерация текста в стиле Гоголя

## Описание проекта

Проект fine-tuning языковых моделей для генерации текста в стиле Николая Гоголя. Использует методы QLoRA и LoRA для эффективного обучения больших моделей.

## 🚀 Внесенные улучшения

### 1. Расширенный датасет
- ✅ **Файл**: `create_extended_dataset.py`
- **Что делает**: Объединяет тексты Гоголя с другими авторами XIX века
- **Запуск**: `python create_extended_dataset.py`

### 2. Валидация датасета
- ✅ **Файл**: `validate_dataset.py`
- **Что делает**: Проверяет токены, распределение длин, проблемные примеры
- **Запуск**: `python validate_dataset.py`

### 3. Унифицированная конфигурация
- ✅ **Файл**: `config_unified.py`
- **Что делает**: Единая конфигурация для всех скриптов обучения
- **Преимущества**: Устранение дублирования, легкое управление параметрами

### 4. Мониторинг обучения
- ✅ **Файл**: `train_with_monitoring.py`
- **Что делает**: Обучение с TensorBoard + логирование в файл
- **Запуск**: `python train_with_monitoring.py`
- **Просмотр логов**: `tensorboard --logdir logs/tensorboard`

### 5. Тестирование качества генерации
- ✅ **Файл**: `test_generation_quality.py`
- **Что делает**: BLEU, ROUGE, Perplexity метрики
- **Запуск**: `python test_generation_quality.py`

### 6. Отладка повторений
- ✅ **Файл**: `debug_repetitions.py`
- **Что делает**: Тестирует параметры для минимизации повторений
- **Запуск**: `python debug_repetitions.py`

### 7. Трекинг экспериментов
- ✅ **Файл**: `experiment_tracker.py`
- **Что делает**: Версионирование экспериментов с MLflow
- **Использование**:
```python
from experiment_tracker import log_training_experiment
log_training_experiment(model_name, dataset_size, params, metrics, model_path)
```

### 8. Автоматические тесты CI/CD
- ✅ **Файл**: `test_quality_ci.py`
- **Что делает**: Юнит-тесты качества генерации
- **Запуск**: `python test_quality_ci.py`

### 9. Визуализация результатов
- ✅ **Файл**: `visualize_results.py`
- **Что делает**: Графики обучения, сравнение экспериментов
- **Запуск**: `python visualize_results.py`

### 10. Улучшение качества данных
- ✅ **Файл**: `improve_data_quality.py`
- **Что делает**: Очистка, дедупликация, фильтрация данных
- **Запуск**: `python improve_data_quality.py --input data/gogol_processed.txt --output data/gogol_cleaned.txt`

## 📋 Требования

```bash
pip install -r requirements.txt
```

Основные зависимости:
- PyTorch 2.0+
- Transformers 4.35+
- Datasets 2.14+
- PEFT 0.6+
- Accelerate 0.25+

Опциональные:
- NLTK (BLEU метрика)
- rouge-score (ROUGE метрика)
- MLflow (трекинг экспериментов)
- Matplotlib (визуализация)

## 🚀 Быстрый старт

### 1. Проверка окружения
```bash
python run_checks.py
```

### 2. Подготовка данных
```bash
# Валидация датасета
python validate_dataset.py

# Опционально: улучшение качества
python improve_data_quality.py
```

### 3. Обучение
```bash
# С мониторингом (рекомендуется)
python train_with_monitoring.py

# Или простое обучение
python train_simple.py
```

### 4. Тестирование
```bash
# Проверка качества генерации
python test_generation_quality.py

# Автоматические тесты
python test_quality_ci.py
```

### 5. Визуализация
```bash
python visualize_results.py
tensorboard --logdir logs/tensorboard
```

## 📁 Структура проекта

```
DeadSouls/
├── config_unified.py        # Единая конфигурация
├── finetune.py              # Основной скрипт обучения (Llama 8B)
├── train_simple.py          # Простое обучение (GPT-2)
├── train_with_monitoring.py # Обучение с мониторингом
├── data/                    # Датасеты
│   ├── gogol_processed.txt
│   ├── tokenized_gpt2/
│   └── combined_dataset.txt
├── logs/                    # Логи обучения
├── visualization/           # Графики и отчеты
├── create_extended_dataset.py      # Расширение датасета
├── validate_dataset.py             # Валидация
├── improve_data_quality.py         # Улучшение качества
├── test_generation_quality.py      # Тестирование качества
├── debug_repetitions.py            # Отладка повторений
├── test_quality_ci.py              # CI тесты
├── experiment_tracker.py           # Трекинг экспериментов
├── visualize_results.py            # Визуализация
├── run_checks.py                   # Проверка окружения
└── README_improvements.md          # Этот файл
```

## 🛠️ Конфигурация

Все параметры обучения находятся в `config_unified.py`:

```python
# Модель
DEFAULT_MODEL = "gpt2_small"  # или "llama8b"

# Обучение
TRAINING = {
    "num_train_epochs": 7,
    "per_device_train_batch_size": 1,
    "learning_rate": 1e-4,
    "lora_r": 8,
    "lora_alpha": 16,
}

# Мониторинг
MONITORING = {
    "enable_tensorboard": True,
    "enable_wandb": False,
}
```

## 📊 Метрики качества

- **Perplexity**: Ожидаемо < 100 после обучения
- **BLEU**: Ожидаемо > 0.1 для stylistic transfer
- **ROUGE-L**: Ожидаемо > 0.2
- **Repetition Rate**: Ожидаемо < 20%

## 🔧 Отладка

### Проблема: Повторения в генерации
```bash
python debug_repetitions.py
```
Рекомендуемые параметры:
- `temperature`: 0.7-0.8
- `repetition_penalty`: 1.2-1.5
- `top_p`: 0.9-0.95

### Проблема: Ошибки при обучении
```bash
# Проверка датасета
python validate_dataset.py

# Проверка GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 Мониторинг

### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
# Откройте http://localhost:6006 в браузере
```

### MLflow (если установлен)
```bash
mlflow ui
# Откройте http://localhost:5000
```

## 🧪 Эксперименты

Для отслеживания экспериментов:

```python
from experiment_tracker import ExperimentTracker

tracker = ExperimentTracker("deadsouls-gogol")
tracker.start_run(
    name="experiment_v1",
    params={"lr": 1e-4, "epochs": 7}
)

# Во время обучения
tracker.log_metric("eval_loss", 3.8, step=100)

tracker.end_run()
```

## 📝 Следующие шаги

1. ✅ Все 10 рекомендаций реализованы
2. 🔄 Запустите `python run_checks.py` для проверки
3. 🔄 Обучите модель: `python train_with_monitoring.py`
4. 🔄 Протестируйте: `python test_quality_ci.py`
5. 🔄 Визуализируйте: `python visualize_results.py`

## 📄 Лицензия

Проект создан в образовательных целях. Тексты Гоголя находятся в общественном достоянии.

## 👥 Контакты

Проект: DeadSouls  
Репозиторий: `C:/Code/JuliaProject/DeadSouls`
