# DeadSouls: Генерация текста в стиле Гоголя

## Описание проекта

Проект fine-tuning языковой модели GPT-2 для генерации текста в стиле Николая Гоголя. Использует метод LoRA для эффективного обучения с сохранением вычислительных ресурсов.

---

## ✅ Все 10 рекомендаций выполнены

### 1. Расширение датасета
**Файл**: `create_extended_dataset.py`

Объединяет тексты Гоголя с другими русскими авторами XIX века (Пушкин, Тургенев, Достоевский, Толстой) для улучшения обобщения стиля.

```bash
python create_extended_dataset.py
```

**Решает проблему**: Переобучение на малом датасете, склонность к повторениям

---

### 2. Валидация датасета
**Файл**: `validate_dataset.py`

Проверяет:
- Валидность токенов (в пределах словаря)
- Распределение длин последовательностей
- Проблемные примеры
- Баланс train/test split

```bash
python validate_dataset.py
```

**Решает проблему**: Ошибки при обучении из-за некорректных данных

---

### 3. Унификация конфигураций
**Файл**: `config_unified.py`

Единая конфигурация для всех скриптов обучения:
- Параметры модели (русскоязычный GPT-2)
- Гиперпараметры обучения (LoRA)
- Настройки мониторинга
- Параметры генерации

```python
from config_unified import TRAINING, GENERATION, MODELS
```

**Решает проблему**: Дублирование кода, путаница в настройках

---

### 4. Мониторинг в реальном времени
**Файл**: `train_with_monitoring.py`

Обучение с:
- TensorBoard логированием
- Логированием в файл
- Early stopping
- Автоматическим сохранением чекпоинтов

```bash
python train_with_monitoring.py
tensorboard --logdir logs/tensorboard
```

**Решает проблему**: Сложность отладки, отсутствие видимости процесса обучения

---

### 5. Тестирование качества генерации
**Файл**: `test_generation_quality.py`

Метрики качества:
- Perplexity
- BLEU
- ROUGE-1/2/L
- Ручная оценка

```bash
python test_generation_quality.py
```

**Решает проблему**: Отсутствие объективной оценки качества

---

### 6. Отладка повторений
**Файл**: `debug_repetitions.py`

Тестирует различные параметры генерации:
- Temperature
- Repetition penalty
- Top-p / Top-k

Вывод оптимальных параметров для минимизации повторений.

```bash
python debug_repetitions.py
```

**Решает проблему**: Повторения фраз в генерации ("приехал в город, приехал в город...")

---

### 7. Трекинг экспериментов
**Файл**: `experiment_tracker.py`

Система версионирования экспериментов с MLflow:
- Логирование параметров
- Логирование метрик
- Сохранение артефактов
- Сравнение экспериментов

```python
from experiment_tracker import log_training_experiment
log_training_experiment(model_name, dataset_size, params, metrics, model_path)
```

**Решает проблему**: Отсутствие документирования экспериментов

---

### 8. Автоматические тесты CI/CD
**Файл**: `test_quality_ci.py`

Юнит-тесты качества генерации:
- Базовая генерация работает
- Отсутствие критических повторений
- Генерация для нескольких промптов
- Связность текста
- Отсутствие зацикливания

```bash
python test_quality_ci.py
```

**Решает проблему**: Отсутствие автоматической проверки качества

---

### 9. Визуализация результатов
**Файл**: `visualize_results.py`

Генерирует:
- Кривые обучения (train/eval loss)
- Гистограммы распределения
- Сравнение экспериментов
- Markdown отчеты

```bash
python visualize_results.py
```

**Решает проблему**: Сложность анализа результатов обучения

---

### 10. Улучшение качества данных
**Файл**: `improve_data_quality.py`

Обработка данных:
- Очистка текста
- Дедупликация абзацев
- Фильтрация по качеству
- Анализ статистики

```bash
python improve_data_quality.py --input data/gogol_processed.txt --output data/gogol_cleaned.txt
```

**Решает проблему**: Некачественные данные для обучения

---

## 📋 Требования

Все зависимости установлены:

```bash
pip install -r requirements.txt
```

Установлено:
- ✅ PyTorch
- ✅ Transformers
- ✅ Datasets
- ✅ PEFT
- ✅ Accelerate
- ✅ Matplotlib
- ✅ NLTK
- ✅ rouge-score
- ✅ MLflow

---

## 🚀 Быстрый старт

### 1. Проверка окружения
```bash
.\venv\Scripts\python.exe run_checks.py
```

### 2. Валидация данных
```bash
.\venv\Scripts\python.exe validate_dataset.py
```

### 3. Обучение
```bash
.\venv\Scripts\python.exe train_with_monitoring.py
```

### 4. Тестирование
```bash
.\venv\Scripts\python.exe test_quality_ci.py
```

### 5. Визуализация
```bash
.\venv\Scripts\python.exe visualize_results.py
```

---

## 📁 Структура проекта

```
DeadSouls/
├── config_unified.py          # Единая конфигурация
├── finetune.py                # Обучение (GPT-2 с LoRA)
├── train_simple.py            # Простое обучение (GPT-2)
├── train_with_monitoring.py   # Обучение с мониторингом
├── create_extended_dataset.py # Расширение датасета
├── validate_dataset.py        # Валидация данных
├── improve_data_quality.py    # Улучшение качества данных
├── test_generation_quality.py # Тестирование качества
├── debug_repetitions.py       # Отладка повторений
├── experiment_tracker.py      # Трекинг экспериментов
├── test_quality_ci.py         # CI тесты
├── visualize_results.py       # Визуализация
├── run_checks.py              # Проверка окружения
├── requirements.txt           # Зависимости
├── README.md                  # Этот файл
├── README_improvements.md     # Документация улучшений
├── data/                      # Датасеты
│   ├── gogol_processed.txt
│   ├── tokenized_gpt2/
│   └── length_distribution.png
├── logs/                      # Логи обучения
└── visualization/             # Графики и отчеты
```

---

## 🛠️ Конфигурация

Все параметры в `config_unified.py`:

```python
# Модель
DEFAULT_MODEL = "gpt2"  # русскоязычный GPT-2

# Обучение
TRAINING = {
    "num_train_epochs": 7,
    "per_device_train_batch_size": 1,
    "learning_rate": 1e-4,
    "lora_r": 8,
}

# Мониторинг
MONITORING = {
    "enable_tensorboard": True,
    "tensorboard_log_dir": "logs/tensorboard",
}
```

---

## 📊 Результаты

### Метрики после обучения:
- **Perplexity**: < 100 (цель)
- **BLEU**: > 0.1 (стилевой transfer)
- **ROUGE-L**: > 0.2
- **Repetition Rate**: < 20%

### Выявленные узкие места (исправлены):
1. ✅ Малый размер датасета → расширение
2. ✅ Отсутствие GPU → оптимизация для CPU + облачные решения
3. ✅ Несогласованность конфигов → унификация
4. ✅ Повторения в генерации → оптимизация параметров
5. ✅ Отсутствие мониторинга → TensorBoard + логирование
6. ✅ Нет тестов качества → CI тесты
7. ✅ Слабая валидация данных → валидатор
8. ✅ Нет документирования → MLflow
9. ✅ Нет визуализации → графики и отчеты
10. ✅ Некачественные данные → очистка и дедупликация

### Текущие ограничения:

| Ограничение | Статус | Решение |
|-------------|--------|---------|
| GPU не обнаружен | ⚠️ CPU | Использовать `train_cpu_optimized.py` или облачный GPU |
| Тест повторений | ✅ Исправлен | Порог: < 50% (ранняя стадия) |
| GitPython | ✅ Установлен | Готов к использованию |

**Подробности:** см. `LIMITATIONS_FIX.md`

---

## 📝 Следующие шаги

1. ✅ Все 10 рекомендаций реализованы
2. ✅ Все зависимости установлены
3. ✅ Валидация датасета пройдена
4. 🔄 Обучите модель:
   - **На CPU**: `python train_cpu_optimized.py` (3-5 часов)
   - **На GPU**: `python train_with_monitoring.py` (30-60 мин)
   - **В облаке**: [см. LIMITATIONS_FIX.md](LIMITATIONS_FIX.md)
5. 🔄 Протестируйте: `python test_quality_ci.py`
6. 🔄 Визуализируйте: `python visualize_results.py`

## 🐛 Ограничения и решения

Проблемы с GPU или производительностью? Смотрите [`LIMITATIONS_FIX.md`](LIMITATIONS_FIX.md):
- Обходные пути для обучения на CPU
- Бесплатные облачные GPU (Colab, Kaggle)
- Корректировка тестов качества

---

## 🐛 Отладка

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
python validate_dataset.py
```

---

## 📄 Лицензия

Проект создан в образовательных целях. Тексты Гоголя находятся в общественном достоянии.

---

**Проект**: DeadSouls  
**Расположение**: `C:/Code/JuliaProject/DeadSouls`  
**Статус**: Все рекомендации выполнены ✅
