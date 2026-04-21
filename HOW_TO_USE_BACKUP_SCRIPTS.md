# 📖 Инструкция по использованию скриптов резервного копирования

## 🚀 Быстрый старт

### Сценарий 1: Обучение с автоматическим резервным копированием

**Открыть 2 терминала:**

**Терминал 1 (обучение):**
```powershell
cd C:\Code\JuliaProject\DeadSouls
.\venv\Scripts\python.exe continue_to_epoch5.py
```

**Терминал 2 (мониторинг и резервное копирование):**
```powershell
cd C:\Code\JuliaProject\DeadSouls
.\venv\Scripts\python.exe auto_backup_checker.py
```

**Опционально (терминал 3 - мониторинг в реальном времени):**
```powershell
.\venv\Scripts\python.exe monitor_training.py
```

---

## 📊 Скрипты

### 1. `auto_backup_checker.py` — Автоматическое резервное копирование

**Что делает:**
- Проверяет папку обучения каждые 60 секунд
- Создаёт резервные копии после каждой эпохи
- Хранит 5 последних резервных копий
- Ведёт лог всех операций

**Запуск:**
```powershell
.\venv\Scripts\python.exe auto_backup_checker.py
```

**Настройка (в начале скрипта):**
```python
CHECKPOINT_DIR = "./gogol_finetuned_final"  # Папка обучения
BACKUP_BASE_DIR = "./gogol_finetuned_backups"  # Папка резервных копий
MAX_BACKUPS_TO_KEEP = 5  # Максимум копий
CHECK_INTERVAL_SECONDS = 60  # Интервал проверки
```

---

### 2. `manual_backup.py` — Ручное резервное копирование

**Что делает:**
- Создаёт резервную копию конкретного чекпоинта
- Сохраняет метаданные (epoch, eval_loss)

**Запуск:**
```powershell
# Резервная копия последнего чекпоинта
.\venv\Scripts\python.exe manual_backup.py

# Резервная копия конкретного чекпоинта
.\venv\Scripts\python.exe manual_backup.py --checkpoint checkpoint-1875

# С собственным именем
.\venv\Scripts\python.exe manual_backup.py --checkpoint checkpoint-1875 --name "final_model"
```

---

### 3. `monitor_training.py` — Мониторинг в реальном времени

**Что делает:**
- Отображает прогресс обучения
- Показывает последние метрики
- Обновляется каждые 5 секунд

**Запуск:**
```powershell
.\venv\Scripts\python.exe monitor_training.py
```

**Пример вывода:**
```
============================================================
СТАТУС ОБУЧЕНИЯ
============================================================
Эпоха: 2.15 / 5
Шаг: 800 / 2625
Прогресс: 30.5%
Best Eval Loss: 3.7954

Последняя оценка:
  Epoch: 2.00
  Eval Loss: 3.7954
  Runtime: 63.5 сек

Последний train step:
  Loss: 3.9055
  Learning Rate: 8.58e-05
============================================================
Обучение в процессе...
============================================================
```

---

## 📁 Структура папок

```
DeadSouls/
├── gogol_finetuned_final/           # Текущее обучение
│   ├── checkpoint-375/
│   ├── checkpoint-750/
│   ├── checkpoint-1125/
│   ├── adapter_config.json
│   └── trainer_state.json
│
├── gogol_finetuned_backups/         # Резервные копии
│   ├── checkpoint-375_epoch1.0_20260418_101500/
│   ├── checkpoint-750_epoch2.0_20260418_111500/
│   ├── checkpoint-1125_epoch3.0_20260418_121500/
│   └── backup_log.json
│
├── auto_backup_checker.py           # Автоматическое копирование
├── manual_backup.py                 # Ручное копирование
├── monitor_training.py              # Мониторинг
└── HOW_TO_USE_BACKUP_SCRIPTS.md     # Эта инструкция
```

---

## 🎯 Типичный сценарий использования

### 1. Перед обучением:
```powershell
# Запустить мониторинг в фоне
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\python.exe auto_backup_checker.py"

# Запустить обучение
.\venv\Scripts\python.exe continue_to_epoch5.py
```

### 2. Во время обучения:
- `auto_backup_checker.py` автоматически создаст резервные копии после каждой эпохи
- Можно запустить `monitor_training.py` для просмотра прогресса

### 3. После обучения:
```powershell
# Проверить резервные копии
Get-ChildItem ./gogol_finetuned_backups | Select-Object Name

# Посмотреть лог
Get-Content ./gogol_finetuned_backups/backup_log.json

# Создать финальную резервную копию
.\venv\Scripts\python.exe manual_backup.py --name "final_epoch5"
```

---

## 🔍 Проверка резервных копий

### Список всех резервных копий:
```powershell
Get-ChildItem -Directory ./gogol_finetuned_backups | Select-Object Name, LastWriteTime
```

### Информация о конкретной копии:
```powershell
Get-Content "./gogol_finetuned_backups/checkpoint-1875_epoch5.0_*/backup_metadata.json" | ConvertFrom-Json | Format-List
```

### Лог всех операций:
```powershell
.\venv\Scripts\python.exe -c "import json; data=json.load(open('./gogol_finetuned_backups/backup_log.json')); [print(f\"{x['timestamp']}: {x['checkpoint']} - {'OK' if x['success'] else 'FAIL'}\") for x in data]"
```

---

## 🚨 Решение проблем

### Ошибка: "Папка обучения не найдена"
**Решение:** Убедитесь, что обучение запущено и папка `./gogol_finetuned_final/` существует.

### Ошибка: "Недостаточно места на диске"
**Решение:** Очистите старые резервные копии или увеличьте `MAX_BACKUPS_TO_KEEP`.

### Резервные копии не создаются
**Решение:**
1. Проверьте, что `auto_backup_checker.py` запущен
2. Посмотрите вывод скрипта на наличие ошибок
3. Убедитесь, что чекпоинты создаются в `./gogol_finetuned_final/`

---

## 💡 Советы

1. **Запускайте `auto_backup_checker.py` ПЕРЕД началом обучения**
2. **Не останавливайте мониторинг во время обучения**
3. **Проверяйте резервные копии после каждой эпохи**
4. **Сохраняйте финальную модель в отдельную папку**

---

**Обновлено:** 2026-04-18  
**Версия:** 1.0
