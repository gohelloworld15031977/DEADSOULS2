# 📋 Инструкция по резервному копированию чекпоинтов

## 🚀 Автоматическое резервное копирование

### Запуск параллельно с обучением

**В терминале 1 (основное обучение):**
```powershell
.\venv\Scripts\python.exe continue_to_epoch5.py
```

**В терминале 2 (мониторинг и резервное копирование):**
```powershell
.\venv\Scripts\python.exe auto_backup_checker.py
```

### Что делает скрипт:

1. **Проверяет** папку `./gogol_finetuned_final/` каждые 60 секунд
2. **Обнаруживает** новые чекпоинты после каждой эпохи
3. **Копирует** чекпоинт в `./gogol_finetuned_backups/`
4. **Сохраняет** метаданные (epoch, eval_loss, timestamp)
5. **Очищает** старые резервные копии (оставляет 5 последних)
6. **Ведёт лог** всех операций в `backup_log.json`

### Пример вывода:
```
============================================================
АВТОМАТИЧЕСКОЕ РЕЗЕРВНОЕ КОПИРОВАНИЕ ЧЕКПОИНТОВ
============================================================
Папка обучения: ./gogol_finetuned_final
Папка резервных копий: ./gogol_finetuned_backups
Интервал проверки: 60 сек
Максимум резервных копий: 5
============================================================

[10:15:32] Найдён новый чекпоинт: checkpoint-750
  Эпоха: 2.00
  Global Step: 750
  Best Metric: 3.7954
  Создание резервной копии...
  УСПЕШНО: резервная копия создана

[10:15:32] Обучение: эпоха 2.15/5
```

---

## 📁 Структура резервных копий

```
gogol_finetuned_backups/
├── checkpoint-750_epoch2.0_20260418_101532/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── trainer_state.json
│   ├── backup_metadata.json
│   └── ...
├── checkpoint-1125_epoch3.0_20260418_111545/
├── checkpoint-1500_epoch4.0_20260418_121600/
├── checkpoint-1875_epoch5.0_20260418_131615/
└── backup_log.json
```

### Метаданные резервной копии (`backup_metadata.json`):
```json
{
  "original_checkpoint": "checkpoint-1875",
  "backup_timestamp": "20260418_131615",
  "epoch": 5.0,
  "global_step": 1875,
  "best_metric": 3.7548
}
```

---

## 🛠️ Ручное резервное копирование

### Скрипт `manual_backup.py`:

```powershell
# Создать резервную копию текущего лучшего чекпоинта
.\venv\Scripts\python.exe manual_backup.py

# Создать резервную копию конкретного чекпоинта
.\venv\Scripts\python.exe manual_backup.py --checkpoint checkpoint-1875

# Создать резервную копию с именем (для маркировки)
.\venv\Scripts\python.exe manual_backup.py --name "final_epoch5"
```

---

## ⚡ Быстрые команды

### Проверить наличие резервных копий:
```powershell
Get-ChildItem -Directory ./gogol_finetuned_backups | Select-Object Name
```

### Посмотреть логи резервного копирования:
```powershell
.\venv\Scripts\python.exe -c "import json; data=json.load(open('./gogol_finetuned_backups/backup_log.json')); print(json.dumps(data, indent=2, ensure_ascii=False))"
```

### Восстановить чекпоинт из резервной копии:
```powershell
# Копировать обратно в основную папку
Copy-Item -Path "./gogol_finetuned_backups/checkpoint-1875_epoch5.0_*/" -Destination "./gogol_finetuned_final/" -Recurse -Force
```

---

## 🎯 Рекомендации

### Перед началом обучения:
1. ✅ Очистить старые чекпоинты (если нужно)
2. ✅ Запустить `auto_backup_checker.py` в фоне
3. ✅ Проверить, что резервное копирование работает

### Во время обучения:
1. ✅ Каждые 2-3 часа проверять папку `./gogol_finetuned_backups/`
2. ✅ Следить, чтобы новые резервные копии создаются после каждой эпохи
3. ✅ При необходимости остановить и проверить

### После завершения обучения:
1. ✅ Убедиться, что чекпоинт финальной эпохи зарезервирован
2. ✅ Скопировать лучшую модель в отдельную папку `./final_models/`
3. ✅ Протестировать генерацию

---

## 📊 Пример сценария обучения

```
10:00 - Запуск обучения (continue_to_epoch5.py)
10:00 - Запуск мониторинга (auto_backup_checker.py)
11:00 - Эпоха 1 завершена → checkpoint-375 → резервная копия создана
12:00 - Эпоха 2 завершена → checkpoint-750 → резервная копия создана
13:00 - Эпоха 3 завершена → checkpoint-1125 → резервная копия создана
14:00 - Эпоха 4 завершена → checkpoint-1500 → резервная копия создана
15:00 - Эпоха 5 завершена → checkpoint-1875 → резервная копия создана
15:05 - Обучение завершено → финальный отчёт
```

---

## 🚨 Что делать в случае сбоя

### Если обучение прервалось:
1. Проверить `./gogol_finetuned_backups/` на наличие последней резервной копии
2. Запустить обучение с чекпоинта:
   ```powershell
   .\venv\Scripts\python.exe resume_training.py --checkpoint ./gogol_finetuned_backups/checkpoint-1125_epoch3.0_*
   ```

### Если резервные копии утеряны:
1. Проверить `gogol_finetuned_final/checkpoint-*/` — чекпоинты могут быть там
2. Если нет — начать обучение заново с автоматическим резервным копированием

---

## 📞 Контакты

При возникновении проблем:
- Проверить логи: `./gogol_finetuned_backups/backup_log.json`
- Посмотреть вывод `auto_backup_checker.py`
- Убедиться, что достаточно места на диске

---

**Обновлено:** 2026-04-18  
**Версия скрипта:** 1.0
