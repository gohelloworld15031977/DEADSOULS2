#!/usr/bin/env python3
"""
Автоматическое резервное копирование чекпоинтов после каждой эпохи
Запускается параллельно с обучением
"""

import os
import shutil
import json
import time
from datetime import datetime
from pathlib import Path

# Конфигурация
CHECKPOINT_DIR = "./gogol_finetuned_final"
BACKUP_BASE_DIR = "./gogol_finetuned_backups"
MAX_BACKUPS_TO_KEEP = 5
CHECK_INTERVAL_SECONDS = 60  # Проверка каждые 60 секунд

def get_checkpoints():
    """Получить список чекпоинтов, отсортированный по шагу"""
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    
    checkpoints = []
    for item in os.listdir(CHECKPOINT_DIR):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, item))
            except ValueError:
                continue
    
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def get_checkpoint_info(checkpoint_name):
    """Получить информацию о чекпоинте из trainer_state.json"""
    state_file = os.path.join(CHECKPOINT_DIR, checkpoint_name, "trainer_state.json")
    if not os.path.exists(state_file):
        return None
    
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        return {
            "epoch": state.get("epoch"),
            "global_step": state.get("global_step"),
            "best_metric": state.get("best_metric"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Ошибка чтения {checkpoint_name}: {e}")
        return None

def backup_checkpoint(checkpoint_name, checkpoint_info):
    """Создать резервную копию чекпоинта"""
    source_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
    
    if not os.path.exists(source_path):
        return False
    
    # Создаём папку для резервных копий с меткой времени
    backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{checkpoint_name}_epoch{checkpoint_info['epoch']:.1f}_{backup_timestamp}"
    backup_path = os.path.join(BACKUP_BASE_DIR, backup_name)
    
    try:
        # Копируем чекпоинт
        shutil.copytree(source_path, backup_path)
        
        # Сохраняем метаданные
        metadata = {
            "original_checkpoint": checkpoint_name,
            "backup_timestamp": backup_timestamp,
            "epoch": checkpoint_info["epoch"],
            "global_step": checkpoint_info["global_step"],
            "best_metric": checkpoint_info["best_metric"]
        }
        
        with open(os.path.join(backup_path, "backup_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Ошибка резервного копирования {checkpoint_name}: {e}")
        return False

def cleanup_old_backups():
    """Удалить старые резервные копии, оставив только MAX_BACKUPS_TO_KEEP"""
    if not os.path.exists(BACKUP_BASE_DIR):
        return
    
    backups = []
    for item in os.listdir(BACKUP_BASE_DIR):
        backup_path = os.path.join(BACKUP_BASE_DIR, item)
        if os.path.isdir(backup_path):
            backups.append((os.path.getmtime(backup_path), item))
    
    # Сортировка по времени (от newest к oldest)
    backups.sort(key=lambda x: x[0], reverse=True)
    
    # Удаляем старые
    if len(backups) > MAX_BACKUPS_TO_KEEP:
        for _, backup_name in backups[MAX_BACKUPS_TO_KEEP:]:
            backup_path = os.path.join(BACKUP_BASE_DIR, backup_name)
            try:
                shutil.rmtree(backup_path)
                print(f"Удалена старая резервная копия: {backup_name}")
            except Exception as e:
                print(f"Ошибка удаления {backup_name}: {e}")

def log_backup(checkpoint_name, success, info):
    """Добавить запись в лог резервных копий"""
    log_file = os.path.join(BACKUP_BASE_DIR, "backup_log.json")
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": checkpoint_name,
        "success": success,
        "epoch": info.get("epoch") if info else None,
        "best_metric": info.get("best_metric") if info else None
    }
    
    # Загрузка существующего лога
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log = json.load(f)
        except:
            log = []
    else:
        log = []
    
    log.append(log_data)
    
    # Сохранение
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

def monitor_training():
    """Основной цикл мониторинга и резервного копирования"""
    print("="*60)
    print("АВТОМАТИЧЕСКОЕ РЕЗЕРВНОЕ КОПИРОВАНИЕ ЧЕКПОИНТОВ")
    print("="*60)
    print(f"Папка обучения: {CHECKPOINT_DIR}")
    print(f"Папка резервных копий: {BACKUP_BASE_DIR}")
    print(f"Интервал проверки: {CHECK_INTERVAL_SECONDS} сек")
    print(f"Максимум резервных копий: {MAX_BACKUPS_TO_KEEP}")
    print("="*60)
    print()
    
    # Создаём папку для резервных копий
    os.makedirs(BACKUP_BASE_DIR, exist_ok=True)
    
    last_backup_step = 0
    training_active = True
    
    try:
        while training_active:
            # Получаем текущие чекпоинты
            checkpoints = get_checkpoints()
            
            if not checkpoints:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Чекпоинтов не найдено, жду...")
                time.sleep(CHECK_INTERVAL_SECONDS)
                continue
            
            # Берём последний чекпоинт
            latest_step, latest_name = checkpoints[-1]
            
            # Проверяем, был ли уже зарезервирован
            if latest_step > last_backup_step:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Найдён новый чекпоинт: {latest_name}")
                
                # Получаем информацию
                info = get_checkpoint_info(latest_name)
                
                if info:
                    print(f"  Эпоха: {info['epoch']:.2f}")
                    print(f"  Global Step: {info['global_step']}")
                    print(f"  Best Metric: {info['best_metric']:.4f}" if info['best_metric'] else "")
                    
                    # Создаём резервную копию
                    print(f"  Создание резервной копии...")
                    success = backup_checkpoint(latest_name, info)
                    
                    if success:
                        print(f"  УСПЕШНО: резервная копия создана")
                        last_backup_step = latest_step
                    else:
                        print(f"  ОШИБКА: не удалось создать резервную копию")
                    
                    # Очищаем старые резервные копии
                    cleanup_old_backups()
                    
                    # Записываем в лог
                    log_backup(latest_name, success, info)
                else:
                    print(f"  ОШИБКА: не удалось получить информацию о чекпоинте")
            
            # Проверяем, закончилось ли обучение
            # Проверяем trainer_state.json в основной папке
            state_file = os.path.join(CHECKPOINT_DIR, "trainer_state.json")
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    
                    current_epoch = state.get("epoch", 0)
                    max_epochs = state.get("num_train_epochs", 7)
                    should_stop = state.get("stateful_callbacks", {}).get(
                        "TrainerControl", {}
                    ).get("attributes", {}).get("should_training_stop", False)
                    
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Обучение: эпоха {current_epoch:.2f}/{max_epochs}", end="")
                    
                    if should_stop or current_epoch >= max_epochs:
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ОБУЧЕНИЕ ЗАВЕРШЕНО!")
                        training_active = False
                        break
                except Exception as e:
                    print(f"\n[Ошибка чтения состояния обучения: {e}]")
            
            time.sleep(CHECK_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Мониторинг прерван пользователем")
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Ошибка: {e}")
    
    # Финальный отчёт
    print("\n" + "="*60)
    print("ФИНАЛЬНЫЙ ОТЧЁТ РЕЗЕРВНОГО КОПИРОВАНИЯ")
    print("="*60)
    
    backups = [d for d in os.listdir(BACKUP_BASE_DIR) 
               if os.path.isdir(os.path.join(BACKUP_BASE_DIR, d)) and d != "runs"]
    
    print(f"Всего резервных копий: {len(backups)}")
    for backup in sorted(backups):
        backup_path = os.path.join(BACKUP_BASE_DIR, backup)
        metadata_file = os.path.join(backup_path, "backup_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"  {backup}:")
            print(f"    Эпоха: {meta.get('epoch'):.2f}")
            print(f"    Step: {meta.get('global_step')}")
            print(f"    Best Loss: {meta.get('best_metric'):.4f}" if meta.get('best_metric') else "")
        else:
            print(f"  {backup}: (без метаданных)")
    
    print(f"\nПапка с резервными копиями: {BACKUP_BASE_DIR}")
    print("="*60)

if __name__ == "__main__":
    monitor_training()
