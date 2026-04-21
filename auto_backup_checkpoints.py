"""
Автоматическое резервное копирование чекпоинтов после каждой эпохи
Запускается в фоновом режиме параллельно с обучением
"""

import os
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

BACKUP_DIR = "./gogol_finetuned_backups"
TRAINING_DIR = "./gogol_finetuned_from_epoch2"
CHECK_INTERVAL = 30  # Проверка каждые 30 секунд
MAX_BACKUPS = 5  # Хранить максимум 5 резервных копий

def get_latest_checkpoint():
    """Находит последнюю папку checkpoint в директории обучения"""
    if not os.path.exists(TRAINING_DIR):
        return None
    
    checkpoints = []
    for item in os.listdir(TRAINING_DIR):
        if item.startswith("checkpoint-"):
            path = os.path.join(TRAINING_DIR, item)
            if os.path.isdir(path):
                # Извлекаем номер чекпоинта
                try:
                    step = int(item.split("-")[1])
                    checkpoints.append((step, item, path))
                except (IndexError, ValueError):
                    continue
    
    if not checkpoints:
        return None
    
    # Сортируем по номеру и возвращаем последний
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1]

def backup_checkpoint(checkpoint_name, checkpoint_path):
    """Создаёт резервную копию чекпоинта"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{checkpoint_name}_backup_{timestamp}"
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Создание резервной копии: {backup_name}")
    
    try:
        shutil.copytree(checkpoint_path, backup_path)
        
        # Создаём metadata
        metadata = {
            "original_checkpoint": checkpoint_name,
            "backup_timestamp": timestamp,
            "backup_path": backup_path,
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(backup_path, "backup_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Резервная копия создана: {backup_path}")
        return True
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Ошибка при создании резервной копии: {e}")
        return False

def cleanup_old_backups():
    """Удаляет старые резервные копии, оставляя только MAX_BACKUPS последних"""
    if not os.path.exists(BACKUP_DIR):
        return
    
    backups = []
    for item in os.listdir(BACKUP_DIR):
        item_path = os.path.join(BACKUP_DIR, item)
        if os.path.isdir(item_path) and "_backup_" in item:
            backups.append(item)
    
    # Сортируем по времени (по имени, так как там timestamp)
    backups.sort(reverse=True)
    
    # Удаляем старые, если больше MAX_BACKUPS
    if len(backups) > MAX_BACKUPS:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Очистка старых резервных копий (оставляем {MAX_BACKUPS})")
        for backup in backups[MAX_BACKUPS:]:
            backup_path = os.path.join(BACKUP_DIR, backup)
            try:
                shutil.rmtree(backup_path)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Удалено: {backup}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Ошибка при удалении {backup}: {e}")

def main():
    print("="*60)
    print("МОНИТОРИНГ ЧЕКПОИНТОВ (автоматическое резервное копирование)")
    print("="*60)
    print(f"Директория обучения: {TRAINING_DIR}")
    print(f"Директория резервных копий: {BACKUP_DIR}")
    print(f"Интервал проверки: {CHECK_INTERVAL} сек")
    print(f"Максимум резервных копий: {MAX_BACKUPS}")
    print("="*60)
    
    # Создаём директорию для резервных копий
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    last_checkpoint = None
    last_step = -1
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Начал мониторинг...")
    print("Нажмите Ctrl+C для остановки\n")
    
    try:
        while True:
            time.sleep(CHECK_INTERVAL)
            
            latest = get_latest_checkpoint()
            
            if latest:
                step, name, path = latest
                
                # Проверяем, новый ли это чекпоинт
                if step > last_step:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Обнаружен новый чекпоинт: {name}")
                    
                    # Создаём резервную копию
                    if backup_checkpoint(name, path):
                        last_checkpoint = name
                        last_step = step
                        
                        # Очищаем старые резервные копии
                        cleanup_old_backups()
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Чекпоинт: {name} (шаг {step})")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Чекпоинты не найдены")
                
    except KeyboardInterrupt:
        print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] Мониторинг остановлен")
        print("="*60)
        
        # Финальная статистика
        if os.path.exists(BACKUP_DIR):
            backups = [item for item in os.listdir(BACKUP_DIR) 
                      if os.path.isdir(os.path.join(BACKUP_DIR, item)) and "_backup_" in item]
            print(f"Всего резервных копий: {len(backups)}")
            for backup in sorted(backups):
                print(f"  - {backup}")

if __name__ == "__main__":
    main()
