#!/usr/bin/env python3
"""
Ручное резервное копирование чекпоинтов
"""

import os
import shutil
import json
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Ручное резервное копирование чекпоинтов")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Имя чекпоинта (например, checkpoint-1875). Если не указано, берётся последний.")
    parser.add_argument("--name", type=str, default=None,
                       help="Имя для резервной копии (опционально)")
    parser.add_argument("--output-dir", type=str, default="./gogol_finetuned_backups",
                       help="Папка для резервных копий")
    
    args = parser.parse_args()
    
    checkpoint_dir = "./gogol_finetuned_final"
    
    # Проверка папки
    if not os.path.exists(checkpoint_dir):
        print(f"Ошибка: папка обучения не найдена: {checkpoint_dir}")
        return
    
    # Получить список чекпоинтов
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("checkpoint-"):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, item))
            except ValueError:
                continue
    
    checkpoints.sort(key=lambda x: x[0])
    
    if not checkpoints:
        print("Ошибка: чекпоинтов не найдено")
        return
    
    # Выбрать чекпоинт
    if args.checkpoint:
        target_checkpoint = args.checkpoint
    else:
        _, target_checkpoint = checkpoints[-1]
        print(f"Чекпоинт не указан, выбран последний: {target_checkpoint}")
    
    # Проверить существование
    source_path = os.path.join(checkpoint_dir, target_checkpoint)
    if not os.path.exists(source_path):
        print(f"Ошибка: чекпоинт не найден: {target_checkpoint}")
        return
    
    # Получить информацию о чекпоинте
    state_file = os.path.join(source_path, "trainer_state.json")
    checkpoint_info = {}
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        checkpoint_info = {
            "epoch": state.get("epoch"),
            "global_step": state.get("global_step"),
            "best_metric": state.get("best_metric")
        }
    
    # Создать имя резервной копии
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.name:
        backup_name = f"{args.name}_{timestamp}"
    elif checkpoint_info:
        epoch_str = f"epoch{checkpoint_info['epoch']:.1f}"
        backup_name = f"{target_checkpoint}_{epoch_str}_{timestamp}"
    else:
        backup_name = f"{target_checkpoint}_{timestamp}"
    
    backup_path = os.path.join(args.output_dir, backup_name)
    
    # Создать папку для резервных копий
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Копировать
    print(f"Копирование: {target_checkpoint} -> {backup_name}")
    try:
        shutil.copytree(source_path, backup_path)
        
        # Сохранить метаданные
        metadata = {
            "original_checkpoint": target_checkpoint,
            "backup_name": backup_name,
            "backup_timestamp": timestamp,
            "epoch": checkpoint_info.get("epoch"),
            "global_step": checkpoint_info.get("global_step"),
            "best_metric": checkpoint_info.get("best_metric"),
            "custom_name": args.name
        }
        
        with open(os.path.join(backup_path, "backup_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"УСПЕШНО: резервная копия создана")
        print(f"Путь: {backup_path}")
        print(f"Эпоха: {checkpoint_info.get('epoch'):.2f}" if checkpoint_info.get('epoch') else "")
        print(f"Best Loss: {checkpoint_info.get('best_metric'):.4f}" if checkpoint_info.get('best_metric') else "")
        
    except Exception as e:
        print(f"ОШИБКА: {e}")

if __name__ == "__main__":
    main()
