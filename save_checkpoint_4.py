#!/usr/bin/env python3
"""
Сохранение чекпоинта эпохи 4 и подготовка к продолжению обучения
"""

import os
import shutil
import json
from datetime import datetime

def main():
    print("=== Сохранение чекпоинта эпохи 4 ===\n")
    
    # Пути
    CHECKPOINT_FROM = "./gogol_finetuned_final/checkpoint-1125"  # Эпоха 3
    CHECKPOINT_CURRENT = "./gogol_finetuned_to_5epochs"  # Текущее обучение
    
    # Проверяем, существует ли новое обучение
    if not os.path.exists(CHECKPOINT_CURRENT):
        print(f"Ошибка: папка обучения не найдена: {CHECKPOINT_CURRENT}")
        print("Сначала запустите обучение: python resume_to_5epochs.py")
        return
    
    # Создаём резервную копию чекпоинта эпохи 3
    backup_dir = "./gogol_finetuned_final/checkpoint-epoch4_backup"
    if os.path.exists(backup_dir):
        print(f"Удаляем старую резервную копию: {backup_dir}")
        shutil.rmtree(backup_dir)
    
    print(f"Сохраняем чекпоинт эпохи 4 в: {backup_dir}")
    
    # Копируем файлы
    files_to_copy = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "rng_state.pth"
    ]
    
    saved_files = []
    for f in files_to_copy:
        src = os.path.join(CHECKPOINT_CURRENT, f)
        dst = os.path.join(backup_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            saved_files.append(f)
            print(f"  ✅ {f}")
        else:
            print(f"  ⚠️  {f} (не найден)")
    
    # Копируем токенизатор
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt"
    ]
    
    for f in tokenizer_files:
        src = os.path.join(CHECKPOINT_CURRENT, f)
        dst = os.path.join(backup_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✅ tokenizer/{f}")
    
    # Создаём metadata
    metadata = {
        "checkpoint_name": "checkpoint_epoch4",
        "timestamp": datetime.now().isoformat(),
        "epoch": 4.0,
        "original_checkpoint": CHECKPOINT_FROM,
        "files_saved": saved_files,
        "notes": "Резервная копия перед продолжением обучения до 5 эпох"
    }
    
    with open(os.path.join(backup_dir, "checkpoint_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Чекпоинт сохранён: {backup_dir}")
    print(f"   Файлов сохранено: {len(saved_files)}")
    print(f"   Время: {metadata['timestamp']}")
    
    # Выводим текущий прогресс из trainer_state.json
    state_file = os.path.join(CHECKPOINT_CURRENT, "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        print(f"\n📊 Текущий прогресс:")
        print(f"   Epoch: {state.get('epoch', 'N/A')}")
        print(f"   Global Step: {state.get('global_step', 'N/A')}")
        print(f"   Best Metric: {state.get('best_metric', 'N/A')}")
        
        # Последние логи
        history = state.get("log_history", [])
        if history:
            print(f"\n📈 Последние метрики:")
            for log in history[-3:]:
                if "eval_loss" in log:
                    print(f"   Epoch {log.get('epoch')}: eval_loss = {log['eval_loss']:.4f}")
                else:
                    print(f"   Step {log.get('step')}: loss = {log.get('loss', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
