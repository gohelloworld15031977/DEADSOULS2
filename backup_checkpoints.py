#!/usr/bin/env python3
"""
Резервное копирование существующих чекпоинтов и поиск потерянных данных
"""

import os
import shutil
import json
from datetime import datetime

def main():
    print("=== Резервное копирование чекпоинтов ===\n")
    
    # Основная папка с обучением
    base_dir = "./gogol_finetuned_final"
    
    if not os.path.exists(base_dir):
        print(f"Ошибка: папка обучения не найдена: {base_dir}")
        return
    
    # Создаём резервную папку
    backup_dir = f"./gogol_finetuned_final_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"Резервная папка: {backup_dir}\n")
    
    # Проверяем все чекпоинты
    checkpoints = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if item.startswith("checkpoint-") and os.path.isdir(item_path):
            checkpoints.append(item)
    
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    
    print(f"Найдено чекпоинтов: {len(checkpoints)}")
    for cp in checkpoints:
        cp_path = os.path.join(base_dir, cp)
        files = os.listdir(cp_path)
        size = sum(os.path.getsize(os.path.join(cp_path, f)) for f in files if os.path.isfile(os.path.join(cp_path, f)))
        print(f"  {cp}: {len(files)} файлов, {size/1024/1024:.2f} MB")
    
    # Копируем все чекпоинты
    print(f"\nКопирование чекпоинтов в резервную папку...")
    for cp in checkpoints:
        src = os.path.join(base_dir, cp)
        dst = os.path.join(backup_dir, cp)
        if os.path.exists(dst):
            print(f"  ⚠️  {cp} уже существует в резерве, пропускаем")
            continue
        shutil.copytree(src, dst)
        print(f"  ✅ {cp} скопирован")
    
    # Копируем основные файлы
    print(f"\nКопирование основных файлов...")
    main_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    for f in main_files:
        src = os.path.join(base_dir, f)
        dst = os.path.join(backup_dir, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✅ {f}")
    
    # Считываем текущее состояние
    state_file = os.path.join(backup_dir, "checkpoint-375", "trainer_state.json")
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        print(f"\n=== ТЕКУЩЕЕ СОСТОЯНИЕ ОБУЧЕНИЯ ===")
        print(f"Эпоха: {state.get('epoch')}")
        print(f"Global Step: {state.get('global_step')}")
        print(f"Best Metric (Eval Loss): {state.get('best_metric'):.4f}")
        print(f"Max Steps: {state.get('max_steps')}")
        print(f"Num Train Epochs: {state.get('num_train_epochs')}")
        
        # История обучения
        history = state.get("log_history", [])
        if history:
            print(f"\n📊 История обучения (последние 5 записей):")
            for log in history[-5:]:
                if "eval_loss" in log:
                    print(f"  Epoch {log.get('epoch')}: eval_loss = {log['eval_loss']:.4f}")
                else:
                    print(f"  Step {log.get('step')}: loss = {log.get('loss', 'N/A'):.4f}")
    
    # Создаём metadata
    metadata = {
        "backup_timestamp": datetime.now().isoformat(),
        "source_directory": base_dir,
        "checkpoints_found": checkpoints,
        "total_checkpoints": len(checkpoints),
        "current_epoch": state.get('epoch') if os.path.exists(state_file) else None,
        "best_metric": state.get('best_metric') if os.path.exists(state_file) else None,
        "notes": "Резервная копия после обнаружения перезапуска обучения"
    }
    
    with open(os.path.join(backup_dir, "backup_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("РЕЗЕРВНОЕ КОПИРОВАНИЕ ЗАВЕРШЕНО")
    print(f"{'='*60}")
    print(f"Папка резервной копии: {backup_dir}")
    print(f"Чекпоинтов сохранено: {len(checkpoints)}")
    print(f"\n⚠️  ВНИМАНИЕ: Обучение началось заново с эпохи 1!")
    print(f"   Потеряны чекпоинты эпох 4-5 (checkpoint-1500, checkpoint-1875)")
    print(f"   Текущее состояние: эпоха {state.get('epoch') if os.path.exists(state_file) else 'N/A'}")
    
    # Проверяем, есть ли старые резервные копии
    print(f"\n{'='*60}")
    print("ПОИСК СТАРЫХ РЕЗЕРВНЫХ КОПИЙ")
    print(f"{'='*60}")
    
    old_backups = [d for d in os.listdir(".") if d.startswith("gogol_finetuned_final_backup")]
    if old_backups:
        print("Найдены старые резервные копии:")
        for backup in sorted(old_backups):
            print(f"  - {backup}")
            
            # Проверяем наличие checkpoint-1500 и checkpoint-1875
            cp1500 = os.path.join(backup, "checkpoint-1500")
            cp1875 = os.path.join(backup, "checkpoint-1875")
            
            if os.path.exists(cp1500):
                print(f"    ✅ checkpoint-1500 (эпоха 4) НАЙДЕН!")
            if os.path.exists(cp1875):
                print(f"    ✅ checkpoint-1875 (эпоха 5) НАЙДЕН!")
    else:
        print("Старых резервных копий не найдено.")
        print("Чекпоинты эпох 4-5, к сожалению, утеряны.")

if __name__ == "__main__":
    main()
