#!/usr/bin/env python3
"""
Мониторинг процесса обучения в реальном времени
"""

import os
import json
import time
from datetime import datetime

CHECKPOINT_DIR = "./gogol_finetuned_final"
CHECK_INTERVAL = 5  # секунд

def get_training_status():
    """Получить текущий статус обучения"""
    state_file = os.path.join(CHECKPOINT_DIR, "trainer_state.json")
    
    if not os.path.exists(state_file):
        return None
    
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        # Получить последние метрики
        history = state.get("log_history", [])
        latest_eval = None
        latest_train = None
        
        for log in reversed(history):
            if "eval_loss" in log and latest_eval is None:
                latest_eval = log
            if "loss" in log and latest_train is None:
                latest_train = log
        
        return {
            "epoch": state.get("epoch"),
            "max_epochs": state.get("num_train_epochs"),
            "global_step": state.get("global_step"),
            "max_steps": state.get("max_steps"),
            "best_metric": state.get("best_metric"),
            "latest_eval": latest_eval,
            "latest_train": latest_train,
            "should_stop": state.get("stateful_callbacks", {}).get(
                "TrainerControl", {}
            ).get("attributes", {}).get("should_training_stop", False)
        }
    except Exception as e:
        return {"error": str(e)}

def print_status(status):
    """Вывести статус в консоль"""
    if not status or "error" in status:
        print(f"Ошибка: {status.get('error', 'Неизвестная ошибка')}")
        return
    
    print("\n" + "="*60)
    print("СТАТУС ОБУЧЕНИЯ")
    print("="*60)
    
    # Основная информация
    epoch = status.get("epoch", 0)
    max_epochs = status.get("max_epochs", 7)
    global_step = status.get("global_step", 0)
    max_steps = status.get("max_steps", 2625)
    
    print(f"Эпоха: {epoch:.2f} / {max_epochs}")
    print(f"Шаг: {global_step} / {max_steps}")
    
    progress = (global_step / max_steps) * 100 if max_steps > 0 else 0
    print(f"Прогресс: {progress:.1f}%")
    
    # Метрики
    if status.get("best_metric"):
        print(f"Best Eval Loss: {status['best_metric']:.4f}")
    
    if status.get("latest_eval"):
        eval_log = status["latest_eval"]
        print(f"\nПоследняя оценка:")
        print(f"  Epoch: {eval_log.get('epoch'):.2f}")
        print(f"  Eval Loss: {eval_log.get('eval_loss'):.4f}" if eval_log.get('eval_loss') else "")
        print(f"  Runtime: {eval_log.get('eval_runtime', 0):.1f} сек" if eval_log.get('eval_runtime') else "")
    
    if status.get("latest_train"):
        train_log = status["latest_train"]
        print(f"\nПоследний train step:")
        print(f"  Loss: {train_log.get('loss'):.4f}" if train_log.get('loss') else "")
        print(f"  Learning Rate: {train_log.get('learning_rate', 0):.2e}" if train_log.get('learning_rate') else "")
    
    # Статус
    print(f"\n{'='*60}")
    if status.get("should_stop"):
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    else:
        print("Обучение в процессе...")
    print("="*60)

def main():
    print("Мониторинг обучения (нажмите Ctrl+C для выхода)")
    
    last_epoch: int = 0
    
    try:
        while True:
            status = get_training_status()
            
            if status and status.get("epoch") is not None:
                current_epoch = status.get("epoch", 0)
                # Приведение к float для корректного сравнения
                if isinstance(current_epoch, (int, float)) and current_epoch > last_epoch:
                    # Новая эпоха - добавить пустую строку для разделения
                    print()
                    last_epoch = int(current_epoch)
            
            print_status(status)
            
            if status and status.get("should_stop"):
                print("\nОбучение завершено. Остановка мониторинга.")
                break
            
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nМониторинг остановлен пользователем")

if __name__ == "__main__":
    main()
