#!/usr/bin/env python3
# Скрипт для проверки прогресса обучения
import os
import time
import json

def main():
    print("=== Мониторинг прогресса обучения ===")
    
    # Пути для проверки
    output_dir = "./gogol_finetuned_resumed"
    checkpoint_dir = "./gogol_finetuned_final/checkpoint-375"
    
    # Проверка существования директорий
    if os.path.exists(output_dir):
        print(f"Выходная директория существует: {output_dir}")
        items = os.listdir(output_dir)
        if items:
            print(f"Найдены файлы: {items}")
            # Проверка чекпоинтов
            checkpoints = [d for d in items if d.startswith('checkpoint-') and os.path.isdir(os.path.join(output_dir, d))]
            if checkpoints:
                print(f"Найдены чекпоинты: {checkpoints}")
                latest = sorted(checkpoints)[-1]
                latest_path = os.path.join(output_dir, latest)
                print(f"Последний чекпоинт: {latest}")
                
                # Проверка trainer_state.json
                state_file = os.path.join(latest_path, 'trainer_state.json')
                if os.path.exists(state_file):
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                    print(f"Текущий шаг: {state.get('global_step', 'N/A')}")
                    print(f"Текущая эпоха: {state.get('epoch', 'N/A')}")
                    print(f"Лучший validation loss: {state.get('best_metric', 'N/A')}")
        else:
            print("Директория пуста - обучение еще не сохранило результаты")
    else:
        print(f"Выходная директория не существует: {output_dir}")
        print("Обучение либо еще не началось, либо завершилось с ошибкой")
    
    # Проверка исходного чекпоинта
    if os.path.exists(checkpoint_dir):
        print(f"\nИсходный чекпоинт существует: {checkpoint_dir}")
        state_file = os.path.join(checkpoint_dir, 'trainer_state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            print(f"Исходный шаг: {state.get('global_step', 'N/A')}")
            print(f"Исходная эпоха: {state.get('epoch', 'N/A')}")
            print(f"Исходный validation loss: {state.get('best_metric', 'N/A')}")
    
    # Проверка активных процессов Python
    print("\n=== Проверка активных процессов ===")
    try:
        import psutil
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'resume_training' in cmdline or 'train' in cmdline:
                        python_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if python_processes:
            print(f"Найдено {len(python_processes)} активных процессов Python:")
            for proc in python_processes:
                print(f"  PID {proc.info['pid']}: {proc.info['cmdline']}")
        else:
            print("Активных процессов обучения не найдено")
    except ImportError:
        print("Модуль psutil не установлен, проверка процессов невозможна")

if __name__ == "__main__":
    main()