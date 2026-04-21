#!/usr/bin/env python3
# Запуск обучения с логированием вывода
import subprocess
import sys
import time

def main():
    log_file = "training_log.txt"
    print(f"Запуск обучения с логированием в {log_file}...")
    
    # Команда для запуска
    cmd = [sys.executable, "resume_training_fixed.py"]
    
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            # Записываем время начала
            f.write(f"=== Начало обучения: {time.ctime()} ===\n")
            f.flush()
            
            # Запускаем процесс
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
                universal_newlines=True
            )
            
            # Читаем вывод в реальном времени
            print("Обучение запущено. Вывод будет записан в файл и выведен здесь.")
            if process.stdout:
                for line in process.stdout:
                    print(line, end="")
                    f.write(line)
                    f.flush()
            
            # Ждем завершения
            process.wait()
            
            # Записываем время окончания и код возврата
            f.write(f"\n=== Обучение завершено: {time.ctime()} ===\n")
            f.write(f"Код возврата: {process.returncode}\n")
            
            print(f"\nОбучение завершено с кодом {process.returncode}")
            print(f"Полный лог сохранен в {log_file}")
            
            if process.returncode != 0:
                print("Обучение завершилось с ошибкой!")
                # Показываем последние 20 строк лога
                with open(log_file, "r", encoding="utf-8") as log:
                    lines = log.readlines()
                    print("\n=== Последние 20 строк лога ===")
                    for line in lines[-20:]:
                        print(line, end="")
            else:
                print("Обучение завершилось успешно!")
                
    except Exception as e:
        print(f"Ошибка при запуске обучения: {e}")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Ошибка: {e}\n")

if __name__ == "__main__":
    main()