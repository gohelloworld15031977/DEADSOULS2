#!/usr/bin/env python3
"""
Расширенный датасет с текстами русских авторов XIX века для улучшения обобщения стиля.
Включает Гоголя, Пушкина, Тургенева, Достоевского, Толстого.
"""

import os
import re
import json
from pathlib import Path

# Базовые тексты Гоголя (уже есть в проекте)
GOGOL_TEXTS = [
    "data/gogol_books.txt",
    "data/gogol_processed.txt"
]

# Дополнительные авторы (нужно скачать или добавить вручную)
# Шаблоны для добавления новых авторов
AUTHOR_TEMPLATES = {
    "pushkin": "data/pushkin/pushkin_texts.txt",
    "turgenev": "data/turgenev/turgenev_texts.txt",
    "dostoevsky": "data/dostoevsky/dostoevsky_texts.txt",
    "tolstoy": "data/tolstoy/tolstoy_texts.txt"
}

def merge_datasets(output_path="data/combined_dataset.txt"):
    """Объединяет все доступные тексты в один датасет"""
    all_paragraphs = []
    source_stats = {}
    
    # Добавляем тексты Гоголя
    for path in GOGOL_TEXTS:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                paragraphs = split_into_paragraphs(text)
                all_paragraphs.extend(paragraphs)
                source_stats[path] = len(paragraphs)
                print(f"Добавлено {len(paragraphs)} абзацев из {path}")
    
    # Добавляем других авторов
    for author, path in AUTHOR_TEMPLATES.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                paragraphs = split_into_paragraphs(text)
                all_paragraphs.extend(paragraphs)
                source_stats[path] = len(paragraphs)
                print(f"Добавлено {len(paragraphs)} абзацев из {path}")
    
    # Сохраняем объединенный датасет
    with open(output_path, 'w', encoding='utf-8') as f:
        for para in all_paragraphs:
            f.write(f"{para}\n\n")
    
    # Сохраняем статистику
    stats_path = output_path.replace('.txt', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_paragraphs": len(all_paragraphs),
            "total_chars": sum(len(p) for p in all_paragraphs),
            "sources": source_stats
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Объединенный датасет ===")
    print(f"Всего абзацев: {len(all_paragraphs)}")
    print(f"Всего символов: {sum(len(p) for p in all_paragraphs):,}")
    print(f"Статистика сохранена в {stats_path}")
    
    return all_paragraphs

def split_into_paragraphs(text, min_length=50, max_length=1000):
    """Разделяет текст на абзацы"""
    paragraphs = []
    raw_paragraphs = text.split('\n\n')
    
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Очистка
        para = re.sub(r'\s+', ' ', para)
        para = re.sub(r'\n', ' ', para)
        
        if len(para) > max_length:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if current_length + len(sentence) > max_length and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= min_length:
                        paragraphs.append(chunk_text)
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
            
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= min_length:
                    paragraphs.append(chunk_text)
        elif len(para) >= min_length:
            paragraphs.append(para)
    
    return paragraphs

def create_download_scripts():
    """Создает скрипты для скачивания текстов"""
    
    # Скрипт для Пушкина
    pushkin_script = """#!/usr/bin/env python3
# Скачивание текстов Пушкина
import requests
import os

os.makedirs("data/pushkin", exist_ok=True)

# Пример источников (нужно проверить легальность использования)
sources = [
    "https://lib.rus.ec/book/index.php?book=123456",  # Заменить на реальные источники
]

print("Скачивание текстов Пушкина...")
# Реализовать скачивание из легальных источников
print("Пожалуйста, добавьте тексты Пушкина вручную из легальных источников")
"""
    
    with open("scripts/download_pushkin.py", 'w', encoding='utf-8') as f:
        f.write(pushkin_script)
    
    print("Скрипты для скачивания созданы в scripts/")

def main():
    print("=== Создание расширенного датасета ===")
    
    # Создаем директорию для скриптов
    os.makedirs("scripts", exist_ok=True)
    
    # Объединяем датасеты
    merge_datasets()
    
    # Создаем скрипты для скачивания
    create_download_scripts()
    
    print("\nДля добавления текстов других авторов:")
    print("1. Скачайте тексты из легальных источников (ФЭСТ, Викитека)")
    print("2. Положите в соответствующие директории:")
    print("   - data/pushkin/pushkin_texts.txt")
    print("   - data/turgenev/turgenev_texts.txt")
    print("   - data/dostoevsky/dostoevsky_texts.txt")
    print("   - data/tolstoy/tolstoy_texts.txt")
    print("3. Запустите этот скрипт снова")

if __name__ == "__main__":
    main()
