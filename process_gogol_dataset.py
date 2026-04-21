#!/usr/bin/env python3
# Обработка и очистка текстов Гоголя для создания датасета
import re
import os
from pathlib import Path

def clean_text(text):
    """Очистка текста от лишних символов и форматирования"""
    # Удаляем BOM символ если есть
    text = text.replace('\ufeff', '')
    
    # Заменяем множественные пробелы на один
    text = re.sub(r'\s+', ' ', text)
    
    # Удаляем лишние переносы строк внутри абзацев
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # Сохраняем абзацы (двойные переносы строк)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Удаляем специальные символы, но сохраняем пунктуацию
    text = re.sub(r'[^\w\s.,!?;:()\"\'-]', '', text)
    
    # Восстанавливаем кавычки
    text = text.replace('``', '"').replace("''", '"')
    
    return text.strip()

def split_into_paragraphs(text, min_length=50, max_length=1000):
    """Разделяет текст на абзацы подходящего размера"""
    paragraphs = []
    
    # Разделяем по двойным переносам строк
    raw_paragraphs = text.split('\n\n')
    
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # Если абзац слишком длинный, разбиваем на предложения
        if len(para) > max_length:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_length = len(sentence)
                
                if current_length + sentence_length > max_length and current_chunk:
                    # Сохраняем текущий чанк
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= min_length:
                        paragraphs.append(chunk_text)
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            # Добавляем последний чанк
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= min_length:
                    paragraphs.append(chunk_text)
        elif len(para) >= min_length:
            paragraphs.append(para)
    
    return paragraphs

def analyze_dataset(text):
    """Анализирует датасет и выводит статистику"""
    paragraphs = split_into_paragraphs(text)
    
    total_chars = len(text)
    total_paragraphs = len(paragraphs)
    
    # Статистика по длине абзацев
    lengths = [len(p) for p in paragraphs]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    
    # Подсчет слов
    words = re.findall(r'\b\w+\b', text)
    total_words = len(words)
    
    print("=== Анализ датасета текстов Гоголя ===")
    print(f"Общий размер: {total_chars:,} символов")
    print(f"Количество слов: {total_words:,}")
    print(f"Количество абзацев: {total_paragraphs:,}")
    print(f"Средняя длина абзаца: {avg_length:.0f} символов")
    print(f"Минимальная длина: {min_length} символов")
    print(f"Максимальная длина: {max_length} символов")
    
    # Распределение по длине
    print("\nРаспределение абзацев по длине:")
    bins = [0, 100, 200, 300, 500, 1000, float('inf')]
    bin_labels = ["<100", "100-200", "200-300", "300-500", "500-1000", ">1000"]
    
    for i in range(len(bins)-1):
        count = sum(1 for l in lengths if bins[i] <= l < bins[i+1])
        percentage = (count / total_paragraphs * 100) if total_paragraphs > 0 else 0
        print(f"  {bin_labels[i]}: {count} абзацев ({percentage:.1f}%)")
    
    return paragraphs

def create_processed_dataset(paragraphs, output_path):
    """Создает обработанный датасет"""
    # Сохраняем как один файл с абзацами
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, para in enumerate(paragraphs, 1):
            f.write(f"{para}\n\n")
    
    # Также сохраняем как JSONL для удобства
    import json
    jsonl_path = output_path.replace('.txt', '.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for para in paragraphs:
            record = {"text": para}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"\nОбработанный датасет сохранен:")
    print(f"  - {output_path}")
    print(f"  - {jsonl_path}")

def main():
    # Пути к файлам
    input_file = "data/gogol_complete.txt"
    output_file = "data/gogol_processed.txt"
    
    if not os.path.exists(input_file):
        print(f"Ошибка: файл {input_file} не найден")
        return
    
    print("Загрузка текстов Гоголя...")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print("Очистка текста...")
    cleaned_text = clean_text(text)
    
    print("Анализ датасета...")
    paragraphs = analyze_dataset(cleaned_text)
    
    print("\nСоздание обработанного датасета...")
    create_processed_dataset(paragraphs, output_file)
    
    # Создаем также уменьшенную версию для тестов
    if len(paragraphs) > 1000:
        test_paragraphs = paragraphs[:1000]
        test_output = "data/gogol_test.txt"
        create_processed_dataset(test_paragraphs, test_output)
        print(f"\nТестовый датасет (1000 абзацев) сохранен в {test_output}")
    
    # Обновляем основной файл датасета для обучения
    print("\nОбновление основного файла датасета...")
    with open("data/gogol_books.txt", 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f"\nОсновной файл датасета обновлен: data/gogol_books.txt")
    print(f"Исходный размер: {len(text):,} символов")
    print(f"Очищенный размер: {len(cleaned_text):,} символов")
    print(f"Количество абзацев для обучения: {len(paragraphs):,}")

if __name__ == "__main__":
    main()