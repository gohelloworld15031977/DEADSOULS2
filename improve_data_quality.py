#!/usr/bin/env python3
"""
Улучшение качества данных: очистка, нормализация и дедупликация.
"""

import re
import os
from collections import Counter

def clean_text(text):
    """
    Расширенная очистка текста.
    """
    # Удаляем BOM и нежелательные символы
    text = text.replace('\ufeff', '')
    
    # Нормализуем пробелы
    text = re.sub(r'\s+', ' ', text)
    
    # Удаляем лишние переносы строк
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Восстанавливаем кавычки
    text = text.replace('``', '"').replace("''", '"')
    
    # Удаляем ссылки и сноски
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    
    # Удаляем строки только из цифр (номера страниц)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Убираем повторяющиеся знаки препинания
    text = re.sub(r'[.,!]{3,}', lambda m: m.group(0)[0] * 2, text)
    
    return text.strip()

def deduplicate_paragraphs(paragraphs, similarity_threshold=0.95):
    """
    Удаляет почти идентичные абзацы.
    
    Args:
        paragraphs: список абзацев
        similarity_threshold: порог сходства (1.0 = идентичные)
    
    Returns:
        Список уникальных абзацев
    """
    unique = []
    seen = Counter()
    
    for para in paragraphs:
        # Нормализация для сравнения
        normalized = para.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Точное совпадение
        if normalized in seen:
            continue
        
        # Проверка на почти идентичные
        is_duplicate = False
        for existing in unique[-100:]:  # Проверяем последние 100
            existing_norm = existing.lower().strip()
            existing_norm = re.sub(r'\s+', ' ', existing_norm)
            
            # Простое сходство по длине и содержанию
            if abs(len(normalized) - len(existing_norm)) < 5:
                # Подсчет общих слов
                common_words = set(normalized.split()) & set(existing_norm.split())
                total_words = len(set(normalized.split()) | set(existing_norm.split()))
                
                if total_words > 0:
                    similarity = len(common_words) / total_words
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            unique.append(para)
            seen[normalized] = seen.get(normalized, 0) + 1
    
    return unique

def filter_quality(paragraphs, min_length=50, max_length=2000):
    """
    Фильтрует абзацы по качеству.
    """
    filtered = []
    
    for para in paragraphs:
        # Длина
        if len(para) < min_length or len(para) > max_length:
            continue
        
        # Проверяем на читаемость
        words = para.split()
        if len(words) < 10:
            continue
        
        # Проверяем на слишком много цифр
        digit_count = sum(c.isdigit() for c in para)
        if digit_count > len(para) * 0.1:  # >10% цифр
            continue
        
        # Проверяем на слишком много специальных символов
        special_count = sum(1 for c in para if not c.isalnum() and not c.isspace())
        if special_count > len(para) * 0.3:  # >30% спецсимволов
            continue
        
        filtered.append(para)
    
    return filtered

def analyze_quality(paragraphs):
    """
    Анализ качества датасета.
    """
    print("\n=== Анализ качества датасета ===\n")
    
    # Статистика длины
    lengths = [len(p) for p in paragraphs]
    print(f"Количество абзацев: {len(paragraphs)}")
    print(f"Минимальная длина: {min(lengths)}")
    print(f"Максимальная длина: {max(lengths)}")
    print(f"Средняя длина: {sum(lengths)/len(lengths):.1f}")
    
    # Распределение
    print("\nРаспределение по длине:")
    bins = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, float('inf'))]
    labels = ["0-100", "100-200", "200-500", "500-1000", ">1000"]
    
    for (low, high), label in zip(bins, labels):
        count = sum(1 for l in lengths if low <= l < high)
        print(f"  {label}: {count} ({count/len(lengths)*100:.1f}%)")
    
    # Анализ словаря
    all_words = []
    for para in paragraphs:
        words = re.findall(r'\b\w+\b', para.lower())
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    print(f"\nУникальных слов: {len(word_counts)}")
    print(f"Всего слов: {len(all_words)}")
    print(f"Средняя длина слова: {sum(len(w) for w in all_words)/len(all_words):.1f}")
    
    # Топ слов
    print("\nТоп 20 слов:")
    for word, count in word_counts.most_common(20):
        print(f"  {word}: {count}")
    
    return {
        "total_paragraphs": len(paragraphs),
        "unique_words": len(word_counts),
        "total_words": len(all_words),
        "avg_length": sum(lengths)/len(lengths)
    }

def process_dataset(input_path, output_path):
    """
    Полная обработка датасета.
    """
    print(f"=== Обработка датасета ===")
    print(f"Входной файл: {input_path}")
    print(f"Выходной файл: {output_path}")
    
    # Загрузка
    print("\nЗагрузка данных...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Разделение на абзацы
    paragraphs = text.split('\n\n')
    print(f"Найдено абзацев: {len(paragraphs)}")
    
    # Очистка
    print("\nОчистка текста...")
    cleaned = [clean_text(p) for p in paragraphs]
    cleaned = [p for p in cleaned if p]  # Удаление пустых
    print(f"После очистки: {len(cleaned)}")
    
    # Фильтрация качества
    print("\nФильтрация по качеству...")
    filtered = filter_quality(cleaned)
    print(f"После фильтрации: {len(filtered)}")
    
    # Дедупликация
    print("\nДедупликация...")
    deduplicated = deduplicate_paragraphs(filtered)
    print(f"После дедупликации: {len(deduplicated)}")
    
    # Сохранение
    print("\nСохранение...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for para in deduplicated:
            f.write(f"{para}\n\n")
    
    # Анализ
    analyze_quality(deduplicated)
    
    print(f"\nОбработанный датасет сохранен в {output_path}")
    
    return deduplicated

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Улучшение качества данных")
    parser.add_argument("--input", default="data/gogol_processed.txt",
                       help="Входной файл")
    parser.add_argument("--output", default="data/gogol_cleaned.txt",
                       help="Выходной файл")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Ошибка: файл {args.input} не найден")
        return
    
    process_dataset(args.input, args.output)

if __name__ == "__main__":
    main()
