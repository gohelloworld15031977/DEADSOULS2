#!/usr/bin/env python3
# Скрипт для скачивания дополнительных текстов Гоголя из интернета
import requests
from urllib.parse import quote
import os
import re
from bs4 import BeautifulSoup
import time

# Основные произведения Гоголя
GOGOL_WORKS = {
    # Повести и сборники
    "Вечера на хуторе близ Диканьки": [
        "Сорочинская ярмарка",
        "Вечер накануне Ивана Купала", 
        "Майская ночь, или Утопленница",
        "Пропавшая грамота",
        "Ночь перед Рождеством",
        "Страшная месть",
        "Иван Фёдорович Шпонька и его тётушка",
        "Заколдованное место"
    ],
    "Миргород": [
        "Старосветские помещики",
        "Тарас Бульба", 
        "Вий",
        "Повесть о том, как поссорился Иван Иванович с Иваном Никифоровичем"
    ],
    "Петербургские повести": [
        "Невский проспект",
        "Нос",
        "Портрет",
        "Шинель",
        "Записки сумасшедшего"
    ],
    # Драматургия
    "Ревизор": [],
    "Женитьба": [],
    # Поэма
    "Мёртвые души": ["Том 1", "Том 2 (фрагменты)"],
    # Другие произведения
    "Тарас Бульба": [],  # отдельное издание
    "Выбранные места из переписки с друзьями": [],
    "Авторская исповедь": []
}

def check_existing_texts():
    """Проверяет, какие тексты уже есть в проекте"""
    existing_works = set()
    
    # Проверяем основной файл
    if os.path.exists("data/gogol_books.txt"):
        with open("data/gogol_books.txt", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Ищем названия произведений в тексте
        for work in GOGOL_WORKS:
            if work.lower() in content.lower():
                existing_works.add(work)
                
    print(f"Найдено произведений в существующем файле: {len(existing_works)}")
    print("Существующие произведения:", ", ".join(existing_works) if existing_works else "нет")
    
    # Определяем недостающие произведения
    all_works = set(GOGOL_WORKS.keys())
    missing_works = all_works - existing_works
    
    print(f"\nНедостающие произведения: {len(missing_works)}")
    for work in missing_works:
        print(f"  - {work}")
    
    return existing_works, missing_works

def download_from_litmir(work_name):
    """Пытается скачать текст с Litmir"""
    try:
        # Кодируем название для URL
        encoded_name = quote(work_name)
        url = f"https://www.litmir.me/br/?b=218836&p=1"  # Пример ID для Гоголя
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Ищем текст произведения (зависит от структуры сайта)
            text_elements = soup.find_all('div', class_='text')
            
            if text_elements:
                text = '\n'.join([elem.get_text() for elem in text_elements])
                return text
                
    except Exception as e:
        print(f"Ошибка при скачивании {work_name}: {e}")
    
    return None

def download_from_ilibrary(work_name):
    """Пытается скачать текст с iLibrary"""
    try:
        # Используем публичные API или парсинг
        search_url = f"https://ilibrary.ru/text/{work_name.lower().replace(' ', '')}/index.html"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Поиск текста (зависит от структуры сайта)
            text_elements = soup.find_all('p')
            
            if text_elements:
                text = '\n'.join([elem.get_text() for elem in text_elements])
                return text
                
    except Exception as e:
        print(f"Ошибка при скачивании с iLibrary {work_name}: {e}")
    
    return None

def download_from_fallback(work_name):
    """Использует fallback - создает заглушку с поиском"""
    fallback_text = f"""
{work_name}
{'=' * len(work_name)}

[Текст произведения "{work_name}" будет загружен из открытых источников]

Для получения полного текста:
1. Посетите https://ilibrary.ru/ и найдите "{work_name}"
2. Или скачайте с https://www.litmir.me/bs/?b=218836
3. Или используйте проект "Все книги Гоголя" на GitHub

Это временная заглушка для структуры датасета.
"""
    return fallback_text

def main():
    print("=== Скачивание текстов Гоголя из интернета ===")
    
    # Проверяем существующие тексты
    existing, missing = check_existing_texts()
    
    if not missing:
        print("\nВсе основные произведения Гоголя уже есть в проекте!")
        return
    
    # Создаем папку для дополнительных текстов
    os.makedirs("data/gogol_additional", exist_ok=True)
    
    downloaded_count = 0
    
    # Пытаемся скачать недостающие произведения
    for work in missing:
        print(f"\nПопытка скачать: {work}")
        
        text = None
        
        # Пробуем разные источники
        text = download_from_litmir(work)
        
        if not text:
            text = download_from_ilibrary(work)
            
        if not text:
            text = download_from_fallback(work)
            print(f"  Использована заглушка для {work}")
        else:
            print(f"  Успешно скачано: {work}")
            downloaded_count += 1
        
        # Сохраняем текст
        filename = work.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
        filepath = f"data/gogol_additional/{filename}.txt"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Пауза между запросами
        time.sleep(1)
    
    # Объединяем все тексты в один файл
    print(f"\nОбъединение всех текстов...")
    combine_all_texts()
    
    print(f"\nГотово! Скачано {downloaded_count} новых произведений.")
    print(f"Все тексты сохранены в data/gogol_additional/")
    print(f"Объединенный файл: data/gogol_complete.txt")

def combine_all_texts():
    """Объединяет все тексты Гоголя в один файл"""
    all_texts = []
    
    # Добавляем существующий файл
    if os.path.exists("data/gogol_books.txt"):
        with open("data/gogol_books.txt", "r", encoding="utf-8") as f:
            all_texts.append(f.read())
    
    # Добавляем дополнительные тексты
    additional_dir = "data/gogol_additional"
    if os.path.exists(additional_dir):
        for filename in os.listdir(additional_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(additional_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    all_texts.append(f"\n\n{'='*80}\n\n")
                    all_texts.append(f.read())
    
    # Сохраняем объединенный файл
    if all_texts:
        with open("data/gogol_complete.txt", "w", encoding="utf-8") as f:
            f.write("".join(all_texts))
        
        print(f"Создан объединенный файл размером: {len(''.join(all_texts)):,} символов")

if __name__ == "__main__":
    main()