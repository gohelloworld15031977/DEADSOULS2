# prepare_dataset.py
from datasets import Dataset
from transformers import AutoTokenizer
from config import MODEL_NAME, DATA_PATH, MAX_LENGTH
import json

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def read_texts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Разбиваем на абзацы (или предложения) – каждый станет отдельным примером
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    return paragraphs

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors=None
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

if __name__ == "__main__":
    try:
        texts = read_texts(DATA_PATH)
        print(f"Прочитано {len(texts)} абзацев")
        dataset = Dataset.from_list([{"text": t} for t in texts])
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized.save_to_disk("./data/tokenized_dataset")
        print(f"Сохранено {len(tokenized)} примеров в ./data/tokenized_dataset")
    except FileNotFoundError:
        print(f"Файл {DATA_PATH} не найден. Пожалуйста, поместите тексты Гоголя в этот файл.")
        print("Скачайте тексты с сайта ilibrary.ru или 'Академии' и сохраните как plain text.")
        print("Пример содержимого: 'Мёртвые души. Том первый...'")