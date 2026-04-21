# download_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_NAME

print(f"Загрузка модели {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # устанавливаем pad токен
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("Готово. Модель сохранена в кэш Hugging Face.")
print(f"Размер модели: {model.num_parameters():,} параметров")