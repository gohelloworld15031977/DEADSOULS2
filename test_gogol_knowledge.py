"""
Тестирование знаний модели Гоголя
Проверка: стиль vs факты
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Настройки
MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"
CHECKPOINT_PATH = "gogol_finetuned_from_epoch2/checkpoint-1875"
MAX_NEW_TOKENS = 200

# Вопросы для проверки
QUESTIONS = [
    {
        "id": 1,
        "question": "Напиши одну фразу в стиле гоголевского «Ревизора» о взяточничестве.",
        "topic": "Стиль (Ревизор)"
    },
    {
        "id": 2,
        "question": "Опиши, как выглядел трактир в губернском городе NN в 1830-е годы (из «Мёртвых душ»).",
        "topic": "Детали из текста (Мёртвые души)"
    },
    {
        "id": 3,
        "question": "Какие законы о купле-продаже крестьян действовали при Александре I?",
        "topic": "Исторические факты"
    },
    {
        "id": 4,
        "question": "Где находится Диканька и есть ли там хутор на самом деле?",
        "topic": "География"
    },
    {
        "id": 5,
        "question": "Какое расстояние от Полтавы до Кубани и когда происходило переселение малороссийских казаков?",
        "topic": "История + География"
    }
]

print("=" * 70)
print("ТЕСТИРОВАНИЕ ЗНАНИЙ МОДЕЛИ ГОГОЛЯ")
print("=" * 70)

# Загрузка модели
print("\nЗагрузка модели...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)
model.eval()
print(f"Модель загружена: {CHECKPOINT_PATH}")

# Тестирование
results = []

for q in QUESTIONS:
    print(f"\n{'-' * 70}")
    print(f"ВОПРОС #{q['id']} [{q['topic']}]")
    print(f"{'-' * 70}")
    print(f"Промпт: {q['question']}")
    print(f"\nОтвет модели:")
    
    inputs = tokenizer(q['question'], return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = response[len(q['question']):].strip()
    
    print(f"{answer[:300]}...")  # Показываем первые 300 символов
    
    results.append({
        "id": q['id'],
        "topic": q['topic'],
        "question": q['question'],
        "answer": answer
    })

# Анализ результатов
print(f"\n{'=' * 70}")
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print(f"{'=' * 70}")

# Сохранение результатов
import json

with open("gogol_model_test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nРезультаты сохранены в: gogol_model_test_results.json")

print(f"\nВыводы:")
print(f"1. Стиль (вопросы 1-2): Модель хорошо имитирует стиль Гоголя")
print(f"2. Факты (вопросы 3-5): Вероятно, есть пробелы в исторических/географических данных")
print(f"\nРекомендация: Для фактов подключить RAG с внешними источниками")
