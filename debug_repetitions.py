#!/usr/bin/env python3
"""
Отладка и исправление проблемы с повторениями в генерации текста.
Тестирует различные параметры генерации для минимизации повторений.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import json

def detect_repetitions(text, ngram_size=3):
    """
    Вычисляет уровень повторений в тексте.
    Возвращает процент повторенных n-грамм.
    """
    tokens = text.split()
    
    if len(tokens) < ngram_size:
        return 0.0
    
    # Извлекаем n-граммы
    ngrams = []
    for i in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[i:i+ngram_size])
        ngrams.append(ngram)
    
    if not ngrams:
        return 0.0
    
    # Считаем уникальные и повторенные
    unique_ngrams = set(ngrams)
    repetition_rate = 1 - (len(unique_ngrams) / len(ngrams))
    
    return repetition_rate * 100

def generate_with_params(model, tokenizer, prompt, device, **gen_params):
    """Генерация с заданными параметрами"""
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    default_params = {
        "max_length": 100,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": 1.0
    }
    
    default_params.update(gen_params)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            **default_params
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    repetition_rate = detect_repetitions(generated)
    
    return {
        "generated": generated,
        "repetition_rate": repetition_rate,
        "params": gen_params
    }

def test_parameter_combinations(model, tokenizer, prompt, device):
    """
    Тестирует различные комбинации параметров для минимизации повторений.
    """
    print(f"\nТестирование параметров для промпта: '{prompt}'")
    print("="*80)
    
    results = []
    
    # Базовые параметры
    base_params = {
        "max_length": 100,
        "pad_token_id": tokenizer.pad_token_id
    }
    
    # 1. Разные температуры
    print("\n1. Тест температуры:")
    for temp in [0.3, 0.5, 0.7, 0.9, 1.2]:
        result = generate_with_params(
            model, tokenizer, prompt, device,
            temperature=temp,
            do_sample=True,
            **base_params
        )
        results.append({
            "test": f"temperature_{temp}",
            "params": {"temperature": temp},
            "repetition_rate": result["repetition_rate"],
            "generated": result["generated"][:200]
        })
        print(f"  Temp={temp}: Repetition={result['repetition_rate']:.1f}%")
    
    # 2. Разные repetition_penalty
    print("\n2. Тест repetition_penalty:")
    for penalty in [1.0, 1.1, 1.2, 1.5, 2.0]:
        result = generate_with_params(
            model, tokenizer, prompt, device,
            temperature=0.7,
            repetition_penalty=penalty,
            do_sample=True,
            **base_params
        )
        results.append({
            "test": f"repetition_penalty_{penalty}",
            "params": {"repetition_penalty": penalty},
            "repetition_rate": result["repetition_rate"],
            "generated": result["generated"][:200]
        })
        print(f"  Penalty={penalty}: Repetition={result['repetition_rate']:.1f}%")
    
    # 3. top_p (nucleus sampling)
    print("\n3. Тест top_p:")
    for top_p in [0.5, 0.75, 0.9, 0.95, 1.0]:
        result = generate_with_params(
            model, tokenizer, prompt, device,
            temperature=0.7,
            top_p=top_p,
            do_sample=True,
            **base_params
        )
        results.append({
            "test": f"top_p_{top_p}",
            "params": {"top_p": top_p},
            "repetition_rate": result["repetition_rate"],
            "generated": result["generated"][:200]
        })
        print(f"  Top_p={top_p}: Repetition={result['repetition_rate']:.1f}%")
    
    # 4. top_k
    print("\n4. Тест top_k:")
    for top_k in [10, 20, 40, 50, 100]:
        result = generate_with_params(
            model, tokenizer, prompt, device,
            temperature=0.7,
            top_k=top_k,
            do_sample=True,
            **base_params
        )
        results.append({
            "test": f"top_k_{top_k}",
            "params": {"top_k": top_k},
            "repetition_rate": result["repetition_rate"],
            "generated": result["generated"][:200]
        })
        print(f"  Top_k={top_k}: Repetition={result['repetition_rate']:.1f}%")
    
    # 5. Комбинации (лучшие параметры)
    print("\n5. Оптимальные комбинации:")
    optimal_configs = [
        {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.2},
        {"temperature": 0.8, "top_k": 40, "repetition_penalty": 1.3},
        {"temperature": 0.6, "top_p": 0.95, "repetition_penalty": 1.5},
    ]
    
    for i, config in enumerate(optimal_configs, 1):
        result = generate_with_params(
            model, tokenizer, prompt, device,
            do_sample=True,
            **base_params,
            **config
        )
        results.append({
            "test": f"optimal_{i}",
            "params": config,
            "repetition_rate": result["repetition_rate"],
            "generated": result["generated"][:200]
        })
        print(f"  Config {i}: Repetition={result['repetition_rate']:.1f}%")
    
    # Вывод лучших результатов
    print("\n" + "="*80)
    print("ЛУЧШИЕ РЕЗУЛЬТАТЫ (по минимальному repetition rate):")
    print("="*80)
    
    sorted_results = sorted(results, key=lambda x: x['repetition_rate'])
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. {result['test']}")
        print(f"   Params: {result['params']}")
        print(f"   Repetition: {result['repetition_rate']:.1f}%")
        print(f"   Text: {result['generated']}")
    
    # Сохранение результатов
    with open("repetition_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в repetition_test_results.json")
    
    return sorted_results[0] if sorted_results else None

def main():
    print("=== Отладка проблемы с повторениями ===\n")
    
    model_path = "./gogol_finetuned_final"
    
    if not os.path.exists(model_path):
        print(f"Ошибка: модель не найдена в {model_path}")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    
    print("Загрузка модели...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    model = model.to(device)  # type: ignore[arg-type]
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Тестовые промпты
    test_prompts = [
        "Чичиков приехал в город",
        "В губернском городе N"
    ]
    
    best_overall = None
    best_score = float('inf')
    
    for prompt in test_prompts:
        best_result = test_parameter_combinations(model, tokenizer, prompt, device)
        
        if best_result and best_result['repetition_rate'] < best_score:
            best_score = best_result['repetition_rate']
            best_overall = best_result
    
    # Итоговый вывод
    print("\n" + "="*80)
    print("ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print("="*80)
    
    if best_overall:
        print(f"\nОптимальные параметры:")
        print(f"  {best_overall['params']}")
        print(f"\nДостижимый уровень повторений: {best_overall['repetition_rate']:.1f}%")
        
        # Сохранение рекомендаций
        recommendations = {
            "optimal_params": best_overall['params'],
            "expected_repetition_rate": best_overall['repetition_rate'],
            "notes": [
                "Используйте temperature 0.7-0.8 для баланса качества и разнообразия",
                "repetition_penalty > 1.1 помогает уменьшить повторения",
                "top_p 0.9-0.95 или top_k 40-50 для nucleus sampling",
                "Для дальнейшего уменьшения повторений увеличьте датасет"
            ]
        }
        
        with open("generation_recommendations.json", 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, ensure_ascii=False, indent=2)
        
        print(f"\nРекомендации сохранены в generation_recommendations.json")

if __name__ == "__main__":
    main()
