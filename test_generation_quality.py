#!/usr/bin/env python3
"""
Тестирование качества генерации с метриками BLEU, ROUGE и человеческой оценкой.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import matplotlib.pyplot as plt

# Опциональные метрики
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

from config_unified import TEST_PROMPTS, GENERATION, MODELS

def generate_samples(model, tokenizer, prompts, max_length=100, device="cpu"):
    """Генерация текста по промптам"""
    model.eval()
    results = []
    
    gen_config = {
        "max_length": max_length,
        "temperature": GENERATION["temperature"],
        "top_p": GENERATION["top_p"],
        "top_k": GENERATION["top_k"],
        "repetition_penalty": GENERATION["repetition_penalty"],
        "do_sample": GENERATION["do_sample"],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                inputs.input_ids,
                **gen_config
            )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results.append({
                "prompt": prompt,
                "generated": generated,
                "full_text": generated
            })
    
    return results

def compute_bleu(references, hypotheses):
    """Вычисление BLEU"""
    if not BLEU_AVAILABLE:
        return None
    
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        
        # Минимальная длина для BLEU
        if len(hyp_tokens) < 4 or len(ref_tokens) < 4:
            continue
        
        score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        bleu_scores.append(score)
    
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

def compute_rouge(references, hypotheses):
    """Вычисление ROUGE"""
    if not ROUGE_AVAILABLE:
        return None
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        "rouge2": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        "rougeL": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    }

def compute_perplexity(model, tokenizer, eval_dataset, device="cpu", max_samples=50):
    """Вычисление perplexity"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    
    with torch.no_grad():
        for example in eval_dataset:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            attention_mask = torch.tensor(example.get("attention_mask", [1]*len(example["input_ids"]))).unsqueeze(0).to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()
            
            total_loss += loss * input_ids.size(1)
            total_tokens += input_ids.size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "samples": len(eval_dataset)
    }

def human_evaluation_prompt(generated_samples):
    """
    Генерирует шаблон для ручной оценки человеком.
    Выводит samples в удобном формате для оценки.
    """
    print("\n" + "="*80)
    print("РУЧНАЯ ОЦЕНКА КАЧЕСТВА ГЕНЕРАЦИИ")
    print("="*80)
    print("\nОцените каждый пример по шкале 1-5:")
    print("1 - Очень плохой (бессмыслица, повторения)")
    print("2 - Плохой (много ошибок, не связно)")
    print("3 - Удовлетворительный (есть смысл, но есть проблемы)")
    print("4 - Хороший (почти без ошибок, связно)")
    print("5 - Отличный (качественный текст в стиле Гоголя)\n")
    
    evaluations = []
    
    for i, sample in enumerate(generated_samples, 1):
        print(f"\n{'='*80}")
        print(f"ПРИМЕР {i}")
        print(f"{'='*80}")
        print(f"Промпт: {sample['prompt']}")
        print(f"\nГенерация:\n{sample['generated']}")
        print(f"\n{'-'*80}")
        
        if BLEU_AVAILABLE or ROUGE_AVAILABLE:
            print("(Автоматические метрики будут вычислены после оценки)")
        
        # Здесь можно сохранить для ручной оценки
        evaluations.append({
            "id": i,
            "prompt": sample["prompt"],
            "generated": sample["generated"],
            "score": None  # Будет заполнено вручную
        })
    
    print("\n" + "="*80)
    print("Сохраните результаты в human_evaluation.json")
    print("="*80)
    
    return evaluations

def save_results(results, output_path="evaluation_results.json"):
    """Сохранение результатов оценки"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты сохранены в {output_path}")

def plot_metrics(metrics_history, output_path="visualization/metrics.png"):
    """Визуализация метрик"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    if 'perplexity' in metrics_history:
        plt.subplot(2, 1, 1)
        plt.plot(range(len(metrics_history['perplexity'])), metrics_history['perplexity'], 'b-o')
        plt.xlabel('Эксперимент')
        plt.ylabel('Perplexity')
        plt.title('Динамика Perplexity')
        plt.grid(True)
    
    if 'bleu' in metrics_history:
        plt.subplot(2, 1, 2)
        plt.plot(range(len(metrics_history['bleu'])), metrics_history['bleu'], 'r-o')
        plt.xlabel('Эксперимент')
        plt.ylabel('BLEU')
        plt.title('Динамика BLEU')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"График сохранен в {output_path}")

def main():
    print("=== Тестирование качества генерации ===\n")
    
    # Загрузка модели
    model_path = "./gogol_finetuned_final"
    
    if not os.path.exists(model_path):
        print(f"Ошибка: модель не найдена в {model_path}")
        print("Сначала запустите обучение: python train_with_monitoring.py")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    
    print("Загрузка модели...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Генерация samples
    print("\nГенерация текста...")
    prompts = TEST_PROMPTS[:5]
    samples = generate_samples(model, tokenizer, prompts, device=device)
    
    # Вывод samples
    print("\n=== Сгенерированные тексты ===\n")
    for i, sample in enumerate(samples, 1):
        print(f"{i}. Промпт: {sample['prompt']}")
        print(f"   Генерация: {sample['generated'][:200]}...")
        print()
    
    # Вычисление метрик
    metrics = {}
    
    # Perplexity
    if os.path.exists("data/tokenized_gpt2"):
        print("\nВычисление Perplexity...")
        dataset = load_from_disk("data/tokenized_gpt2")
        if not hasattr(dataset, 'keys'):
            dataset = dataset.train_test_split(test_size=0.1, seed=42)
        eval_dataset = dataset["test"]
        
        ppl_result = compute_perplexity(model, tokenizer, eval_dataset, device)
        metrics.update(ppl_result)
        print(f"Perplexity: {metrics['perplexity']:.2f}")
    
    # BLEU и ROUGE (если доступны)
    if BLEU_AVAILABLE:
        # Для BLEU нужны reference тексты
        # Используем первые N абзацев из датасета как reference
        if os.path.exists("data/gogol_processed.txt"):
            with open("data/gogol_processed.txt", 'r', encoding='utf-8') as f:
                references = [line.strip() for line in f if line.strip()][:len(samples)]
            
            bleu = compute_bleu(references, [s['generated'] for s in samples])
            metrics['bleu'] = bleu
            print(f"BLEU: {bleu:.4f}")
    
    if ROUGE_AVAILABLE:
        if os.path.exists("data/gogol_processed.txt"):
            with open("data/gogol_processed.txt", 'r', encoding='utf-8') as f:
                references = [line.strip() for line in f if line.strip()][:len(samples)]
            
            rouge = compute_rouge(references, [s['generated'] for s in samples])
            metrics['rouge'] = rouge
            print(f"ROUGE-1: {rouge['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge['rougeL']:.4f}")
    
    # Сохранение результатов
    results = {
        "model_path": model_path,
        "device": device,
        "metrics": metrics,
        "samples": samples,
        "BLEU_available": BLEU_AVAILABLE,
        "ROUGE_available": ROUGE_AVAILABLE
    }
    
    save_results(results)
    
    # Ручная оценка
    human_evaluation_prompt(samples)
    
    print("\n=== Тестирование завершено ===")

if __name__ == "__main__":
    main()
