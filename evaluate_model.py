#!/usr/bin/env python3
"""
Оценка обученной модели на текстах Гоголя.
Загружает модель, вычисляет perplexity на валидационном датасете и генерирует примеры текста.
Добавлены метрики BLEU и ROUGE для количественной оценки качества генерации.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import math
import argparse
import sys

# Попытка импорта метрик BLEU и ROUGE (опционально)
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # type: ignore[import-untyped]
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    sentence_bleu = None  # type: ignore[assignment]
    SmoothingFunction = None  # type: ignore[assignment]
    print("Предупреждение: nltk не установлен. BLEU метрика недоступна. Установите: pip install nltk")

try:
    from rouge_score import rouge_scorer  # type: ignore[import-untyped]
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    rouge_scorer = None  # type: ignore[assignment]
    print("Предупреждение: rouge-score не установлен. ROUGE метрика недоступна. Установите: pip install rouge-score")

def evaluate_perplexity(model, tokenizer, eval_dataset, device="cpu", max_samples=None):
    """
    Вычисляет perplexity модели на датасете.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    if max_samples is not None:
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    
    print(f"Оценка perplexity на {len(eval_dataset)} примерах...")
    
    for i, example in enumerate(eval_dataset):
        if i % 10 == 0:
            print(f"  Обработано {i}/{len(eval_dataset)} примеров")
        
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()
        
        total_loss += loss * input_ids.size(1)
        total_tokens += input_ids.size(1)
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_samples": len(eval_dataset)
    }

def generate_samples(model, tokenizer, prompts, max_length=100, temperature=0.7, device="cpu"):
    """
    Генерирует текст по заданным промптам.
    """
    model.eval()
    results = []
    
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append((prompt, generated))
    
    return results

def compute_bleu_rouge(references, hypotheses):
    """
    Вычисляет метрики BLEU и ROUGE между списками референсных и сгенерированных текстов.
    Возвращает словарь с результатами.
    """
    metrics = {}
    
    # BLEU
    if BLEU_AVAILABLE and sentence_bleu is not None and SmoothingFunction is not None:  # type: ignore[truthy-function]
        smoothie = SmoothingFunction().method4
        bleu_scores = []
        for ref, hyp in zip(references, hypotheses):
            # Референс должен быть списком списков токенов
            ref_tokens = [ref.split()]
            hyp_tokens = hyp.split()
            score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)
            bleu_scores.append(score)
        metrics["bleu"] = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    else:
        metrics["bleu"] = None
    
    # ROUGE
    if ROUGE_AVAILABLE and rouge_scorer is not None:  # type: ignore[truthy-function]
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref, hyp)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        metrics["rouge1"] = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
        metrics["rouge2"] = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
        metrics["rougeL"] = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
    else:
        metrics["rouge1"] = None
        metrics["rouge2"] = None
        metrics["rougeL"] = None
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Оценка fine-tuned модели")
    parser.add_argument("--model_path", type=str, default="./gogol_finetuned_final",
                       help="Путь к обученной модели")
    parser.add_argument("--dataset_path", type=str, default="data/tokenized_gpt2",
                       help="Путь к токенизированному датасету")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Устройство для вычислений (cuda/cpu)")
    parser.add_argument("--max_eval_samples", type=int, default=50,
                       help="Максимальное количество примеров для оценки perplexity")
    parser.add_argument("--generate", action="store_true",
                       help="Сгенерировать примеры текста")
    parser.add_argument("--compute_metrics", action="store_true",
                       help="Вычислить BLEU и ROUGE метрики (требует установки nltk и rouge-score)")
    
    args = parser.parse_args()
    
    print("=== Оценка модели на текстах Гоголя ===")
    print(f"Модель: {args.model_path}")
    print(f"Устройство: {args.device}")
    
    # Загрузка модели и токенизатора
    print("Загрузка модели...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
        print(f"Модель загружена, параметры: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        print("Попытка загрузить оригинальную модель с адаптерами LoRA...")
        # Если модель сохранена как PEFT, может потребоваться специальная загрузка
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
        model = PeftModel.from_pretrained(base_model, args.model_path)
        model = model.merge_and_unload()  # type: ignore[assignment]
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")
        model = model.to(args.device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Загрузка датасета
    print(f"Загрузка датасета из {args.dataset_path}...")
    try:
        dataset = load_from_disk(args.dataset_path)
        
        from datasets import DatasetDict
        
        if isinstance(dataset, DatasetDict):
            # Датасет уже в формате DatasetDict
            if "test" in dataset:
                eval_dataset = dataset["test"]
            elif "validation" in dataset:
                eval_dataset = dataset["validation"]
            elif "train" in dataset:
                # Если есть только train, создаём test
                split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
                eval_dataset = split_dataset["test"]
            else:
                eval_dataset = dataset[list(dataset.keys())[0]]
        else:
            # Датасет — обычный Dataset, нужно разделить
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
            eval_dataset = split_dataset["test"]
        
        print(f"Валидационный датасет: {len(eval_dataset)} примеров")
    except Exception as e:
        print(f"Ошибка загрузки датасета: {e}")
        eval_dataset = None
    
    # Оценка perplexity
    if eval_dataset is not None:
        metrics = evaluate_perplexity(model, tokenizer, eval_dataset, 
                                     device=args.device, max_samples=args.max_eval_samples)
        print("\n=== Результаты оценки ===")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"Perplexity: {metrics['perplexity']:.2f}")
        print(f"Оценено токенов: {metrics['total_tokens']:,}")
        print(f"Оценено примеров: {metrics['num_samples']}")
    
    # Генерация примеров
    if args.generate:
        print("\n=== Генерация текста ===")
        prompts = [
            "Чичиков приехал в город",
            "В губернском городе N",
            "Однажды вечером сидел я",
            "Россия, куда ж несешься ты",
            "Мертвые души - это"
        ]
        
        samples = generate_samples(model, tokenizer, prompts, device=args.device)
        
        for prompt, generated in samples:
            print(f"\nПромпт: {prompt}")
            print(f"Сгенерировано: {generated}")
            print("-" * 80)
    
    print("\nОценка завершена.")

if __name__ == "__main__":
    main()