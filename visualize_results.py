#!/usr/bin/env python3
"""
Визуализация результатов обучения и генерации.
Создает графики и отчеты для анализа качества модели.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_tensorboard_logs(log_dir):
    """Парсинг логов TensorBoard (упрощенный)"""
    # В реальном проекте используйте tensorboard.summary
    # Здесь простая заглушка
    return {
        "train_loss": [],
        "eval_loss": []
    }

def load_training_logs(log_file="logs/training.log"):
    """Загрузка логов обучения из файла"""
    losses = {"train": [], "eval": []}
    
    if not os.path.exists(log_file):
        return losses
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if "train_loss" in line:
                try:
                    loss = float(line.split("train_loss:")[1].split()[0])
                    losses["train"].append(loss)
                except:
                    pass
            elif "eval_loss" in line:
                try:
                    loss = float(line.split("eval_loss:")[1].split()[0])
                    losses["eval"].append(loss)
                except:
                    pass
    
    return losses

def plot_training_curves(losses, output_path="visualization/training_curves.png"):
    """Построение кривых обучения"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    if losses["train"]:
        plt.plot(losses["train"], label="Train Loss", marker='o')
    
    if losses["eval"]:
        plt.plot(losses["eval"], label="Eval Loss", marker='s')
    
    plt.xlabel("Шаг")
    plt.ylabel("Loss")
    plt.title("Кривые обучения")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Кривые обучения сохранены в {output_path}")

def plot_loss_histogram(losses, output_path="visualization/loss_histogram.png"):
    """Гистограмма распределения loss"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    if losses["train"]:
        plt.subplot(1, 2, 1)
        plt.hist(losses["train"], bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel("Train Loss")
        plt.ylabel("Частота")
        plt.title("Распределение Train Loss")
        plt.grid(True, alpha=0.3)
    
    if losses["eval"]:
        plt.subplot(1, 2, 2)
        plt.hist(losses["eval"], bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel("Eval Loss")
        plt.ylabel("Частота")
        plt.title("Распределение Eval Loss")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Гистограмма сохранена в {output_path}")

def plot_generation_samples(samples, output_path="visualization/generation_samples.txt"):
    """Сохранение примеров генерации для анализа"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== Примеры генерации текста ===\n\n")
        
        for i, sample in enumerate(samples, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"ПРИМЕР {i}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Промпт: {sample['prompt']}\n\n")
            f.write(f"Генерация:\n{sample['generated']}\n\n")
    
    print(f"Примеры генерации сохранены в {output_path}")

def plot_metrics_comparison(experiments, output_path="visualization/experiments_comparison.png"):
    """Сравнение нескольких экспериментов"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    for exp_name, metrics in experiments.items():
        if "eval_loss" in metrics:
            plt.plot(
                range(len(metrics["eval_loss"])),
                metrics["eval_loss"],
                marker='o',
                label=exp_name
            )
    
    plt.xlabel("Эпоха")
    plt.ylabel("Eval Loss")
    plt.title("Сравнение экспериментов")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Сравнение экспериментов сохранено в {output_path}")

def generate_report(stats, output_path="visualization/report.md"):
    """Генерация Markdown отчета"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = f"""# Отчет о качестве модели DeadSouls

## Общая информация
- **Дата создания**: {stats.get('date', 'N/A')}
- **Модель**: {stats.get('model_name', 'N/A')}
- **Датасет**: {stats.get('dataset_size', 0)} примеров

## Метрики обучения
- **Final Train Loss**: {stats.get('final_train_loss', 'N/A'):.4f}
- **Final Eval Loss**: {stats.get('final_eval_loss', 'N/A'):.4f}
- **Perplexity**: {stats.get('perplexity', 'N/A'):.2f}

## Метрики генерации
- **BLEU**: {stats.get('bleu', 'N/A')}
- **ROUGE-1**: {stats.get('rouge1', 'N/A')}
- **ROUGE-L**: {stats.get('rougeL', 'N/A')}

## Выводы
{stats.get('conclusions', 'Нет выводов')}

## Приложения
- Кривые обучения: `training_curves.png`
- Примеры генерации: `generation_samples.txt`
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Отчет сохранен в {output_path}")

def main():
    print("=== Визуализация результатов ===\n")
    
    # Создание директории
    os.makedirs("visualization", exist_ok=True)
    
    # Загрузка логов
    print("Загрузка логов...")
    losses = load_training_logs("logs/training.log")
    
    # Построение графиков
    if losses["train"] or losses["eval"]:
        plot_training_curves(losses)
        plot_loss_histogram(losses)
    else:
        print("Логов обучения не найдено. Пропуск графиков.")
    
    # Загрузка примеров генерации
    samples_path = "evaluation_results.json"
    if os.path.exists(samples_path):
        with open(samples_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "samples" in data:
            plot_generation_samples(data["samples"])
    
    # Генерация отчета
    stats = {
        "date": "2026-04-15",
        "model_name": "ai-forever/rugpt3small_based_on_gpt2",
        "dataset_size": 1664,
        "final_train_loss": losses["train"][-1] if losses["train"] else None,
        "final_eval_loss": losses["eval"][-1] if losses["eval"] else None,
        "conclusions": """
- Модель успешно обучена
- Loss стабилизировался
- Требуется дальнейшая оптимизация параметров генерации
        """.strip()
    }
    
    generate_report(stats)
    
    print("\nВизуализация завершена!")
    print("Откройте visualization/ для просмотра результатов.")

if __name__ == "__main__":
    main()
