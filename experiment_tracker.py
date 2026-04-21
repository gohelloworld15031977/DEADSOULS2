#!/usr/bin/env python3
"""
Система экспериментов с MLflow для версионирования и отслеживания результатов.
"""

import os
import json
import git
import uuid
from datetime import datetime

try:
    import mlflow  # type: ignore[import-untyped]
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore[assignment]
    print("Предупреждение: mlflow не установлен. Установите: pip install mlflow")

class ExperimentTracker:
    """Трекер экспериментов с MLflow"""
    
    def __init__(self, experiment_name="deadsouls-gogol"):
        self.experiment_name = experiment_name
        self.run_id = None
        self.mlflow_available = MLFLOW_AVAILABLE
        
        if self.mlflow_available and mlflow is not None:  # type: ignore[truthy-function]
            # Установка эксперимента
            mlflow.set_experiment(experiment_name)
    
    def get_git_info(self):
        """Получение информации о Git commitе"""
        try:
            repo = git.Repo(search_parent_directories=True)
            return {
                "commit": repo.head.object.hexsha,
                "branch": repo.active_branch.name,
                "dirty": repo.is_dirty()
            }
        except Exception:
            return {
                "commit": None,
                "branch": None,
                "dirty": False
            }
    
    def start_run(self, name=None, params=None, tags=None):
        """Запуск нового эксперимента"""
        if not self.mlflow_available or mlflow is None:  # type: ignore[truthy-function]
            print("MLflow не доступен. Эксперименты не будут отслеживаться.")
            return None
        
        run_name = name or f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Начинаем run
        run = mlflow.start_run(run_name=run_name)
        self.run_id = run.info.run_id
        
        # Добавляем теги
        if tags:
            mlflow.set_tags(tags)
        
        # Добавляем git info
        git_info = self.get_git_info()
        if git_info["commit"]:
            mlflow.log_param("git_commit", git_info["commit"])
        if git_info["branch"]:
            mlflow.set_tag("git_branch", git_info["branch"])
        
        # Добавляем пользовательские параметры
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        
        print(f"Эксперимент запущен: {run_name}")
        print(f"Run ID: {self.run_id}")
        
        return run
    
    def log_metrics(self, metrics):
        """Логирование метрик"""
        if not self.mlflow_available or mlflow is None or not self.run_id:  # type: ignore[truthy-function]
            return
        
        for step, metric_dict in enumerate(metrics):
            mlflow.log_metrics(metric_dict, step=step)
    
    def log_metric(self, name, value, step=None):
        """Логирование отдельной метрики"""
        if not self.mlflow_available or mlflow is None or not self.run_id:  # type: ignore[truthy-function]
            return
        
        mlflow.log_metric(name, value, step=step)
    
    def log_params(self, params):
        """Логирование параметров"""
        if not self.mlflow_available or mlflow is None:  # type: ignore[truthy-function]
            return
        
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_artifact(self, artifact_path):
        """Логирование артефакта (модель, конфиг и т.д.)"""
        if not self.mlflow_available or mlflow is None or not self.run_id:  # type: ignore[truthy-function]
            return
        
        mlflow.log_artifact(artifact_path)
    
    def end_run(self):
        """Завершение эксперимента"""
        if not self.mlflow_available or mlflow is None:  # type: ignore[truthy-function]
            return
        
        mlflow.end_run()
        self.run_id = None
    
    def save_run_info(self, output_path="experiment_run.json"):
        """Сохранение информации о запуске в файл"""
        run_info = {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "git_info": self.get_git_info()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2)
        
        print(f"Информация о запуске сохранена в {output_path}")

def log_training_experiment(
    model_name,
    dataset_size,
    training_params,
    metrics_history,
    final_model_path
):
    """
    Полное логирование эксперимента обучения.
    
    Пример использования:
    tracker = ExperimentTracker()
    tracker.start_run(
        name="gogol_gpt2_small_v1",
        params={
            "model": "ai-forever/rugpt3small_based_on_gpt2",
            "dataset_size": 1664,
            "epochs": 7,
            "lr": 1e-4
        }
    )
    
    # Во время обучения
    tracker.log_metric("eval_loss", 3.8387, step=375)
    tracker.log_metric("train_loss", 3.982, step=375)
    
    # После обучения
    tracker.log_artifact("./gogol_finetuned_final")
    tracker.end_run()
    """
    
    if not MLFLOW_AVAILABLE:
        print("MLflow не установлен. Используйте локальное логирование.")
        
        # Локальное сохранение
        experiment_data = {
            "model_name": model_name,
            "dataset_size": dataset_size,
            "training_params": training_params,
            "metrics_history": metrics_history,
            "final_model_path": final_model_path,
            "timestamp": datetime.now().isoformat()
        }
        
        os.makedirs("experiments", exist_ok=True)
        run_id = str(uuid.uuid4())[:8]
        output_path = f"experiments/experiment_{run_id}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_data, f, ensure_ascii=False, indent=2)
        
        print(f"Эксперимент сохранен локально: {output_path}")
        return
    
    # MLflow логирование
    tracker = ExperimentTracker()
    
    if mlflow is None:  # type: ignore[truthy-function]
        print("MLflow не доступен")
        return
    
    # Запуск
    tracker.start_run(
        params={
            "model_name": model_name,
            "dataset_size": dataset_size,
            **training_params
        },
        tags={
            "project": "deadsouls",
            "task": "style_transfer"
        }
    )
    
    # Логирование метрик
    for step, metrics in enumerate(metrics_history):
        tracker.log_metrics(metrics)
    
    # Логирование модели
    if os.path.exists(final_model_path):
        tracker.log_artifact(final_model_path)
    
    # Завершение
    tracker.end_run()
    tracker.save_run_info()
    
    print(f"Эксперимент завершен и сохранен в MLflow")

def compare_experiments():
    """Сравнение предыдущих экспериментов"""
    if not MLFLOW_AVAILABLE or mlflow is None:  # type: ignore[truthy-function]
        print("MLflow не установлен")
        return
    
    experiment_name = "deadsouls-gogol"
    
    # Получение всех запусков
    client = mlflow.tracking.MlflowClient()  # type: ignore[attr-defined]
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Эксперимент {experiment_name} не найден")
        return
    
    runs = client.search_runs(experiment.experiment_id)
    
    print(f"\n=== Сравнение экспериментов: {experiment_name} ===\n")
    
    for run in runs[:10]:  # Последние 10 запусков
        print(f"Run: {run.info.run_name}")
        print(f"  Commit: {run.data.params.get('git_commit', 'N/A')[:7]}")
        print(f"  Metrics:")
        for key, value in run.data.metrics.items():
            print(f"    {key}: {value:.4f}")
        print()

def main():
    """Демо использование"""
    print("=== Система экспериментов DeadSouls ===\n")
    
    # Пример запуска эксперимента
    tracker = ExperimentTracker("deadsouls-demo")
    
    tracker.start_run(
        name="demo_run_1",
        params={
            "model": "ai-forever/rugpt3small_based_on_gpt2",
            "lr": 1e-4,
            "epochs": 7
        },
        tags={"environment": "test"}
    )
    
    # Симуляция метрик
    metrics = [
        {"eval_loss": 4.5, "train_loss": 4.2},
        {"eval_loss": 4.0, "train_loss": 3.8},
        {"eval_loss": 3.8, "train_loss": 3.5}
    ]
    
    for i, metric in enumerate(metrics):
        tracker.log_metric("eval_loss", metric["eval_loss"], step=i)
        tracker.log_metric("train_loss", metric["train_loss"], step=i)
        print(f"Step {i}: eval_loss={metric['eval_loss']:.4f}, train_loss={metric['train_loss']:.4f}")
    
    tracker.end_run()
    tracker.save_run_info()
    
    print("\nДемо завершено!")
    print("\nДля запуска реального эксперимента используйте:")
    print("  from experiment_tracker import log_training_experiment")
    print("  log_training_experiment(model_name, dataset_size, params, metrics, model_path)")

if __name__ == "__main__":
    main()
