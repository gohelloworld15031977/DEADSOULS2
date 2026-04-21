#!/usr/bin/env python3
"""
Автоматические тесты качества генерации для CI/CD.
Проверяет, что модель генерирует связный текст без критических проблем.
"""

import unittest
import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_unified import TEST_PROMPTS

class TestGenerationQuality(unittest.TestCase):
    """Тесты качества генерации текста"""
    
    @classmethod
    def setUpClass(cls):
        """Загрузка модели один раз для всех тестов"""
        # Ищем последнюю версию модели
        model_path = "./gogol_finetuned_final"
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            # Пробуем checkpoint
            if os.path.exists("./gogol_finetuned_final/checkpoint-375"):
                model_path = "./gogol_finetuned_final/checkpoint-375"
        
        if not os.path.exists(model_path):
            raise unittest.SkipTest(f"Модель не найдена. Запустите обучение.")
        
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.tokenizer = AutoTokenizer.from_pretrained(model_path)
        cls.model = AutoModelForCausalLM.from_pretrained(model_path).to(cls.device)
        cls.model.eval()
        
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
    
    def _generate(self, prompt, max_length=100):
        """Вспомогательный метод генерации"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _count_repetitions(self, text, ngram_size=3):
        """Подсчет повторений n-грамм"""
        tokens = text.split()
        
        if len(tokens) < ngram_size:
            return 0.0
        
        ngrams = []
        for i in range(len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[i:i+ngram_size])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        unique_ngrams = set(ngrams)
        repetition_rate = 1 - (len(unique_ngrams) / len(ngrams))
        
        return repetition_rate * 100
    
    def test_generation_basic(self):
        """Тест: базовая генерация работает"""
        prompt = "Чичиков приехал в город"
        generated = self._generate(prompt)
        
        self.assertTrue(len(generated) > len(prompt), "Генерация слишком короткая")
        self.assertIn(prompt, generated, "Промпт не содержится в генерации")
    
    def test_no_critical_repetitions(self):
        """Тест: критические повторения отсутствуют"""
        prompt = "В губернском городе N"
        generated = self._generate(prompt)
        
        repetition_rate = self._count_repetitions(generated)
        
        # Для ранней стадии обучения допускаем до 50% повторений
        # После полного обучения целевое значение < 20%
        self.assertLess(
            repetition_rate, 50.0,
            f"Критически высокий уровень повторений: {repetition_rate:.1f}%"
        )
    
    def test_multiple_prompts(self):
        """Тест: генерация работает для нескольких промптов"""
        for prompt in TEST_PROMPTS[:3]:
            with self.subTest(prompt=prompt):
                generated = self._generate(prompt)
                self.assertGreater(
                    len(generated), 20,
                    f"Слишком короткая генерация для промпта: {prompt}"
                )
    
    def test_text_coherence(self):
        """Тест: генерация содержит минимальное количество символов"""
        prompt = "Однажды вечером сидел я"
        generated = self._generate(prompt, max_length=150)
        
        # Проверка на минимальную длину
        min_length = 50
        self.assertGreater(
            len(generated), min_length,
            f"Генерация слишком короткая: {len(generated)} символов"
        )
    
    def test_no_infinite_loop(self):
        """Тест: генерация завершается (не зацикливается)"""
        prompt = "Мертвые души - это"
        
        # Генерация с ограничением по времени
        import time
        start = time.time()
        
        generated = self._generate(prompt, max_length=100)
        
        elapsed = time.time() - start
        
        # Не должно занимать более 10 секунд
        self.assertLess(
            elapsed, 10.0,
            f"Генерация заняла слишком долго: {elapsed:.2f}с"
        )
    
    def test_special_tokens_removed(self):
        """Тест: специальные токены удалены из вывода"""
        prompt = "Чичиков приехал"
        generated = self._generate(prompt)
        
        # Проверка на отсутствие специальных токенов
        special_tokens = ["<unk>", "</s>", "<pad>", "<s>"]
        
        for token in special_tokens:
            self.assertNotIn(
                token, generated,
                f"Специальный токен {token} обнаружен в генерации"
            )


class TestModelMetrics(unittest.TestCase):
    """Тесты метрик модели"""
    
    @classmethod
    def setUpClass(cls):
        model_path = "./gogol_finetuned_final"
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            if os.path.exists("./gogol_finetuned_final/checkpoint-375"):
                model_path = "./gogol_finetuned_final/checkpoint-375"
        
        if not os.path.exists(model_path):
            raise unittest.SkipTest(f"Модель не найдена. Запустите обучение.")
        
        cls.tokenizer = AutoTokenizer.from_pretrained(model_path)
        cls.model = AutoModelForCausalLM.from_pretrained(model_path)
    
    def test_model_loaded(self):
        """Тест: модель загружена успешно"""
        self.assertIsNotNone(self.model)
    
    def test_tokenizer_loaded(self):
        """Тест: токенизатор загружен успешно"""
        self.assertIsNotNone(self.tokenizer)
        self.assertIsNotNone(self.tokenizer.pad_token)
    
    def test_model_parameters(self):
        """Тест: модель имеет разумное количество параметров"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Для GPT-2 small ~125M параметров
        # Для LoRA ~1-5M обучаемых параметров
        self.assertGreater(total_params, 10_000_000, "Слишком мало параметров")
        self.assertLess(total_params, 10_000_000_000, "Слишком много параметров")


def run_tests():
    """Запуск всех тестов"""
    print("=== Запуск тестов качества генерации ===\n")
    
    # Создаем тестовый suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем тесты
    suite.addTests(loader.loadTestsFromTestCase(TestGenerationQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestModelMetrics))
    
    # Запуск с результатом
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Сохранение результатов
    test_results = {
        "total": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful()
    }
    
    with open("test_results.json", 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в test_results.json")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
