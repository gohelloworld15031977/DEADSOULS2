"""
ReAct (Reason + Act) агент для модели Гоголя

Паттерн:
1. Reason: LLM анализирует запрос и планирует действия
2. Act: Агент вызывает инструменты (RAG, поиск, генерация)
3. Observe: Получает результаты
4. Повторяет до формирования ответа
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from search_module import InternetSearch

class ReActAgent:
    """ReAct агент с инструментами"""
    
    def __init__(self, checkpoint_path="gogol_finetuned_from_epoch2/checkpoint-1875"):
        # Загрузка модели
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ai-forever/rugpt3small_based_on_gpt2",
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "ai-forever/rugpt3small_based_on_gpt2",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        self.model.eval()
        
        # Инструменты
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.chroma_client = chromadb.PersistentClient(path="data/vector_db")
        self.collection = self.chroma_client.get_collection(name="historical_docs")
        self.search = InternetSearch()
        
        print("ReAct агент готов!")
    
    def tool_search_rag(self, query: str) -> str:
        """Инструмент: поиск в базе знаний"""
        query_embedding = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents"]
        )
        if not results['documents'] or not results['documents'][0]:
            return "Ничего не найдено"
        return "\n\n".join(results['documents'][0])
    
    def tool_search_internet(self, query: str) -> str:
        """Инструмент: поиск в интернете"""
        return self.search.search(query, max_results=3)
    
    def tool_generate(self, prompt: str) -> str:
        """Инструмент: генерация текстом"""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            output = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result[len(prompt):].strip()
    
    def reason(self, query: str, context: str = "") -> dict:
        """Reason: анализ запроса и решение о действиях"""
        
        # Простая эвристика для демонстрации
        if not context:
            context = "Нет контекста"
        
        # Определяем тип запроса
        if any(word in query.lower() for word in ["сейчас", "новости", "2024", "2025", "купить"]):
            action = "search_internet"
            action_input = query
        elif any(word in query.lower() for word in ["законы", "история", "1830", "1840", "Александр I"]):
            action = "search_rag"
            action_input = query
        else:
            action = "final_answer"
            action_input = query
        
        return {
            "thought": f"Нужно {'искать в интернете' if action == 'search_internet' else 'искать в базе' if action == 'search_rag' else 'ответить напрямую'}",
            "action": action,
            "action_input": action_input,
            "context": context
        }
    
    def run(self, query: str, max_steps: int = 3) -> str:
        """Запуск ReAct цикла"""
        
        print(f"\n{'=' * 70}")
        print(f"ЗАПРОС: {query}")
        print(f"{'=' * 70}")
        
        context = ""
        
        for step in range(max_steps):
            print(f"\n[Шаг {step + 1}]")
            
            # Reason: анализ
            decision = self.reason(query, context)
            print(f"\n[Мысль]: {decision['thought']}")
            print(f"[Действие]: {decision['action']}")
            
            # Act: выполнение действия
            if decision['action'] == "search_rag":
                print(f"[RAG запрос]: {decision['action_input']}")
                observation = self.tool_search_rag(decision['action_input'])
                context += f"\n[Результат RAG]: {observation}\n"
                print(f"[Результат]: {observation[:200]}...")
            
            elif decision['action'] == "search_internet":
                print(f"[Интернет запрос]: {decision['action_input']}")
                observation = self.tool_search_internet(decision['action_input'])
                context += f"\n[Результат поиска]: {observation}\n"
                print(f"[Результат]: {observation[:200]}...")
            
            elif decision['action'] == "final_answer":
                print(f"[Генерация ответа...]")
                
                if context and "Нет контекста" not in context:
                    full_prompt = f"""Используй информацию для ответа в стиле Гоголя:

ИНФОРМАЦИЯ:
{context}

ВОПРОС: {query}

ОТВЕТ:"""
                else:
                    full_prompt = query
                
                answer = self.tool_generate(full_prompt)
                print(f"{'=' * 70}")
                print(f"ОТВЕТ: {answer}")
                print(f"{'=' * 70}")
                return answer
        
        # Если достигли лимита шагов
        print("[Достигнут лимит шагов, генерирую ответ...]")
        return self.tool_generate(query)

def demo():
    """Демонстрация ReAct агента"""
    agent = ReActAgent()
    
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ ReAct АГЕНТА")
    print("=" * 70)
    
    test_cases = [
        "Какие законы о крепостном праве были при Александре I?",
        "Где можно купить книги Гоголя сейчас в 2024 году?",
        "Напиши фразу в стиле Гоголя о чиновниках"
    ]
    
    for i, query in enumerate(test_cases, 1):
        if i > 1:
            input("\nНажмите Enter для следующего примера...")
        
        agent.run(query)

if __name__ == "__main__":
    demo()
