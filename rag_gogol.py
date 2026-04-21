"""
Модель Гоголя с RAG и поиском в интернете
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from search_module import InternetSearch

class GogolRAG:
    """Модель Гоголя с RAG и поиском"""
    
    def __init__(self, checkpoint_path="gogol_finetuned_from_epoch2/checkpoint-1875"):
        self.model_name = "ai-forever/rugpt3small_based_on_gpt2"
        self.load_model(checkpoint_path)
        self.load_rag_modules()
    
    def load_model(self, checkpoint_path):
        """Загрузка модели"""
        print("Загрузка модели Гоголя...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        self.model.eval()
        print("Модель загружена!")
    
    def load_rag_modules(self):
        """Загрузка модулей RAG"""
        print("\nЗагрузка RAG модулей...")
        
        # Эмбеддинги
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Векторная база
        self.chroma_client = chromadb.PersistentClient(path="data/vector_db")
        self.collection = self.chroma_client.get_collection(name="historical_docs")
        print(f"Векторная база: {self.collection.count()} чанков")
        
        # Поиск
        self.search = InternetSearch()
        
        print("RAG модули загружены!")
    
    def retrieve_from_db(self, query, top_k=3):
        """Поиск в локальной базе"""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        if not results['documents'] or not results['documents'][0]:
            return ""
        
        context = "\n\n".join(results['documents'][0])
        return context
    
    def should_search_internet(self, query):
        """Решение: искать в интернете или нет"""
        # Ключевые слова для интернета
        internet_keywords = [
            "сейчас", "сегодня", "новости", "актуально",
            "2024", "2025", "2026", "современный", "последний",
            "купить", "цена", "где найти"
        ]
        
        # Ключевые слова для локальной базы
        local_keywords = [
            "Александр I", "1830", "1840", "1801", "1810", "1820",
            "законы", "крепостное", "Диканька", "Полтава", "история",
            "XIX век", "XVIII век", "казаки", "империя"
        ]
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in internet_keywords):
            return "internet"
        elif any(word in query_lower for word in local_keywords):
            return "local"
        else:
            return "model_only"
    
    def generate_with_rag(self, prompt, use_rag=True):
        """Генерация с RAG"""
        
        # Решение о поиске
        search_type = self.should_search_internet(prompt)
        
        context = ""
        source_info = ""
        
        if use_rag:
            if search_type == "local":
                source_info = "[Поиск в исторической базе]"
                context = self.retrieve_from_db(prompt)
            elif search_type == "internet":
                source_info = "[Поиск в интернете]"
                context = self.search.search(prompt)
        
        # Формирование промпта
        if context:
            full_prompt = f"""Используй следующую информацию для ответа в стиле Гоголя:

ИНФОРМАЦИЯ:
{context}

ВОПРОС: {prompt}

ОТВЕТ:"""
        else:
            full_prompt = prompt
        
        # Генерация
        inputs = self.tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            output = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )
        
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = result[len(full_prompt):].strip()
        
        return answer, source_info
    
    def chat(self):
        """Интерактивный чат"""
        print("=" * 70)
        print("ГОГОЛЬ-RAG ЧАТ")
        print("Модель с доступом к исторической базе и интернету")
        print("Введите 'exit' для выхода")
        print("=" * 70)
        
        while True:
            try:
                user_input = input("\nВы: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'выход', 'q']:
                    print("До свидания!")
                    break
                
                if not user_input:
                    continue
                
                print("\nГоголь:", end=" ", flush=True)
                response, source = self.generate_with_rag(user_input)
                
                if source:
                    print(f"\n{source}")
                print(response)
            
            except KeyboardInterrupt:
                print("\nДо свидания!")
                break
            except Exception as e:
                print(f"\nОшибка: {e}")

def test_rag():
    """Тестирование RAG"""
    rag = GogolRAG()
    
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ RAG")
    print("=" * 70)
    
    test_questions = [
        "Какие законы о крепостном праве действовали при Александре I?",
        "Где находится Диканька?",
        "Как выглядел трактир в губернском городе в 1830-е?",
        "Напиши фразу в стиле Гоголя о чиновниках"
    ]
    
    for q in test_questions:
        print(f"\n{'-' * 70}")
        print(f"Вопрос: {q}")
        answer, source = rag.generate_with_rag(q)
        print(f"Источник: {source if source else 'Только модель'}")
        print(f"Ответ: {answer[:300]}...")

if __name__ == "__main__":
    import sys
    
    # Тест или чат
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_rag()
    else:
        rag = GogolRAG()
        rag.chat()
