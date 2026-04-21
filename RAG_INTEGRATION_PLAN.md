 # План интеграции RAG и поиска в интернет к модели Гоголя

## 📋 Обзор

**Цель:** Добавить к существующей дообученной модели GPT-2 возможность:
1. Обращаться к локальной векторной базе (исторические документы эпохи Александра I)
2. Искать информацию в интернете через API
3. Автоматически выбирать источник знаний в зависимости от вопроса

---

## 🛠️ Часть 1: Необходимые библиотеки

```bash
# Установка зависимостей
pip install chromadb  # или: pip install faiss-cpu
pip install langchain
pip install duckduckgo-search
pip install requests
pip install sentence-transformers
pip install accelerate
```

**requirements.txt:**
```txt
transformers>=4.37.0
peft>=0.8.0
torch>=2.1.0
chromadb>=0.4.0
langchain>=0.1.0
duckduckgo-search>=4.0
sentence-transformers>=2.2.0
accelerate>=0.26.0
```

---

## 📚 Часть 2: Подготовка исторических документов

### Шаг 1: Структура папок

```
data/
├── gogol_texts/           # Тексты Гоголя (уже есть)
├── historical_docs/       # Исторические документы
│   ├── laws_alexander1.txt
│   ├── geography_1830s.txt
│   ├── customs_19th_century.txt
│   └── legal_system.txt
└── vector_db/            # Векторная база (создастся автоматически)
```

### Шаг 2: Индексация документов

```python
# rag_index.py
import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader

def index_documents(docs_dir, db_path="data/vector_db"):
    """Индексация исторических документов"""
    
    # Загрузка эмбеддингов (русский язык)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Создание текстового сплиттера
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", ".", " "]
    )
    
    # Инициализация ChromaDB
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.create_collection(name="historical_docs")
    
    all_documents = []
    
    # Чтение всех файлов
    for filename in os.listdir(docs_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(docs_dir, filename)
            loader = TextLoader(filepath, encoding='utf-8')
            documents = loader.load()
            all_documents.extend(documents)
    
    # Разделение на чанки
    chunks = text_splitter.split_documents(all_documents)
    
    # Генерация эмбеддингов и сохранение
    print(f"Индексация {len(chunks)} чанков...")
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            embeddings=embeddings.embed_query(chunk.page_content),
            metadatas=[{"source": chunk.metadata.get('source', 'unknown')}],
            ids=[f"doc_{i}"]
        )
    
    print(f"✅ Индексация завершена: {collection.count()} чанков")
    return collection

if __name__ == "__main__":
    index_documents("data/historical_docs")
```

**Запуск индексации:**
```bash
python rag_index.py
```

---

## 🔍 Часть 3: Модуль поиска

```python
# search_module.py
from duckduckgo_search import DDGS
import requests

class InternetSearch:
    """Поиск в интернете через DuckDuckGo"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
    
    def search(self, query, max_results=3):
        """Поиск информации в интернете"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            # Извлечение текста
            snippets = []
            for r in results[:max_results]:
                snippet = f"Title: {r.get('title', '')}\n{r.get('body', '')}"
                snippets.append(snippet)
            
            return "\n\n".join(snippets)
        
        except Exception as e:
            return f"Ошибка поиска: {str(e)}"
    
    def search_wikipedia(self, query, lang="ru"):
        """Поиск в Википедии"""
        import wikipedia
        
        try:
            wikipedia.set_lang(lang)
            search_results = wikipedia.search(query, results=3)
            
            snippets = []
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    snippets.append(f"Wiki: {title}\n{page.summary[:500]}")
                except:
                    continue
            
            return "\n\n".join(snippets)
        
        except Exception as e:
            return f"Ошибка Wikipedia: {str(e)}"

if __name__ == "__main__":
    search = InternetSearch()
    print(search.search("Гоголь Мёртвые души сюжет"))
```

---

## 🧠 Часть 4: RAG-оркестрация

```python
# rag_gogol.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings

class GogolRAG:
    """Модель Гоголя с RAG и поиском"""
    
    def __init__(self, model_path="gogol_finetuned_from_epoch2/checkpoint-1875"):
        self.model_name = "ai-forever/rugpt3small_based_on_gpt2"
        self.load_model(model_path)
        self.load_rag_modules()
    
    def load_model(self, checkpoint_path):
        """Загрузка модели"""
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
        print("✅ Модель загружена")
    
    def load_rag_modules(self):
        """Загрузка модулей RAG"""
        # Эмбеддинги
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Векторная база
        self.chroma_client = chromadb.PersistentClient(path="data/vector_db")
        self.collection = self.chroma_client.get_collection(name="historical_docs")
        
        # Поиск
        from search_module import InternetSearch
        self.search = InternetSearch()
        
        print("✅ RAG модули загружены")
    
    def retrieve_from_db(self, query, top_k=3):
        """Поиск в локальной базе"""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        context = "\n\n".join(results['documents'][0])
        return context
    
    def should_search_internet(self, query):
        """Решение: искать в интернете или нет"""
        # Ключевые слова для интернета
        internet_keywords = [
            "сейчас", "сегодня", "новости", "актуально",
            "2024", "2025", "современный", "последний"
        ]
        
        # Ключевые слова для локальной базы
        local_keywords = [
            "Александр I", "1830", "1840", "законы", "крепостное",
            "Диканька", "Полтава", "история", "XIX век"
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
        
        if search_type == "local" and use_rag:
            context = self.retrieve_from_db(prompt)
        elif search_type == "internet":
            context = self.search.search(prompt)
        
        # Формирование промпта
        if context:
            full_prompt = f"""Используя следующую информацию, ответь на вопрос в стиле Гоголя:

Информация:
{context}

Вопрос: {prompt}

Ответ:"""
        else:
            full_prompt = prompt
        
        # Генерация
        inputs = self.tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True
            )
        
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return result[len(full_prompt):].strip()
    
    def chat(self):
        """Интерактивный чат"""
        print("🤖 Гоголь-RAG чат (введите 'exit' для выхода)")
        
        while True:
            user_input = input("\nВы: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'выход']:
                break
            
            response = self.generate_with_rag(user_input)
            print(f"\nГоголь: {response}")

if __name__ == "__main__":
    rag = GogolRAG()
    rag.chat()
```

---

## 🧪 Часть 5: Тестирование

```python
# test_rag.py
from rag_gogol import GogolRAG

rag = GogolRAG()

# Тест 1: Локальная база
print("Тест 1: Исторический вопрос (локальная база)")
response = rag.generate_with_rag("Какие законы о крепостном праве действовали в 1830-е?")
print(f"Ответ: {response}\n")

# Тест 2: Интернет
print("Тест 2: Актуальная информация (интернет)")
response = rag.generate_with_rag("Где можно купить книги Гоголя сейчас в 2024 году?")
print(f"Ответ: {response}\n")

# Тест 3: Только модель
print("Тест 3: Стиль (без поиска)")
response = rag.generate_with_rag("Напиши фразу в стиле Гоголя о чиновниках")
print(f"Ответ: {response}\n")
```

**Запуск теста:**
```bash
python test_rag.py
```

---

## 📊 Итоговая архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Пользовательский запрос                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Анализ     │
                    │  запроса    │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
   │ Локальная│       │ Интернет │       │ Только  │
   │  база   │       │  поиск   │       │  модель │
   └────┬────┘       └────┬────┘       └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    ┌──────▼──────┐
                    │   GPT-2     │
                    │ (Гоголь)    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Ответ     │
                    └─────────────┘
```

---

## 🚀 Следующие шаги

1. **Создать папку `data/historical_docs/`** и добавить исторические документы
2. **Запустить индексацию:** `python rag_index.py`
3. **Протестировать:** `python test_rag.py`
4. **Запустить чат:** `python rag_gogol.py`

**Время реализации:** ~2-4 часа (без сбора документов)

---

## ⚠️ Ограничения

| Ограничение | Решение |
|-------------|---------|
| Скорость эмбеддингов | Использовать CPU-оптимизированную модель |
| Точность поиска | Настроить top_k и threshold |
| Стоимость API | Использовать DuckDuckGo (бесплатно) |
| Язык | MiniLM поддерживает русский |

---

**Файлы для создания:**
- `rag_index.py` — индексация документов
- `search_module.py` — модуль поиска
- `rag_gogol.py` — основная RAG-логика
- `test_rag.py` — тестирование
- `requirements.txt` — зависимости

Готов начать реализацию, когда вы подтвердите! 🚀
