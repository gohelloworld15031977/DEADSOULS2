"""
Индексация исторических документов для RAG
"""

import os
import sys
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

def index_documents(docs_dir="data/historical_docs", db_path="data/vector_db"):
    """Индексация исторических документов"""
    
    print("=" * 60)
    print("ИНДЕКСАЦИЯ ИСТОРИЧЕСКИХ ДОКУМЕНТОВ")
    print("=" * 60)
    
    # Загрузка эмбеддингов (русский язык)
    print("\nЗагрузка модели эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Создание текстового сплиттера
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Инициализация ChromaDB
    print(f"Создание базы данных: {db_path}")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    # Проверка существования коллекции
    try:
        collection = chroma_client.get_collection(name="historical_docs")
        print(f"Коллекция уже существует, удаление...")
        chroma_client.delete_collection(name="historical_docs")
    except:
        pass
    
    collection = chroma_client.create_collection(name="historical_docs")
    
    all_documents = []
    
    # Чтение всех файлов
    print("\nЧтение документов...")
    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(docs_dir, filename)
            print(f"  - {filename}")
            
            try:
                loader = TextLoader(filepath, encoding='utf-8')
                documents = loader.load()
                
                # Добавляем метаданные
                for doc in documents:
                    doc.metadata['source'] = filename
                
                all_documents.extend(documents)
            except Exception as e:
                print(f"    Ошибка: {e}")
    
    print(f"\nВсего документов: {len(all_documents)}")
    
    # Разделение на чанки
    print("\nРазделение на чанки...")
    chunks = text_splitter.split_documents(all_documents)
    print(f"Количество чанков: {len(chunks)}")
    
    # Генерация эмбеддингов и сохранение
    print("\nГенерация эмбеддингов и сохранение...")
    
    batch_size = 50
    total = len(chunks)
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        
        # Эмбеддинги для чанка
        texts = [chunk.page_content for chunk in batch]
        batch_embeddings = embeddings.embed_documents(texts)
        
        # Метаданные и ID
        metadatas = [{"source": chunk.metadata.get('source', 'unknown')} for chunk in batch]
        ids = [f"doc_{i+j}" for j in range(len(batch))]
        
        # Добавление в коллекцию (конвертация типов)
        collection.add(
            documents=texts,
            embeddings=[emb.tolist() if hasattr(emb, 'tolist') else emb for emb in batch_embeddings],  # type: ignore
            metadatas=metadatas,  # type: ignore
            ids=ids
        )
        
        progress = min(i + batch_size, total)
        print(f"  Прогресс: {progress}/{total} чанков")
    
    # Итоги
    print("\n" + "=" * 60)
    print(f"ИНДЕКСАЦИЯ ЗАВЕРШЕНА!")
    print(f"   Коллекция: {collection.name}")
    print(f"   Всего чанков: {collection.count()}")
    print(f"   База данных: {db_path}")
    print("=" * 60)
    
    return collection

def test_search(collection, query="законы о крепостном праве"):
    """Тестовый поиск"""
    print(f"\nТестовый поиск: '{query}'")
    
    # Эмбеддинг запроса
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    query_embedding = embeddings.embed_query(query)
    
    # Поиск
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    print("\nРезультаты:")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n{i+1}. [Источник: {meta['source']}, Дистанция: {dist:.3f}]")
        print(f"   {doc[:200]}...")

if __name__ == "__main__":
    # Индексация
    collection = index_documents()
    
    # Тест
    test_search(collection, "законы о крепостном праве")
    test_search(collection, "расстояние Полтава Киев")
    test_search(collection, "трактир быт")
