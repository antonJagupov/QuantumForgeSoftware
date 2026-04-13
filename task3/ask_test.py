from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Загружаем существующий индекс
embeddings = HuggingFaceEmbeddings(model_name="ai-forever/sbert_large_nlu_ru")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Функция поиска
def search(query, k=5):
    results = vectorstore.similarity_search(query, k=k)
    print(f"\nРезультаты поиска по запросу: '{query}'\n")
    for i, doc in enumerate(results):
        print(f"Результат {i+1}:")
        print(f"Источник: {doc.metadata['source']}")
        print(f"Чанк ID: {doc.metadata.get('chunk_id', 'N/A')}")
        print(f"Текст: {doc.page_content[:300]}...\n")

# Пример поиска
search("Кто из сыновей ампиратора поддался Ереси?")
