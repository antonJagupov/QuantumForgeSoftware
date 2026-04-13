import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS           # <-- Импорт из langchain_community
from langchain_community.embeddings import HuggingFaceEmbeddings   # <-- Импорт эмбеддингов

# --- Конфигурация ---
DATA_DIR = "knowledge_base"          # Папка с вашими 33 txt файлами
INDEX_DIR = "faiss_index"            # Папка для сохранения индекса
CHUNK_SIZE = 500                     # Размер чанка в символах
CHUNK_OVERLAP = 50                   # Перекрытие между чанками

# --- 1. Загрузка документов из txt файлов ---
def load_documents_from_dir(directory: str):
    """Загружает все .txt файлы из указанной директории и возвращает список объектов Document."""
    documents = []
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Директория '{directory}' не найдена. Создайте её и поместите туда txt файлы.")
    
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    text = file.read()
                    # Создаём объект Document LangChain с содержимым и метаданными
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": filename,  # Исходный файл
                            "path": filepath     # Полный путь
                        }
                    )
                    documents.append(doc)
                    print(f"Загружен: {filename}")
            except Exception as e:
                print(f"Ошибка при чтении {filename}: {e}")
    
    print(f"\nВсего загружено документов: {len(documents)}")
    return documents

# --- 2. Разбиение документов на чанки ---
def split_documents(docs):
    """Разбивает документы на чанки с сохранением метаданных."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(docs)
    
    # Добавляем id для каждого чанка
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    print(f"Документы разбиты на {len(chunks)} чанков.")
    # Пример первого чанка для проверки
    if chunks:
        print(f"Пример чанка: {chunks[0].page_content[:100]}...")
    return chunks

# --- 3. Создание эмбеддингов с помощью sbert_large_nlu_ru ---
def get_embeddings_model():
    """Загружает модель эмбеддингов из Hugging Face."""
    model_name = "ai-forever/sbert_large_nlu_ru"
    model_kwargs = {'device': 'cpu'}  # Используйте 'cuda' для GPU, если доступно
    encode_kwargs = {'normalize_embeddings': True}  # Нормализация для косинусного сходства
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f"Модель эмбеддингов '{model_name}' загружена.")
    return embeddings

# --- 4. Создание и сохранение индекса FAISS ---
def create_and_save_faiss_index(chunks, embeddings, save_dir):
    """Создаёт индекс FAISS из чанков и сохраняет его на диск."""
    # Создаём векторное хранилище FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Сохраняем индекс локально
    vectorstore.save_local(save_dir)
    print(f"Индекс FAISS успешно сохранён в директории: {save_dir}")
    
    # Дополнительно сохраняем чанки и метаданные в pickle для отладки
    with open(os.path.join(save_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print("Чанки и их метаданные сохранены в 'chunks.pkl'.")
    
    return vectorstore

# --- Основной блок выполнения ---
if __name__ == "__main__":
    print("=== Начало создания векторного индекса ===\n")
    
    # Шаг 1: Загрузка документов
    docs = load_documents_from_dir(DATA_DIR)
    if not docs:
        print("Не найдено ни одного .txt файла. Проверьте содержимое директории 'knowledge_base'.")
        exit(1)
    
    # Шаг 2: Разбиение на чанки
    chunks = split_documents(docs)
    if not chunks:
        print("Не удалось создать чанки.")
        exit(1)
    
    # Шаг 3: Инициализация модели эмбеддингов
    embeddings = get_embeddings_model()
    
    # Шаг 4: Создание и сохранение индекса
    vectorstore = create_and_save_faiss_index(chunks, embeddings, INDEX_DIR)
    
    print("\n=== Индекс успешно создан! ===")
    print(f"Вы можете найти его в папке '{INDEX_DIR}'.")
