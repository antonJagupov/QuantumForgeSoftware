#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
from typing import List, Tuple
import requests
import json

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ========== КОНФИГУРАЦИЯ ==========
INDEX_DIR = "faiss_index"           # папка с индексом FAISS
EMBEDDING_MODEL = "ai-forever/sbert_large_nlu_ru"
OLLAMA_URL = "http://localhost:1234/v1/chat/completions"
OLLAMA_MODEL = "llama3.2:1b"        # или "mistral:7b", "llama3:8b"
TOP_K = 5                           # количество извлекаемых чанков
TEMPERATURE = 0.2                   # меньше креативности для фактов

# ========== ЗАГРУЗКА ВЕКТОРНОЙ БД ==========
print("Загрузка модели эмбеддингов...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("Загрузка индекса FAISS...")
if not os.path.exists(INDEX_DIR):
    print(f"Ошибка: папка с индексом '{INDEX_DIR}' не найдена.")
    print("Сначала создайте индекс через скрипт индексации.")
    sys.exit(1)

vectorstore = FAISS.load_local(
    INDEX_DIR, 
    embeddings, 
    allow_dangerous_deserialization=True
)
print("Готово! Индекс загружен.")

# ========== FEW-SHOT ПРИМЕРЫ (из той же предметной области) ==========
# В реальной системе эти примеры можно извлекать из самой базы знаний.
# Здесь они заданы вручную, основываясь на типичных вопросах из IT/SaaS-компании.
FEW_SHOT_EXAMPLES = [
    {
        "question": "Как активировать виртуальное окружение Python в Windows?",
        "answer": (
            "1. Нужно сначала создать окружение: `python -m venv .venv`.\n"
            "2. Для активации в командной строке выполнить: `.venv\\Scripts\\activate.bat`.\n"
            "3. В PowerShell: `.venv\\Scripts\\Activate.ps1`, возможно потребуется разрешить выполнение скриптов."
        )
    },
    {
        "question": "Какая технология используется для оркестрации контейнеров в продакшене QuantumForge?",
        "answer": (
            "1. В продакшене используется AWS EKS (Elastic Kubernetes Service).\n"
            "2. Управление конфигурацией через Helm charts.\n"
            "3. Деплой через ArgoCD."
        )
    }
]

# ========== ПОСТРОЕНИЕ ПРОМПТА (CoT + Few-shot) ==========
def build_prompt(query: str, retrieved_docs: List[Document]) -> str:
    """
    Формирует промпт с:
    - системной инструкцией Chain-of-Thought
    - few-shot примерами
    - контекстом из базы знаний
    - пользовательским вопросом
    """
    # Системная инструкция с CoT
    system_instruction = (
        "Ты — технический ассистент компании QuantumForge Software.\n"
        "Всегда следуй этим шагам перед ответом:\n"
        "1. Проанализируй вопрос пользователя и определи, какие ключевые понятия нужно найти.\n"
        "2. Изучи предоставленные фрагменты документации (контекст).\n"
        "3. Если контекст содержит ответ — выдели его и объясни, опираясь на источник.\n"
        "4. Если ответа нет в контексте — честно скажи: «Я не знаю. Возможно, информация отсутствует в документации.»\n"
        "5. Не придумывай факты, не указанные в контексте.\n"
        "6. Отвечай на русском языке, подробно, но по делу.\n"
    )
    
    # Few-shot примеры
    few_shot_text = "Вот примеры правильных ответов на похожие вопросы:\n"
    for ex in FEW_SHOT_EXAMPLES:
        few_shot_text += f"Вопрос: {ex['question']}\nОтвет: {ex['answer']}\n\n"
    
    # Контекст из поиска
    context_text = "Вот релевантные фрагменты из базы знаний:\n"
    for i, doc in enumerate(retrieved_docs, 1):
        context_text += f"\n--- Фрагмент {i} (источник: {doc.metadata.get('source', 'unknown')}) ---\n"
        context_text += doc.page_content[:800] + "\n"
    
    # Финальная часть
    user_prompt = (
        f"Теперь ответь на вопрос пользователя, строго следуя инструкции и используя только предоставленный контекст.\n"
        f"Вопрос: {query}\n"
        f"Ответ (сначала опиши свои шаги рассуждения, затем дай финальный ответ):"
    )
    
    full_prompt = f"{system_instruction}\n\n{few_shot_text}\n\n{context_text}\n\n{user_prompt}"
    return full_prompt

# ========== ВЫЗОВ ЛОКАЛЬНОЙ LLM (Ollama) ==========
def ask_llm(prompt: str) -> str:
    """Отправляет запрос к локальному серверу LM Studio."""
    payload = {
        "model": "local-model", # Название не важно, LM Studio сам его подставит
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=360)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Ошибка при обращении к LM Studio: {e}\nПроверьте, запущен ли сервер (кнопка Start Server)."

# ========== ОСНОВНАЯ ФУНКЦИЯ RAG ==========
def rag_answer(query: str) -> Tuple[str, List[Document]]:
    """
    Принимает вопрос, ищет в векторной БД, формирует промпт,
    вызывает LLM и возвращает ответ вместе с использованными документами.
    """
    # Поиск похожих чанков
    docs = vectorstore.similarity_search(query, k=TOP_K)
    if not docs:
        return "Я не знаю. В базе знаний нет релевантных документов по вашему запросу.", []
    
    # Формирование промпта
    prompt = build_prompt(query, docs)
    
    # Генерация ответа
    answer = ask_llm(prompt)
    
    return answer, docs

# ========== КОНСОЛЬНЫЙ ИНТЕРФЕЙС (REPL) ==========
def main():
    print("\n" + "="*60)
    print("RAG-бот компании QuantumForge Software")
    print("Введите ваш вопрос (или 'exit' для выхода, 'source' для показа источников последнего ответа)")
    print("="*60)
    
    last_docs = []
    while True:
        user_input = input("\n👉 Ваш вопрос: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            print("До свидания!")
            break
        if user_input.lower() == "source":
            if last_docs:
                print("\n📚 Источники последнего ответа:")
                for i, doc in enumerate(last_docs, 1):
                    src = doc.metadata.get('source', 'неизвестный файл')
                    print(f"{i}. {src} (фрагмент: {doc.page_content[:150]}...)")
            else:
                print("Нет сохранённых источников. Сначала задайте вопрос.")
            continue
        if not user_input:
            continue
        
        print("\n🤔 Думаю... (поиск в базе знаний + генерация ответа с CoT)\n")
        answer, docs = rag_answer(user_input)
        last_docs = docs
        print("📢 ОТВЕТ:")
        print(answer)
        
        # Показываем источники (первые 3)
        if docs:
            print("\n🔗 Источники:")
            for i, doc in enumerate(docs[:3], 1):
                src = doc.metadata.get('source', 'unknown')
                print(f"   {i}. {src}")

if __name__ == "__main__":
    main()
