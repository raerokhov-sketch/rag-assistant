"""Скрипт для оценки качества RAG-системы через RAGAS и ProxyAPI."""

from collections.abc import Sequence

from rag_assistant.config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    OPENAI_BASE_URL,
    require_api_key,
)
from rag_assistant.pipeline import answer_question_with_context

# Предопределённые вопросы для оценки
EVALUATION_QUESTIONS = [
    "Что такое машинное обучение и какие у него основные типы?",
    "Что такое RAG и почему он полезен для языковых моделей?",
    "Зачем в RAG-системах используются векторные базы данных? Приведите примеры.",
    "Какие метрики используются для оценки качества RAG-систем?",
    "Как температура влияет на ответы языковой модели и какое значение лучше выбрать для фактических "
    "RAG-ответов?"
]

GROUND_TRUTHS = {
    "Что такое машинное обучение и какие у него основные типы?":
        "Машинное обучение — это раздел искусственного интеллекта, позволяющий компьютерам обучаться "
        "на данных без явного программирования. Основные типы: обучение с учителем, обучение "
        "без учителя и обучение с подкреплением. При обучении с учителем используются размеченные "
        "данные для предсказания результатов.",
    "Что такое RAG и почему он полезен для языковых моделей?":
        "RAG (Retrieval-Augmented Generation) — подход, который сочетает извлечение информации из базы "
        "знаний с генерацией текста языковой моделью. Сначала система находит релевантные документы, "
        "затем использует их как контекст для генерации ответа. Это позволяет отвечать на основе "
        "актуальной информации без дополнительного обучения модели.",
    "Зачем в RAG-системах используются векторные базы данных? Приведите примеры.":
        "Векторные базы данных оптимизированы для хранения векторных представлений и быстрого поиска "
        "ближайших соседей. Они применяются в RAG-системах, рекомендательных системах и семантическом поиске. "
        "Примеры: ChromaDB, Pinecone и Weaviate.",
    "Какие метрики используются для оценки качества RAG-систем?":
        "Для оценки RAG-систем используются специализированные метрики: faithfulness, answer relevance "
        "и context precision. Также для генерации текста применяются BLEU и ROUGE, а для классификации — "
        "точность. Человеческая оценка остаётся важным эталоном, но автоматические метрики ускоряют итерации.",
    "Как температура влияет на ответы языковой модели и какое значение лучше выбрать для фактических "
    "RAG-ответов?":
        "Температура контролирует случайность генерации. Низкая температура, примерно 0.1–0.3, делает "
        "ответы более детерминированными и сфокусированными; высокая, примерно 0.8–1.0, увеличивает "
        "креативность и разнообразие. Для фактических вопросов в RAG обычно используют низкую температуру."
}


def prepare_dataset(questions: Sequence[str]):
    """Подготавливает ответы, найденный контекст и эталонные ответы для RAGAS."""
    from datasets import Dataset

    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("Получение ответов от ассистента...")
    
    for i, question in enumerate(questions, 1):
        print(f"  Обработка вопроса {i}/{len(questions)}: {question}")
        
        answer, context = answer_question_with_context(question)

        questions_list.append(question)
        answers_list.append(answer)
        contexts_list.append([chunk.text for chunk in context])
        ground_truths_list.append(GROUND_TRUTHS.get(question, ""))

    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list,
    }
    return Dataset.from_dict(dataset_dict)


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы
    """
    try:
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "Не удалось загрузить RAGAS и LangChain. Установите совместимые "
            "зависимости командой: pip install -r requirements.txt"
        ) from exc

    print("=" * 60)
    print("Оценка качества RAG-системы через RAGAS")
    print("=" * 60)

    # Подготавливаем датасет
    dataset = prepare_dataset(EVALUATION_QUESTIONS)
    
    print("\nЗапуск оценки метрик...")
    print("Метрики: faithfulness, answer_relevancy, context_precision")
    
    openai_api_key = require_api_key()

    # Оба клиента RAGAS используют тот же ProxyAPI endpoint, что и RAG.
    langchain_embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=openai_api_key,
        base_url=OPENAI_BASE_URL,
    )
    langchain_llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=openai_api_key,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
    ragas_llm = LangchainLLMWrapper(langchain_llm)
    metrics_to_use = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextPrecision(llm=ragas_llm),
    ]
    
    # Запускаем оценку
    result = evaluate(
        dataset=dataset,
        metrics=metrics_to_use,
    )
    
    # Выводим результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 60)
    
    # RAGAS хранит построчные результаты в result.scores.
    import math

    scores = result.scores

    for metric, description in [
        ("faithfulness", "верность ответа"),
        ("answer_relevancy", "релевантность ответа"),
        ("context_precision", "точность контекста"),
    ]:
        values = [
            float(row[metric])
            for row in scores
            if row.get(metric) is not None and not math.isnan(float(row[metric]))
        ]
        if values:
            average = sum(values) / len(values)
            print(f"\n{metric} ({description}): {average:.4f}")
        else:
            print(f"\n{metric} ({description}): не удалось вычислить")

    print("\n" + "=" * 60)
    print("ДЕТАЛИ ПО ВОПРОСАМ")
    print("=" * 60)

    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\nВопрос {i + 1}: {question}")
        if i >= len(scores):
            print("  Для вопроса нет результата")
            continue

        for metric in ("faithfulness", "answer_relevancy", "context_precision"):
            value = scores[i].get(metric)
            if value is None or math.isnan(float(value)):
                print(f"  {metric}: не удалось вычислить")
            else:
                print(f"  {metric}: {float(value):.4f}")

if __name__ == "__main__":
    evaluate_rag_system()
