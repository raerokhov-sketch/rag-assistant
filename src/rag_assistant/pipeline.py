"""Оркестрация полного цикла RAG."""

from .cache import SQLiteAnswerCache
from .generation import NO_CONTEXT_ANSWER, generate_answer
from .retrieval import RetrievedChunk, search_documents


def answer_question(
    question: str,
    cache: SQLiteAnswerCache | None = None,
    top_k: int = 4,
) -> str:
    """Ищет ответ в кеше, затем в документах и при необходимости вызывает LLM."""
    normalized_question = " ".join(question.split())
    if not normalized_question:
        raise ValueError("Вопрос не должен быть пустым.")
    if top_k < 1:
        raise ValueError("top_k должен быть положительным.")

    answer_cache = cache if cache is not None else SQLiteAnswerCache()
    cached_answer = answer_cache.get(normalized_question)
    if cached_answer is not None:
        return cached_answer

    context: list[RetrievedChunk] = search_documents(normalized_question, top_k=top_k)
    if not context:
        return NO_CONTEXT_ANSWER

    answer = generate_answer(normalized_question, context)
    if answer.strip():
        answer_cache.put(normalized_question, answer)
    return answer


def answer_question_with_context(
    question: str,
    cache: SQLiteAnswerCache | None = None,
    top_k: int = 4,
) -> tuple[str, list[RetrievedChunk]]:
    """Возвращает ответ и найденный контекст, в том числе при попадании в кеш."""
    normalized_question = " ".join(question.split())
    if not normalized_question:
        raise ValueError("Вопрос не должен быть пустым.")
    if top_k < 1:
        raise ValueError("top_k должен быть положительным.")

    answer_cache = cache if cache is not None else SQLiteAnswerCache()
    cached_answer = answer_cache.get(normalized_question)
    context = search_documents(normalized_question, top_k=top_k)
    if cached_answer is not None:
        return cached_answer, context
    if not context:
        return NO_CONTEXT_ANSWER, []

    answer = generate_answer(normalized_question, context)
    if answer.strip():
        answer_cache.put(normalized_question, answer)
    return answer, context
