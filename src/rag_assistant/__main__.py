"""Единая точка запуска компонентов RAG-ассистента."""

import argparse
from collections.abc import Callable


def _run_cache() -> None:
    from .cache import SQLiteAnswerCache

    cache = SQLiteAnswerCache()

    def show_statistics() -> None:
        statistics = cache.get_statistics()
        print(
            "\nСтатистика кеша: "
            f"записей — {statistics.entry_count}, "
            f"попаданий — {statistics.hits}, "
            f"промахов — {statistics.misses}, "
            f"hit rate — {statistics.hit_rate:.1%}, "
            f"размер БД — {statistics.database_size_bytes} байт."
        )

    print(
        "Кеш ответов. Введите вопрос для поиска, 'stats' для статистики, "
        "'clear' для очистки, 'exit' для выхода."
    )
    show_statistics()
    while True:
        command = input("\nКеш: ").strip()
        if command.lower() in {"exit", "quit", "выход"}:
            show_statistics()
            return
        if command.lower() == "clear":
            cache.clear()
            print("Кеш очищен.")
            continue
        if command.lower() == "stats":
            show_statistics()
            continue
        if not command:
            continue

        answer = cache.get(command)
        print(answer if answer is not None else "Ответа в кеше нет.")


def _run_chunking() -> None:
    from .chunking import chunk_documents

    documents = chunk_documents()
    for source, chunks in documents:
        print(f"\n=== Документ: {source} ===")
        if not chunks:
            print("Текст пустой, чанки не созданы.")
            continue
        for index, chunk in enumerate(chunks, start=1):
            print(f"\n--- Чанк {index} ({len(chunk)} символов) ---\n{chunk}")


def _run_assistant() -> None:
    from .pipeline import answer_question

    print("RAG-ассистент готов. Для выхода введите 'exit'.")
    while True:
        question = input("\nВопрос: ").strip()
        if question.lower() in {"exit", "quit", "выход"}:
            return
        if question:
            print(f"\n{answer_question(question)}")


def _run_ragas() -> None:
    from evaluation.evaluate_rag import evaluate_rag_system

    evaluate_rag_system()


def _run_ingestion() -> None:
    from .ingestion import ingest_documents

    indexed_count = ingest_documents()
    print(f"Проиндексировано чанков: {indexed_count}")


def _run_all() -> None:
    actions: dict[str, tuple[str, Callable[[], None]]] = {
        "1": ("Кеш ответов", _run_cache),
        "2": ("Чанкинг текста", _run_chunking),
        "3": ("RAG-ассистент", _run_assistant),
        "4": ("Оценка RAGAS", _run_ragas),
        "5": ("Индексация документов", _run_ingestion),
    }
    print("RAG-ассистент: кеш, чанкинг, ассистент, RAGAS и индексация.")
    while True:
        print("\n1. Кеш  2. Чанкинг  3. Ассистент  4. RAGAS  5. Индексация  0. Выход")
        choice = input("Выберите режим: ").strip()
        if choice == "0":
            return
        action = actions.get(choice)
        if action is None:
            print("Неизвестный режим.")
            continue
        print(f"\n--- {action[0]} ---")
        action[1]()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Запуск компонентов RAG-ассистента по отдельности или через общее меню."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("cache", "chunking", "assistant", "ragas", "ingest", "all"),
        default="all",
        help="режим запуска (по умолчанию: all)",
    )
    args = parser.parse_args()
    actions: dict[str, Callable[[], None]] = {
        "cache": _run_cache,
        "chunking": _run_chunking,
        "assistant": _run_assistant,
        "ragas": _run_ragas,
        "ingest": _run_ingestion,
        "all": _run_all,
    }
    actions[args.mode]()


if __name__ == "__main__":
    main()
