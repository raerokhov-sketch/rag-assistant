"""SQLite-кеш ответов на вопросы."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT

DEFAULT_CACHE_PATH = PROJECT_ROOT / "storage" / "cache" / "answers.sqlite3"


@dataclass(frozen=True)
class CacheStatistics:
    entry_count: int
    hits: int
    misses: int
    hit_rate: float
    database_size_bytes: int


def _normalize_question(question: str) -> str:
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("Вопрос не должен быть пустым.")
    return normalized


class SQLiteAnswerCache:
    """Хранит готовые ответы по нормализованному тексту вопроса."""

    def __init__(self, database_path: Path = DEFAULT_CACHE_PATH) -> None:
        self.database_path = Path(database_path)
        self._hits = 0
        self._misses = 0
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS answers (
                        question TEXT PRIMARY KEY,
                        answer TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def get(self, question: str) -> str | None:
        normalized_question = _normalize_question(question)
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                "SELECT answer FROM answers WHERE question = ?",
                (normalized_question,),
            ).fetchone()
        if row is None:
            self._misses += 1
        else:
            self._hits += 1
        return None if row is None else str(row[0])

    def put(self, question: str, answer: str) -> None:
        normalized_question = _normalize_question(question)
        if not answer.strip():
            raise ValueError("Нельзя сохранить пустой ответ в кеш.")

        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO answers (question, answer)
                    VALUES (?, ?)
                    ON CONFLICT(question) DO UPDATE SET
                        answer = excluded.answer,
                        created_at = CURRENT_TIMESTAMP
                    """,
                    (normalized_question, answer),
                )

    def clear(self) -> None:
        """Удаляет все ответы, например после обновления документов."""
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute("DELETE FROM answers")

    def get_statistics(self) -> CacheStatistics:
        """Возвращает размер кеша и счётчики обращений текущего экземпляра."""
        with closing(sqlite3.connect(self.database_path)) as connection:
            entry_count = int(
                connection.execute("SELECT COUNT(*) FROM answers").fetchone()[0]
            )
        total_requests = self._hits + self._misses
        return CacheStatistics(
            entry_count=entry_count,
            hits=self._hits,
            misses=self._misses,
            hit_rate=self._hits / total_requests if total_requests else 0.0,
            database_size_bytes=self.database_path.stat().st_size,
        )
