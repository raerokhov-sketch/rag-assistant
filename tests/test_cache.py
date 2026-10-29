import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rag_assistant.cache import SQLiteAnswerCache
from rag_assistant.__main__ import _run_cache
from rag_assistant.pipeline import answer_question
from rag_assistant.retrieval import RetrievedChunk


class SQLiteAnswerCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache = SQLiteAnswerCache(Path(self.temp_dir.name) / "answers.sqlite3")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_returns_none_for_cache_miss(self) -> None:
        self.assertIsNone(self.cache.get("Какой вопрос?"))

    def test_stores_and_reads_answer_ignoring_whitespace(self) -> None:
        self.cache.put("  Какой   вопрос?  ", "Ответ")

        self.assertEqual(self.cache.get("Какой вопрос?"), "Ответ")

    def test_answer_persists_in_a_new_cache_instance(self) -> None:
        self.cache.put("Вопрос", "Ответ")
        reopened_cache = SQLiteAnswerCache(self.cache.database_path)

        self.assertEqual(reopened_cache.get("Вопрос"), "Ответ")

    def test_pipeline_skips_search_and_generation_on_cache_hit(self) -> None:
        self.cache.put("Вопрос", "Сохранённый ответ")

        with (
            patch("rag_assistant.pipeline.search_documents") as search,
            patch("rag_assistant.pipeline.generate_answer") as generate,
        ):
            answer = answer_question("Вопрос", self.cache)

        self.assertEqual(answer, "Сохранённый ответ")
        search.assert_not_called()
        generate.assert_not_called()

    def test_pipeline_searches_generates_and_caches_answer(self) -> None:
        context = [RetrievedChunk("Факт из документа.", "guide.txt", 0.1)]

        with (
            patch(
                "rag_assistant.pipeline.search_documents",
                return_value=context,
            ) as search,
            patch(
                "rag_assistant.pipeline.generate_answer",
                return_value="Ответ из документов",
            ) as generate,
        ):
            answer = answer_question("Новый вопрос", self.cache, top_k=3)

        self.assertEqual(answer, "Ответ из документов")
        search.assert_called_once_with("Новый вопрос", top_k=3)
        generate.assert_called_once_with("Новый вопрос", context)
        self.assertEqual(self.cache.get("Новый вопрос"), "Ответ из документов")

    def test_pipeline_returns_context_with_cached_answer(self) -> None:
        self.cache.put("Вопрос", "Сохранённый ответ")
        context = [RetrievedChunk("Факт из документа.", "guide.txt", 0.1)]

        with (
            patch(
                "rag_assistant.pipeline.search_documents",
                return_value=context,
            ) as search,
            patch("rag_assistant.pipeline.generate_answer") as generate,
        ):
            from rag_assistant.pipeline import answer_question_with_context

            answer, retrieved_context = answer_question_with_context(
                "Вопрос",
                self.cache,
            )

        self.assertEqual(answer, "Сохранённый ответ")
        self.assertEqual(retrieved_context, context)
        search.assert_called_once_with("Вопрос", top_k=4)
        generate.assert_not_called()

    def test_pipeline_does_not_cache_no_context_answer(self) -> None:
        with (
            patch("rag_assistant.pipeline.search_documents", return_value=[]),
            patch("rag_assistant.pipeline.generate_answer") as generate,
        ):
            answer = answer_question("Вопрос без контекста", self.cache)

        self.assertIn("нет информации", answer)
        self.assertIsNone(self.cache.get("Вопрос без контекста"))
        generate.assert_not_called()

    def test_pipeline_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            answer_question("  ", self.cache)

    def test_clear_removes_cached_answers(self) -> None:
        self.cache.put("Вопрос", "Ответ")

        self.cache.clear()

        self.assertIsNone(self.cache.get("Вопрос"))

    def test_statistics_include_entries_hits_misses_and_hit_rate(self) -> None:
        self.cache.put("Вопрос", "Ответ")
        self.assertEqual(self.cache.get("Вопрос"), "Ответ")
        self.assertIsNone(self.cache.get("Другой вопрос"))

        statistics = self.cache.get_statistics()

        self.assertEqual(statistics.entry_count, 1)
        self.assertEqual(statistics.hits, 1)
        self.assertEqual(statistics.misses, 1)
        self.assertEqual(statistics.hit_rate, 0.5)
        self.assertGreater(statistics.database_size_bytes, 0)

    def test_empty_cache_statistics_have_zero_hit_rate(self) -> None:
        statistics = self.cache.get_statistics()

        self.assertEqual(statistics.entry_count, 0)
        self.assertEqual(statistics.hits, 0)
        self.assertEqual(statistics.misses, 0)
        self.assertEqual(statistics.hit_rate, 0.0)

    def test_cache_cli_shows_statistics_at_start_and_exit(self) -> None:
        self.cache.put("Вопрос", "Ответ")
        output = io.StringIO()

        with (
            patch("rag_assistant.cache.SQLiteAnswerCache", return_value=self.cache),
            patch("builtins.input", side_effect=["Вопрос", "exit"]),
            redirect_stdout(output),
        ):
            _run_cache()

        self.assertEqual(output.getvalue().count("Статистика кеша:"), 2)
        self.assertIn("записей — 1", output.getvalue())
        self.assertIn("попаданий — 1", output.getvalue())

    def test_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            self.cache.get("  ")


if __name__ == "__main__":
    unittest.main()
