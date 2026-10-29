import tempfile
import unittest
from pathlib import Path

from rag_assistant.chunking import chunk_documents, create_chunks


class CreateChunksTests(unittest.TestCase):
    def test_returns_no_chunks_for_empty_text(self) -> None:
        self.assertEqual(create_chunks(" \n\n "), [])

    def test_uses_paragraphs_as_boundaries_for_short_text(self) -> None:
        first = "Первый абзац. " * 8
        second = "Второй абзац. " * 8

        chunks = create_chunks(f"{first}\n\n{second}", min_size=1)

        self.assertEqual(len(chunks), 1)
        self.assertIn("Первый абзац.", chunks[0])
        self.assertIn("Второй абзац.", chunks[0])

    def test_splits_long_paragraph_at_sentence_boundaries(self) -> None:
        sentences = [
            f"Предложение номер {index} " + ("слово " * 24) + f"{index}."
            for index in range(1, 5)
        ]

        chunks = create_chunks(" ".join(sentences), min_size=1, max_size=500)

        self.assertGreater(len(chunks), 1)
        for sentence in sentences:
            self.assertTrue(any(sentence in chunk for chunk in chunks))

    def test_overlaps_chunks_without_exceeding_maximum(self) -> None:
        sentences = [
            f"Предложение номер {index} " + ("слово " * 24) + f"{index}."
            for index in range(1, 5)
        ]

        chunks = create_chunks(" ".join(sentences), min_size=1, max_size=500)

        self.assertGreater(len(chunks[0]), 100)
        overlap_start = chunks[0][-100:].split(" ", 1)[1]
        self.assertTrue(chunks[1].startswith(overlap_start))
        self.assertGreaterEqual(len(overlap_start), 80)
        self.assertLessEqual(len(overlap_start), 100)
        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))

    def test_keeps_a_sentence_longer_than_maximum_intact(self) -> None:
        sentence = "Очень длинное предложение " + ("слово " * 100) + "."

        chunks = create_chunks(sentence, min_size=1, max_size=100)

        self.assertEqual(chunks, [sentence])

    def test_rejects_invalid_chunk_sizes(self) -> None:
        with self.assertRaises(ValueError):
            create_chunks("Текст", min_size=200, max_size=100)


class ChunkDocumentsTests(unittest.TestCase):
    def test_chunks_supported_documents_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory)
            nested_dir = source_dir / "nested"
            nested_dir.mkdir()
            (source_dir / "guide.TXT").write_text(
                "Содержимое документа.",
                encoding="utf-8",
            )
            (nested_dir / "notes.md").write_text("Заметки.", encoding="utf-8")
            (source_dir / "ignored.pdf").write_text("Не текстовый файл.", encoding="utf-8")

            documents = chunk_documents(source_dir)

        self.assertEqual(
            documents,
            [
                ("guide.TXT", ["Содержимое документа."]),
                ("nested/notes.md", ["Заметки."]),
            ],
        )

    def test_rejects_directory_without_supported_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                chunk_documents(Path(temporary_directory))


if __name__ == "__main__":
    unittest.main()
