import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_assistant.ingestion import ingest_documents


class IngestDocumentsTests(unittest.TestCase):
    def test_indexes_text_and_removes_stale_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory) / "raw"
            source_dir.mkdir()
            (source_dir / "guide.txt").write_text(
                " ".join(f"Слово{index}." for index in range(150)),
                encoding="utf-8",
            )
            collection = _FakeCollection(
                ids=["old-chunk"],
                metadatas=[{"source": "removed.txt"}],
            )

            with (
                patch(
                    "rag_assistant.ingestion.embed_texts",
                    side_effect=lambda texts: [[0.1] for _ in texts],
                ),
                patch("rag_assistant.ingestion.get_collection", return_value=collection),
            ):
                count = ingest_documents(source_dir)

        self.assertGreater(count, 0)
        self.assertEqual(len(collection.upserted["ids"]), count)
        self.assertEqual(collection.deleted_ids, ["old-chunk"])
        self.assertEqual(collection.upserted["metadatas"][0]["source"], "guide.txt")

    def test_rejects_directory_without_supported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                ingest_documents(Path(temporary_directory))

    def test_skips_unchanged_file_without_creating_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_dir = Path(temporary_directory) / "raw"
            source_dir.mkdir()
            contents = "Содержимое уже проиндексированного файла."
            (source_dir / "guide.txt").write_text(contents, encoding="utf-8")
            content_hash = hashlib.sha256(contents.encode("utf-8")).hexdigest()
            collection = _FakeCollection(
                ids=["existing-chunk"],
                metadatas=[
                    {
                        "source": "guide.txt",
                        "chunk_index": 0,
                        "content_hash": content_hash,
                    }
                ],
            )

            with (
                patch("rag_assistant.ingestion.embed_texts") as embed_texts,
                patch("rag_assistant.ingestion.get_collection", return_value=collection),
            ):
                count = ingest_documents(source_dir)

        self.assertEqual(count, 0)
        embed_texts.assert_not_called()
        self.assertEqual(collection.upserted, {})
        self.assertEqual(collection.deleted_ids, [])


class _FakeCollection:
    def __init__(self, ids: list[str], metadatas: list[dict[str, str]]) -> None:
        self.ids = ids
        self.metadatas = metadatas
        self.upserted: dict[str, object] = {}
        self.deleted_ids: list[str] = []

    def get(self, include: list[str]) -> dict[str, list[object]]:
        return {"ids": self.ids, "metadatas": self.metadatas}

    def upsert(self, **kwargs: object) -> None:
        self.upserted = kwargs

    def delete(self, ids: list[str]) -> None:
        self.deleted_ids = ids


if __name__ == "__main__":
    unittest.main()
