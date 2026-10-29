import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chromadb

from rag_assistant.ingestion import ingest_documents
from rag_assistant.retrieval import search_documents
from rag_assistant.vector_store import get_collection


class ChromaIntegrationTests(unittest.TestCase):
    def test_ingests_and_retrieves_document_without_external_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_dir = temporary_path / "raw"
            source_dir.mkdir()
            (source_dir / "guide.txt").write_text(
                "Эмбеддинги помогают искать документы по смыслу. " * 5,
                encoding="utf-8",
            )
            chroma_path = temporary_path / "chroma"
            client = chromadb.PersistentClient(path=str(chroma_path))

            try:
                with (
                    patch("rag_assistant.vector_store.CHROMA_DIR", chroma_path),
                    patch(
                        "rag_assistant.vector_store.chromadb.PersistentClient",
                        return_value=client,
                    ),
                ):
                    with patch(
                        "rag_assistant.ingestion.embed_texts",
                        side_effect=lambda texts: [[1.0, 0.0] for _ in texts],
                    ):
                        indexed_count = ingest_documents(source_dir)

                    with patch(
                        "rag_assistant.retrieval.embed_texts",
                        return_value=[[1.0, 0.0]],
                    ):
                        results = search_documents("Как искать документы?", top_k=1)

                    collection = get_collection()
                    stored_count = collection.count()
            finally:
                client.close()

        self.assertEqual(indexed_count, 1)
        self.assertEqual(stored_count, 1)
        self.assertEqual(len(results), 1)
        self.assertIn("Эмбеддинги помогают искать документы", results[0].text)
        self.assertEqual(results[0].source, "guide.txt")


if __name__ == "__main__":
    unittest.main()
