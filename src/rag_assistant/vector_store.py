"""Общие настройки и функции для локального векторного хранилища ChromaDB."""

from collections.abc import Sequence

import chromadb

from .config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL, create_openai_client


def get_collection():
    """Открывает локальную коллекцию документов, создавая её при необходимости."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Создаёт эмбеддинги через настроенный OpenAI-совместимый endpoint."""
    if not texts:
        return []

    response = create_openai_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=list(texts),
    )
    ordered_embeddings = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered_embeddings]
