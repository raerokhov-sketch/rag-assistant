"""Семантический поиск по документам в ChromaDB."""

from dataclasses import dataclass

from .vector_store import embed_texts, get_collection


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    distance: float


def search_documents(query: str, top_k: int = 4) -> list[RetrievedChunk]:
    """Возвращает до ``top_k`` ближайших чанков для запроса."""
    if not query.strip():
        raise ValueError("Поисковый запрос не должен быть пустым.")
    if top_k < 1:
        raise ValueError("top_k должен быть положительным.")

    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    return [
        RetrievedChunk(
            text=document,
            source=str(metadata["source"]),
            distance=float(distance),
        )
        for document, metadata, distance in zip(documents, metadatas, distances)
        if document is not None and metadata is not None and distance is not None
    ]
