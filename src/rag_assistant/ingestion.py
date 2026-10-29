"""Загрузка текстовых документов и индексация чанков в ChromaDB."""

import hashlib
from pathlib import Path

from .chunking import SUPPORTED_SUFFIXES, create_chunks
from .config import RAW_DATA_DIR
from .vector_store import embed_texts, get_collection

def _chunk_id(source: str, chunk_index: int) -> str:
    identity = f"{source}:{chunk_index}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()


def ingest_documents(
    source_dir: Path = RAW_DATA_DIR,
    batch_size: int = 100,
) -> int:
    """Индексирует Markdown- и TXT-файлы; возвращает число сохранённых чанков.

    При повторном запуске обновляет чанки изменённых документов и удаляет
    записи для файлов или чанков, которых больше нет в исходных данных.
    """
    source_dir = Path(source_dir)
    if batch_size < 1:
        raise ValueError("Размер пакета эмбеддингов должен быть положительным.")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Папка с исходными документами не найдена: {source_dir}")

    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError(
            f"В {source_dir} нет файлов поддерживаемых форматов: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )

    documents: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    ids: list[str] = []
    source_hashes: dict[str, str] = {}
    source_texts: dict[str, str] = {}
    for path in files:
        source = path.relative_to(source_dir).as_posix()
        contents = path.read_bytes()
        source_hashes[source] = hashlib.sha256(contents).hexdigest()
        source_texts[source] = contents.decode("utf-8")

    collection = get_collection()
    existing = collection.get(include=["metadatas"])
    retained_ids: set[str] = set()
    unchanged_sources: set[str] = set()
    for existing_id, metadata in zip(existing["ids"], existing["metadatas"]):
        if metadata is None:
            continue
        source = metadata.get("source")
        if (
            source in source_hashes
            and metadata.get("content_hash") == source_hashes[source]
        ):
            unchanged_sources.add(source)
            retained_ids.add(existing_id)

    for path in files:
        source = path.relative_to(source_dir).as_posix()
        if source in unchanged_sources:
            continue
        for chunk_index, chunk in enumerate(create_chunks(source_texts[source])):
            documents.append(chunk)
            metadatas.append(
                {
                    "source": source,
                    "chunk_index": chunk_index,
                    "content_hash": source_hashes[source],
                }
            )
            ids.append(_chunk_id(source, chunk_index))

    if not documents and not retained_ids:
        raise ValueError("В исходных файлах не найден текст для индексации.")

    embeddings: list[list[float]] = []
    for start in range(0, len(documents), batch_size):
        embeddings.extend(embed_texts(documents[start : start + batch_size]))
    if len(embeddings) != len(documents):
        raise RuntimeError(
            "Количество полученных эмбеддингов не совпадает с количеством чанков."
        )

    desired_ids = retained_ids.union(ids)
    stale_ids = [
        existing_id
        for existing_id in existing["ids"]
        if existing_id not in desired_ids
    ]

    if documents:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
    if stale_ids:
        collection.delete(ids=stale_ids)
    return len(documents)


if __name__ == "__main__":
    indexed_count = ingest_documents()
    print(f"Проиндексировано чанков: {indexed_count}")
