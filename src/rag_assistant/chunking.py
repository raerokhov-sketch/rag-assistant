"""Разбиение текста на чанки с сохранением границ абзацев и предложений."""

import re
from pathlib import Path

from .config import RAW_DATA_DIR

MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

_PARAGRAPH_SEPARATOR = re.compile(r"\n\s*\n+")
_SENTENCE_SEPARATOR = re.compile(r'(?<=[.!?…])["»”)]*\s+')


def _join_units(units: list[str]) -> str:
    return " ".join(units)


def _split_long_paragraph(paragraph: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in _SENTENCE_SEPARATOR.split(paragraph)
        if sentence.strip()
    ]


def _trailing_overlap(text: str, overlap: int) -> str:
    if overlap == 0 or len(text) <= overlap:
        return text if overlap else ""

    start = len(text) - overlap
    next_word = text.find(" ", start)
    if next_word == -1:
        return ""
    return text[next_word + 1 :]


def _pack_units(units: list[str], max_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    for unit in units:
        candidate = _join_units([*current, unit])
        if len(candidate) <= max_size:
            current.append(unit)
            continue

        if current:
            previous_chunk = _join_units(current)
            chunks.append(previous_chunk)
            current_text = _trailing_overlap(previous_chunk, overlap)
            current = [current_text] if current_text else []

        candidate = _join_units([*current, unit])
        if current and len(candidate) > max_size:
            available_overlap = max(0, max_size - len(unit) - 1)
            current_text = _trailing_overlap(
                previous_chunk,
                min(overlap, available_overlap),
            )
            current = [current_text] if current_text else []

        # An unusually long sentence stays intact rather than being cut mid-sentence.
        current.append(unit)
        if len(_join_units(current)) > max_size:
            chunks.append(_join_units(current))
            current = []

    if current:
        chunks.append(_join_units(current))
    return chunks


def _enforce_minimum(chunks: list[str], min_size: int, max_size: int) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        if (
            merged
            and len(chunk) < min_size
            and len(merged[-1]) + 1 + len(chunk) <= max_size
        ):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


def create_chunks(
    text: str,
    min_size: int = MIN_CHUNK_SIZE,
    max_size: int = MAX_CHUNK_SIZE,
    overlap: int | None = None,
) -> list[str]:
    """Делит текст по абзацам, а длинные абзацы — по предложениям.

    Размеры измеряются в символах. Предложение длиннее ``max_size``
    сохраняется целиком, чтобы не разрывать его посередине.
    """
    if min_size < 1 or max_size < 1:
        raise ValueError("Размеры чанка должны быть положительными.")
    if min_size > max_size:
        raise ValueError("Минимальный размер чанка не может превышать максимальный.")
    if overlap is None:
        overlap = min(CHUNK_OVERLAP, max_size // 5)
    if overlap < 0 or overlap >= max_size:
        raise ValueError("Overlap должен быть неотрицательным и меньше max_size.")

    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_SEPARATOR.split(text.strip())
        if paragraph.strip()
    ]
    if not paragraphs:
        return []

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_size:
            units.append(paragraph)
        else:
            units.extend(_split_long_paragraph(paragraph))

    return _enforce_minimum(_pack_units(units, max_size, overlap), min_size, max_size)


def chunk_documents(
    source_dir: Path = RAW_DATA_DIR,
) -> list[tuple[str, list[str]]]:
    """Разбивает поддерживаемые текстовые документы из папки на чанки."""
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Папка с исходными документами не найдена: {source_dir}"
        )

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

    return [
        (
            path.relative_to(source_dir).as_posix(),
            create_chunks(path.read_text(encoding="utf-8")),
        )
        for path in files
    ]
