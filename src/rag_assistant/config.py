"""Загрузка настроек проекта из .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"
CHROMA_COLLECTION = "documents"
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://api.proxyapi.ru/openai/v1",
).strip()
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0"))


def require_api_key() -> str:
    """Возвращает ключ API или сообщает, как настроить проект."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY не задан. Укажите его в .env в корне проекта."
        )
    return OPENAI_API_KEY


def create_openai_client():
    """Создаёт OpenAI SDK-клиент с настройками ProxyAPI из .env."""
    from openai import OpenAI

    return OpenAI(
        api_key=require_api_key(),
        base_url=OPENAI_BASE_URL,
    )
