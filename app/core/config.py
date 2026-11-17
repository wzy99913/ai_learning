"""Centralised configuration loaded from the project's local .env file."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    project_dir: Path
    upload_dir: Path
    chroma_db_path: Path
    chroma_collection_name: str
    deepseek_api_key: str | None
    deepseek_base_url: str
    deepseek_model: str
    redis_url: str
    chat_history_ttl: int


@lru_cache
def get_settings() -> Settings:
    """Return the one immutable configuration object used by the project."""
    legacy_db_path = Path(r"E:\AI学习\LLM_api_learning\chroma_db")

    return Settings(
        project_dir=PROJECT_DIR,
        upload_dir=PROJECT_DIR / "knowledge_base" / "uploads",
        chroma_db_path=Path(
            os.getenv("CHROMA_DB_PATH", str(legacy_db_path))
        ),
        chroma_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME", "njust_knowledge"
        ),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        deepseek_base_url=os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        redis_url=os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0"
),
       chat_history_ttl=int(
    os.getenv(
        "CHAT_HISTORY_TTL",
        "86400"
    )
),
    )
