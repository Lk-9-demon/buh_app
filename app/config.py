from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Buh Chat"
    debug: bool = False

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-small"
    answer_model: str = "gpt-5.4-mini"
    answer_temperature: float = 0.1

    documents_dir: Path = BASE_DIR / "data" / "documents"
    storage_dir: Path = BASE_DIR / "storage"
    chroma_dir: Path = BASE_DIR / "storage" / "chroma"
    state_db_path: Path = BASE_DIR / "storage" / "state.db"

    allowed_extensions: tuple[str, ...] = (".pdf", ".xlsx", ".docx", ".csv", ".txt")
    max_chunk_chars: int = 1400
    chunk_overlap_chars: int = 160
    top_k_per_collection: int = 5
    max_context_chunks: int = 8
    index_schema_version: str = "v1"
    max_upload_size_mb: int = 25

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
