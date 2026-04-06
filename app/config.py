from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Buh Chat"
    debug: bool = False

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    embedding_model: str = Field(default="qwen3-embedding:0.6b", alias="OLLAMA_EMBEDDING_MODEL")
    answer_model: str = Field(default="qwen3:8b", alias="OLLAMA_CHAT_MODEL")
    answer_temperature: float = Field(default=0.1, alias="OLLAMA_TEMPERATURE")

    documents_dir: Path = BASE_DIR / "data" / "documents"
    storage_dir: Path = BASE_DIR / "storage"
    chroma_dir: Path = BASE_DIR / "storage" / "chroma"
    state_db_path: Path = BASE_DIR / "storage" / "state.db"

    allowed_extensions: tuple[str, ...] = (".pdf", ".xls", ".xlsx", ".docx", ".csv", ".txt")
    max_chunk_chars: int = 1400
    chunk_overlap_chars: int = 160
    top_k_per_collection: int = 5
    max_context_chunks: int = 8
    max_search_distance: float = 0.65
    max_distance_spread: float = 0.12
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
