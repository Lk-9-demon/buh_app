from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParsedChunk:
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class SearchHit:
    text: str
    metadata: dict[str, Any]
    distance: float
    collection_name: str


@dataclass(slots=True)
class FileRecord:
    path: str
    sha256: str
    modified_at: str
    indexed_at: str | None
    status: str
    chunk_count: int
    last_error: str | None
    schema_version: str
    embedding_model: str


@dataclass(slots=True)
class IndexingReport:
    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IndexingStatus:
    running: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    total_files: int = 0
    processed_files: int = 0
    current_file: str | None = None
    message: str | None = None
    report: IndexingReport = field(default_factory=IndexingReport)


@dataclass(slots=True)
class ChatAnswer:
    question: str
    answer: str
    sources: list[SearchHit]
